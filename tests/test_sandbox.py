"""沙箱执行器、Git Worktree 与容器池测试。"""

from __future__ import annotations

import asyncio
import sys

import pytest

from src.sandbox.executor import (
    compress_test_output,
    extract_frames,
    extract_traceback,
    fingerprint_error,
    run_command,
)
from src.sandbox.git_snapshot import WorktreeSnapshot, ensure_git_repo, prune_worktrees, run_git
from src.sandbox.pool_manager import PoolManager, SandboxSession

PY = sys.executable

# --------------------------------------------------------------------------- #
# executor
# --------------------------------------------------------------------------- #


async def test_run_command_success():
    result = await run_command([PY, "-c", "print('hello')"], timeout=30)
    assert result.ok
    assert "hello" in result.stdout


async def test_run_command_exit_code():
    result = await run_command([PY, "-c", "import sys; sys.exit(3)"], timeout=30)
    assert result.returncode == 3
    assert not result.ok


async def test_run_command_timeout_kills():
    result = await run_command(
        [PY, "-c", "import time; time.sleep(30)"],
        timeout=1.5,
        poll_interval=0.2,
    )
    assert result.timed_out
    assert not result.ok


async def test_run_command_env_and_cwd(tmp_path):
    result = await run_command(
        [PY, "-c", "import os; print(os.getenv('NEXUS_TEST')); print(os.getcwd())"],
        cwd=tmp_path,
        env={"NEXUS_TEST": "42"},
        timeout=30,
    )
    assert "42" in result.stdout
    assert str(tmp_path).split("\\")[-1].split("/")[-1] in result.stdout


async def test_run_command_rejects_shell_with_sequence():
    """序列命令 + shell=True 会丢失引号语义，必须显式报错。"""
    with pytest.raises(ValueError, match="shell=True"):
        await run_command([PY, "-c", "print(1)"], shell=True)


async def test_run_command_stall_grace_when_grandchild_holds_pipe():
    """父进程退出但孙进程仍持有 stdout 管道时，stall_grace 兜底避免永久挂起。"""
    code = (
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])\n"
        "print('done')\n"
    )
    result = await run_command(
        [PY, "-c", code], timeout=30, poll_interval=0.05, stall_grace_s=1.0
    )
    # 父进程本身已正常退出（exit=0）；读取被兜底终止 → 标记结果不完整
    assert result.returncode == 0
    assert result.timed_out is True
    assert "done" in result.stdout


async def test_run_command_truncation_flag_on_single_chunk_overflow():
    """单块输出超出上限且随后流结束：截断标志必须置位（旧实现漏报）。"""
    result = await run_command([PY, "-c", "print('x' * 500)"], max_output_bytes=100, timeout=30)
    assert result.ok
    assert result.truncated is True
    assert len(result.stdout) <= 100


def test_extract_traceback():
    text = (
        "Traceback (most recent call last):\n"
        '  File "/a/b.py", line 10, in foo\n'
        "    bar()\n"
        '  File "/a/c.py", line 20, in bar\n'
        "    raise ValueError('boom')\n"
        "ValueError: boom\n"
    )
    tb = extract_traceback(text)
    assert "ValueError: boom" in tb
    assert "b.py" in tb


def test_extract_frames():
    text = '  File "/a/b.py", line 10, in foo\n  File "/a/c.py", line 20, in bar\n'
    frames = extract_frames(text)
    assert ("/a/b.py", 10, "foo") in frames
    assert ("/a/c.py", 20, "bar") in frames


def test_compress_short_clean_output_passthrough():
    """短且无 FAILURES 的输出走快速通道，原样返回。"""
    text = "========================= 3 passed in 0.02s ==========================\n"
    assert compress_test_output(text) == text


def test_compress_real_pytest_failure_output():
    """真实 pytest 失败输出（含 FAILURES 段）必须走压缩路径且不崩溃。

    回归背景：_PYTEST_SUMMARY_RE 含两个捕获组，findall 返回元组，
    旧实现直接对元组 .strip() 会在失败路径抛 AttributeError。
    """
    text = (
        "============================= test session starts =============================\n"
        "platform win32 -- Python 3.11.9, pytest-8.2.0\n"
        "collected 3 items\n"
        "\n"
        "tests/test_calculator.py F..                                             [ 66%]\n"
        "\n"
        "================================== FAILURES ===================================\n"
        "__________________________________ test_add ___________________________________\n"
        "\n"
        "    def test_add():\n"
        ">       assert add(1, 2) == 3\n"
        "E       assert -1 == 3\n"
        "\n"
        "tests/test_calculator.py:4: AssertionError\n"
        "=========================== short test summary info ===========================\n"
        "FAILED tests/test_calculator.py::test_add - assert -1 == 3\n"
        "========================= 1 failed, 2 passed in 0.12s =========================\n"
    )
    compressed = compress_test_output(text)
    assert "FAILED tests/test_calculator.py::test_add" in compressed
    assert "assert -1 == 3" in compressed
    # 总结行（两个捕获组的正则）必须被正确提取
    assert "Summary: 1 failed, 2 passed in 0.12s" in compressed
    assert len(compressed) < len(text)


def test_compress_handles_long_output_with_sections():
    """超过 4000 字符的输出同样走压缩路径（长日志场景）。"""
    text = (
        "================================== FAILURES ===================================\n"
        "FAILED tests/test_x.py::test_a - assert 1 == 2\n"
        "FAILED tests/test_x.py::test_b - assert 1 == 3\n"
        "=================== 2 failed, 0 passed in 0.10s ===================\n"
        + ("noise line\n" * 600)
    )
    compressed = compress_test_output(text)
    assert "tests/test_x.py::test_a" in compressed
    assert "tests/test_x.py::test_b" in compressed
    assert len(compressed) < 2000


def test_fingerprint_stable_across_line_numbers():
    a = 'Traceback (most recent call last):\n  File "x.py", line 10, in f\nValueError: bad\n'
    b = 'Traceback (most recent call last):\n  File "x.py", line 99, in f\nValueError: bad\n'
    assert fingerprint_error(a) == fingerprint_error(b)


def test_fingerprint_differs_for_different_errors():
    a = 'Traceback (most recent call last):\n  File "x.py", line 10, in f\nValueError: bad\n'
    b = 'Traceback (most recent call last):\n  File "y.py", line 10, in g\nKeyError: nope\n'
    assert fingerprint_error(a) != fingerprint_error(b)


def test_command_result_summary():
    import asyncio

    result = asyncio.run(run_command([PY, "-c", "print('ok')"], timeout=30))
    assert "ok" in result.summary()


# --------------------------------------------------------------------------- #
# git worktree
# --------------------------------------------------------------------------- #


async def test_worktree_create_apply_reset(tmp_git_repo, worktree_root, git_commit):
    # 提交 .gitignore 让 *.tmp 成为被忽略文件：reset 必须连 ignored 文件一起清理（clean -fdx）
    (tmp_git_repo / ".gitignore").write_text("*.tmp\n", encoding="utf-8")
    git_commit(tmp_git_repo)
    async with WorktreeSnapshot(tmp_git_repo, worktree_root=worktree_root, name="wt1") as snap:
        assert snap.path.exists()
        assert (snap.path / "calculator.py").exists()

        target = snap.path / "calculator.py"
        await snap.write_file("calculator.py", target.read_text(encoding="utf-8").replace("a - b", "a + b"))
        (snap.path / "scratch.tmp").write_text("junk", encoding="utf-8")
        changed = await snap.changed_files()
        assert "calculator.py" in changed

        await snap.reset()
        assert (snap.path / "calculator.py").read_text(encoding="utf-8").count("a - b") >= 1
        assert not (snap.path / "scratch.tmp").exists()

    assert not (worktree_root / "wt1").exists()


async def test_worktree_export_patch(tmp_git_repo, worktree_root):
    async with WorktreeSnapshot(tmp_git_repo, worktree_root=worktree_root, name="wt2") as snap:
        content = (snap.path / "calculator.py").read_text(encoding="utf-8")
        await snap.write_file("calculator.py", content.replace("return a - b", "return a + b", 1))
        patch = await snap.export_patch()
        assert "return a + b" in patch


async def test_prune_worktrees(tmp_git_repo, worktree_root):
    snap = WorktreeSnapshot(tmp_git_repo, worktree_root=worktree_root, name="wt3")
    await snap.create()
    await prune_worktrees(tmp_git_repo)
    await snap.cleanup()


async def test_worktree_requires_git_repo(tmp_path):
    from src.sandbox.git_snapshot import GitError

    with pytest.raises(GitError):
        async with WorktreeSnapshot(tmp_path / "not_a_repo"):
            pass


async def test_ensure_git_repo_initializes_plain_dir(tmp_path):
    repo = tmp_path / "plain"
    repo.mkdir()
    (repo / "hello.py").write_text("print('hi')\n", encoding="utf-8")

    assert not (repo / ".git").exists()
    assert await ensure_git_repo(repo) is True
    assert (repo / ".git").exists()

    # 基线提交应包含工作区文件
    async with WorktreeSnapshot(repo, worktree_root=tmp_path / "wt", name="wt") as snap:
        assert (snap.path / "hello.py").exists()


async def test_ensure_git_repo_is_idempotent(tmp_git_repo):
    # 已是仓库：不应再初始化，也不应改动其历史
    before = await run_git(tmp_git_repo, "rev-parse", "HEAD")
    assert await ensure_git_repo(tmp_git_repo) is False
    after = await run_git(tmp_git_repo, "rev-parse", "HEAD")
    assert before.stdout == after.stdout


async def test_worktree_on_named_branch(tmp_git_repo, worktree_root):
    """具名分支 Worktree：在独立分支上提交，源仓库工作区不受影响。"""
    async with WorktreeSnapshot(
        tmp_git_repo, worktree_root=worktree_root, name="wt-branch", branch="nexus/fix-test"
    ) as snap:
        await snap.write_file("calculator.py", "def add(a, b):\n    return a + b\n")
        commit = await snap.commit("fix add")
        assert commit

    # 分支引用仍存在且指向该提交
    verify = await run_git(
        tmp_git_repo, "rev-parse", "--verify", "refs/heads/nexus/fix-test", check=False
    )
    assert verify.ok
    assert verify.stdout.strip() == commit
    # 源仓库工作区保持干净
    status = await run_git(tmp_git_repo, "status", "--porcelain", check=False)
    assert status.stdout.strip() == ""


# --------------------------------------------------------------------------- #
# pool manager
# --------------------------------------------------------------------------- #


def test_pool_backend_local_by_default(monkeypatch):
    monkeypatch.setenv("NEXUS_SANDBOX__BACKEND", "local")
    from src.config.settings import SandboxSettings

    pool = PoolManager(settings=SandboxSettings(backend="local"))
    assert pool.backend == "local"


async def test_local_session_exec(tmp_path):
    from src.config.settings import SandboxSettings

    pool = PoolManager(settings=SandboxSettings(backend="local", command_timeout_s=30))
    async with pool.session(tmp_path) as session:
        assert isinstance(session, SandboxSession)
        result = await session.exec([PY, "-c", "print('sandbox')"])
    assert "sandbox" in result.stdout


async def test_local_session_run_pytest(tmp_git_repo):
    from src.config.settings import SandboxSettings

    pool = PoolManager(settings=SandboxSettings(backend="local", test_timeout_s=120))
    async with pool.session(tmp_git_repo) as session:
        result = await session.run_pytest(["tests"])
    # 仓库内含加法缺陷，测试应当失败
    assert not result.ok


def test_run_pytest_uses_container_python_in_docker_backend(tmp_path):
    """docker 后端必须用镜像内 python：宿主路径在容器里不存在（回归护栏）。"""
    from src.config.settings import SandboxSettings

    pool = PoolManager(settings=SandboxSettings(backend="docker"))
    pool._backend = "docker"
    session = SandboxSession(workdir=tmp_path, pool=pool)
    assert session._python_executable() == "python"

    pool._backend = "local"
    assert session._python_executable() == PY


async def test_container_pool_bounded_by_pool_size():
    """容器总数不得超过 pool_size：队列空且已达上限时等待释放，而不是新建。"""
    from src.config.settings import SandboxSettings

    pool = PoolManager(settings=SandboxSettings(backend="docker", pool_size=1))
    pool._backend = "docker"
    pool._queue = asyncio.Queue()

    created: list[object] = []

    def _fake_create() -> object:
        obj = object()
        created.append(obj)
        return obj

    pool._create_container = _fake_create  # type: ignore[method-assign]
    first = await pool._acquire_container()
    assert created == [first]

    waiting = asyncio.create_task(pool._acquire_container())
    await asyncio.sleep(0.05)
    assert not waiting.done()  # 已达上限：挂起等待空闲容器

    await pool._release_container(first)
    second = await asyncio.wait_for(waiting, timeout=1.0)
    assert second is first
    assert len(created) == 1  # 未突破上限新建容器


async def test_container_pool_wait_timeout_raises():
    """等待空闲容器必须有上限：超时快速失败（RuntimeError），而不是永久挂起。"""
    from src.config.settings import SandboxSettings

    pool = PoolManager(settings=SandboxSettings(backend="docker", pool_size=1))
    pool._backend = "docker"
    pool._queue = asyncio.Queue()
    pool.settings.acquire_timeout_s = 0.05  # 本实例私有，测试后无需恢复

    created: list[object] = []

    def _fake_create() -> object:
        obj = object()
        created.append(obj)
        return obj

    pool._create_container = _fake_create  # type: ignore[method-assign]
    held = await pool._acquire_container()

    with pytest.raises(RuntimeError, match="空闲沙箱容器"):
        await pool._acquire_container()
    assert len(created) == 1  # 超时路径不新建容器（池上限仍成立）

    await pool._release_container(held)
    assert await asyncio.wait_for(pool._acquire_container(), timeout=1.0) is held


def test_pool_refuses_silent_local_degrade_in_prod(monkeypatch):
    """prod + backend=docker 且 daemon 不可用时必须拒绝启动，不得静默降级。

    回归背景：``validate_production_config`` 只校验**配置值**为 docker，
    而 ``_resolve_backend`` 在 daemon 不可用时静默回退 local——被测仓库
    （可来自外部 Issue / PR）于是在生产环境以宿主权限执行，隔离边界失效
    且没有任何报错。
    """
    from src.config.settings import SandboxSettings, settings
    from src.sandbox import pool_manager as pm

    class _BrokenDocker:
        @staticmethod
        def from_env():  # type: ignore[no-untyped-def]
            raise RuntimeError("daemon down")

    monkeypatch.setattr(pm, "docker", _BrokenDocker)
    monkeypatch.setattr(settings, "env", "prod")

    with pytest.raises(RuntimeError, match="拒绝静默降级"):
        PoolManager(settings=SandboxSettings(backend="docker"))

    # 显式认领风险后允许降级（与 validate_production_config 的口径一致）
    pool = PoolManager(settings=SandboxSettings(backend="docker", allow_local_in_prod=True))
    assert pool.backend == "local"

    # 非生产环境保持优雅降级，不影响本地开发
    monkeypatch.setattr(settings, "env", "test")
    assert PoolManager(settings=SandboxSettings(backend="docker")).backend == "local"

    # 未要求 docker 时不受影响
    monkeypatch.setattr(settings, "env", "prod")
    assert PoolManager(settings=SandboxSettings(backend="local")).backend == "local"
