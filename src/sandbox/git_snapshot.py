"""基于 Git Worktree 的多分支并发补丁测试与毫秒级回滚。

核心优势：

* **零拷贝并发隔离**：同一代码库共享 ``.git`` 对象库，为每个 Agent 探索分支创建
  独立 Worktree 目录，互不干扰，可并发试错；
* **毫秒级回滚**：``git reset --hard`` / ``git checkout -- .`` 秒级恢复干净状态；
* **异常安全清理**：通过异步上下文管理器 ``async with WorktreeSnapshot(...)`` 保证
  无论成功失败都回收 Worktree，并配合 ``git worktree prune`` 兜底清理残留锁文件。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from src.config import get_logger
from src.config.settings import settings as global_settings

logger = get_logger(__name__)


class GitError(RuntimeError):
    """Git 命令执行失败。"""


@dataclass(slots=True)
class GitResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


async def run_git(
    repo: str | Path,
    *args: str,
    check: bool = True,
    timeout: float = 60.0,
) -> GitResult:
    """在仓库中异步执行 git 命令。"""
    cmd = ["git", "-C", str(repo), *args]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"},
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError as exc:  # pragma: no cover - 极端场景
        proc.kill()
        raise GitError(f"git {' '.join(args)} timed out") from exc
    result = GitResult(
        proc.returncode or 0,
        stdout_b.decode("utf-8", "replace"),
        stderr_b.decode("utf-8", "replace"),
    )
    if check and not result.ok:
        raise GitError(
            f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result


@dataclass
class WorktreeSnapshot:
    """单个 Git Worktree 快照（异步上下文管理器）。

    典型用法::

        async with WorktreeSnapshot(repo, name="fix-1") as snap:
            await snap.apply_patch(patch_text)
            ...运行测试...
            await snap.reset()      # 毫秒级回滚到干净状态
    """

    repo_path: str | Path
    worktree_root: str | Path | None = None
    base_ref: str = "HEAD"
    name: str | None = None
    branch: str | None = None
    _dir: Path | None = field(default=None, init=False, repr=False)
    _created: bool = field(default=False, init=False)

    @property
    def path(self) -> Path:
        if self._dir is None:
            raise RuntimeError("worktree 尚未初始化，请使用 async with 上下文")
        return self._dir

    async def __aenter__(self) -> WorktreeSnapshot:
        await self.create()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.cleanup()

    # -- 生命周期 --------------------------------------------------------- #

    async def create(self) -> Path:
        repo = Path(self.repo_path).resolve()
        if not (repo / ".git").exists():
            raise GitError(f"not a git repository: {repo}")

        wt_name = self.name or f"nexus-{uuid.uuid4().hex[:8]}"
        root = (
            Path(self.worktree_root)
            if self.worktree_root
            else global_settings.resolve_path(global_settings.sandbox.worktree_root)
        )
        root = root if root.is_absolute() else (Path.cwd() / root)
        root.mkdir(parents=True, exist_ok=True)
        target = (root / wt_name).resolve()

        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

        # 确保存在至少一次提交
        head = await run_git(repo, "rev-parse", "--verify", self.base_ref, check=False)
        if not head.ok:
            # 该仓库没有任何提交：为建立 worktree 必须补一次基线提交。
            # 注意：这在语义上会把当前未提交的改动一起纳入 baseline，
            # 否则 worktree 中看不到待修复的缺陷代码。
            logger.warning(
                "目标仓库尚无任何提交，自动创建 baseline（包含当前未提交改动）",
                extra={"repo": str(repo)},
            )
            await run_git(repo, "add", "-A", check=False)
            await run_git(
                repo,
                "-c",
                "user.email=nexus@local",
                "-c",
                "user.name=nexus",
                "commit",
                "-m",
                "nexus: baseline snapshot",
                "--allow-empty",
                check=False,
            )

        if self.branch:
            await run_git(
                repo, "worktree", "add", "--force", "-B", self.branch, str(target), self.base_ref
            )
        else:
            await run_git(repo, "worktree", "add", "--detach", "--force", str(target), self.base_ref)
        self._dir = target
        self._created = True
        logger.info(
            "worktree created",
            extra={"path": str(target), "base": self.base_ref, "branch": self.branch or ""},
        )
        return target

    async def cleanup(self) -> None:
        if not self._created or self._dir is None:
            return
        repo = Path(self.repo_path).resolve()
        await run_git(repo, "worktree", "remove", "--force", str(self._dir), check=False)
        if self._dir.exists():  # pragma: no cover - 最后的兜底
            shutil.rmtree(self._dir, ignore_errors=True)
        await run_git(repo, "worktree", "prune", check=False)
        self._created = False
        logger.info("worktree cleaned", extra={"path": str(self._dir)})

    # -- 操作 ------------------------------------------------------------- #

    async def apply_patch(self, patch_text: str, *, fuzz: bool = True) -> None:
        """应用 Unified Diff 补丁（``git apply``）。"""
        flag = ["--recount"] if fuzz else []
        proc = await asyncio.create_subprocess_exec(
            "git",
            "apply",
            *flag,
            "--whitespace=nowarn",
            "-",
            cwd=str(self.path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate(patch_text.encode("utf-8"))
        if proc.returncode != 0:
            raise GitError(f"git apply failed: {stderr.decode('utf-8', 'replace')}")

    async def reset(self) -> None:
        """毫秒级硬回滚到基线状态。"""
        await run_git(self.path, "reset", "--hard", self.base_ref, check=False)
        await run_git(self.path, "clean", "-fdx", check=False)

    async def write_file(self, rel_path: str, content: str) -> Path:
        target = self.path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    async def read_file(self, rel_path: str) -> str:
        return (self.path / rel_path).read_text(encoding="utf-8")

    async def diff(self, *, staged: bool = False) -> str:
        args = ["diff", "--no-color"]
        if staged:
            args.append("--cached")
        result = await run_git(self.path, *args, check=False)
        return result.stdout

    async def changed_files(self) -> list[str]:
        result = await run_git(self.path, "status", "--porcelain", check=False)
        files: list[str] = []
        for line in result.stdout.splitlines():
            if len(line) > 3:
                files.append(line[3:].strip())
        return files

    async def commit(self, message: str, *, author: tuple[str, str] = ("nexus", "nexus@local")) -> str:
        await run_git(self.path, "add", "-A", check=False)
        await run_git(
            self.path,
            "-c",
            f"user.name={author[0]}",
            "-c",
            f"user.email={author[1]}",
            "commit",
            "-m",
            message,
            "--allow-empty",
            check=False,
        )
        head = await run_git(self.path, "rev-parse", "HEAD", check=False)
        return head.stdout.strip()

    async def export_patch(self, *, base: str | None = None) -> str:
        """导出相对基线的标准 Git Patch。"""
        result = await run_git(self.path, "diff", base or self.base_ref, "--no-color", check=False)
        return result.stdout


async def ensure_git_repo(path: str | Path) -> bool:
    """确保目录是 Git 仓库；若不是则初始化并创建基线提交。

    仅当目录**完全没有** ``.git`` 时才会初始化，绝不触碰已有仓库的历史。
    返回是否执行了初始化。
    """
    repo = Path(path)
    if (repo / ".git").exists():
        return False
    await run_git(repo, "init", check=False)
    await run_git(repo, "add", "-A", check=False)
    await run_git(
        repo,
        "-c", "user.email=nexus@local",
        "-c", "user.name=nexus",
        "commit", "-m", "nexus: baseline snapshot", "--allow-empty",
        check=False,
    )
    logger.warning("目标目录非 Git 仓库，已自动初始化基线提交", extra={"path": str(repo)})
    return True


async def reset_workdir(path: str | Path) -> None:
    """将 Worktree 工作区恢复到基线（丢弃未提交改动）——毫秒级回滚。"""
    await run_git(path, "checkout", "--", ".", check=False)
    await run_git(path, "clean", "-fd", check=False)


async def worktree_diff(path: str | Path, *, base: str = "HEAD") -> str:
    """返回 Worktree 相对基线的差异（标准 Git Patch）。"""
    result = await run_git(path, "diff", base, "--no-color", check=False)
    return result.stdout


async def worktree_changed_files(path: str | Path) -> list[str]:
    """返回 Worktree 中已变更的文件列表（忽略缓存等运行时产物）。"""
    result = await run_git(path, "status", "--porcelain", check=False)
    ignored = ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache")
    files: list[str] = []
    for line in result.stdout.splitlines():
        if len(line) <= 3:
            continue
        name = line[3:].strip()
        if name.endswith(".pyc") or any(token in name for token in ignored):
            continue
        files.append(name)
    return files


#: 锁文件失龄阈值：仅清理明显陈旧的 ``*.lock``，避免误删进行中操作持有的活锁
_LOCK_STALE_S = 3600.0


async def prune_worktrees(repo: str | Path) -> None:
    """兜底清理残留 Worktree 引用与**陈旧**锁文件（高并发异常退出后调用）。"""
    repo_path = Path(repo)
    await run_git(repo_path, "worktree", "prune", check=False)
    # 仅删除超过 _LOCK_STALE_S 未更新的 *.lock：index.lock / HEAD.lock 等
    # 是 git 并发操作的活锁，无差别删除会破坏其它进行中的命令。
    git_dir = repo_path / ".git"
    if git_dir.is_dir():
        now = time.time()
        for lock in git_dir.rglob("*.lock"):
            try:
                if now - lock.stat().st_mtime < _LOCK_STALE_S:
                    continue
                lock.unlink()
            except OSError:  # pragma: no cover
                continue
