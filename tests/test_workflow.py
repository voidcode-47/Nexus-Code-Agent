"""端到端工作流测试（离线启发式修复）。"""

from __future__ import annotations

import asyncio
import time

import pytest

from src.agent.workflow import NexusWorkflow, WorkflowDeps

ISSUE = (
    "test_add 失败：AssertionError: assert -1 == 3\n"
    "Traceback (most recent call last):\n"
    '  File "tests/test_calculator.py", line 4, in test_add\n'
    "    assert add(1, 2) == 3\n"
    "AssertionError: assert -1 == 3\n"
    "疑似 add() 把加法写成了减法"
)


async def test_workflow_heals_single_bug(tmp_git_repo, worktree_root):
    deps = WorkflowDeps.from_repo(tmp_git_repo)
    workflow = NexusWorkflow(deps, worktree_root=worktree_root)
    state = await workflow.run(ISSUE, tmp_git_repo, task_id="t-heal")

    assert state["status"] == "SUCCESS"
    assert state["current_iteration"] == 1
    assert "return a + b" in state["final_diff"]
    assert state["termination_reason"]
    assert any(ev.get("node") == "tester" for ev in state["events"])
    # worktree 已在 finally 中清理
    assert not any(worktree_root.iterdir())


async def test_workflow_streams_nodes(tmp_git_repo, worktree_root):
    deps = WorkflowDeps.from_repo(tmp_git_repo)
    workflow = NexusWorkflow(deps, worktree_root=worktree_root)
    seen: list[str] = []
    async for chunk in workflow.astream(ISSUE, tmp_git_repo, task_id="t-stream"):
        seen.extend(chunk.keys())
    assert "localizer" in seen
    assert "coder" in seen
    assert "tester" in seen
    assert "critic" in seen


async def test_workflow_fallback_orchestrator(tmp_git_repo, worktree_root):
    """强制走无 LangGraph 的兜底编排路径。"""
    deps = WorkflowDeps.from_repo(tmp_git_repo)
    workflow = NexusWorkflow(deps, use_langgraph=False, worktree_root=worktree_root)
    state = await workflow.run(ISSUE, tmp_git_repo, task_id="t-fallback")
    assert state["status"] == "SUCCESS"
    assert "return a + b" in state["final_diff"]


async def test_workflow_escalates_when_unfixable(tmp_git_repo, worktree_root, git_commit):
    """无可修复候选时应升级人工介入（而非死循环）。"""
    # 把源码改成没有可用变异的形式：函数体返回常量
    calc = tmp_git_repo / "calculator.py"
    calc.write_text(
        "def add(a, b):\n    return 0\n\n\ndef subtract(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    (tmp_git_repo / "tests" / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    git_commit(tmp_git_repo, "make add unfixable by mutation")
    deps = WorkflowDeps.from_repo(tmp_git_repo)
    workflow = NexusWorkflow(deps, worktree_root=worktree_root)
    state = await workflow.run(ISSUE, tmp_git_repo, task_id="t-escalate", max_iterations=3)
    assert state["status"] in {"HUMAN_ESCALATION", "SUCCESS"}
    assert state["current_iteration"] >= 1


RECOVERABLE_ISSUE = (
    "test_offset 失败：AssertionError: assert 6 == 4\n"
    "Traceback (most recent call last):\n"
    '  File "tests/test_offset.py", line 4, in test_offset\n'
    "    assert offset() == 4\n"
    '  File "offset.py", line 2, in offset\n'
    "    return 5\n"
    "AssertionError: assert 6 == 4\n"
    "疑似 offset() 的返回值错误\n"
)


async def test_workflow_recovers_from_failed_test_round(tmp_git_repo, worktree_root, git_commit):
    """首轮补丁错误 → 真实 pytest 失败 → 熔断路由 → 下一候选修复成功。

    回归背景：失败测试输出含 "FAILURES" 段会触发 compress_test_output；
    旧实现会在该路径抛 AttributeError，使「测试失败后的处理链路」整体崩溃。
    """
    # 夹具自带的 calculator 缺陷会持续失败，替换为单缺陷场景
    (tmp_git_repo / "calculator.py").unlink()
    (tmp_git_repo / "tests" / "test_calculator.py").unlink()
    (tmp_git_repo / "offset.py").write_text(
        "def offset() -> int:\n    return 5\n", encoding="utf-8"
    )
    (tmp_git_repo / "tests" / "test_offset.py").write_text(
        "from offset import offset\n\n\ndef test_offset():\n    assert offset() == 4\n",
        encoding="utf-8",
    )
    git_commit(tmp_git_repo, "introduce offset bug")

    deps = WorkflowDeps.from_repo(tmp_git_repo)
    workflow = NexusWorkflow(deps, worktree_root=worktree_root)
    state = await workflow.run(RECOVERABLE_ISSUE, tmp_git_repo, task_id="t-recover")

    assert state["status"] == "SUCCESS"
    assert state["current_iteration"] >= 2
    history = state["patch_history"]
    assert any(not rec["test_passed"] for rec in history), "应存在一轮真实失败的测试"
    assert "return 4" in state["final_diff"]


async def test_workflow_closes_owned_deps(tmp_git_repo, worktree_root):
    """close 开关按所有权拆分：容器池与 LLM 连接分别释放。"""
    deps = WorkflowDeps.from_repo(tmp_git_repo)
    calls: list[tuple[bool, bool]] = []
    original_close = deps.aclose

    async def _spy(*, pool: bool = True, llm: bool = True) -> None:
        calls.append((pool, llm))
        await original_close(pool=False, llm=False)

    deps.aclose = _spy  # type: ignore[method-assign]
    workflow = NexusWorkflow(deps, worktree_root=worktree_root, close_pool=True, close_llm=True)
    state = await workflow.run(ISSUE, tmp_git_repo, task_id="t-close-deps")

    assert state["status"] == "SUCCESS"
    assert calls == [(True, True)]


async def test_workflow_shared_pool_only_closes_llm(tmp_git_repo, worktree_root):
    """共享池模式（API 长跑）：只关 LLM 客户端，绝不动容器池。"""
    deps = WorkflowDeps.from_repo(tmp_git_repo)
    calls: list[tuple[bool, bool]] = []
    original_close = deps.aclose

    async def _spy(*, pool: bool = True, llm: bool = True) -> None:
        calls.append((pool, llm))
        await original_close(pool=False, llm=False)

    deps.aclose = _spy  # type: ignore[method-assign]
    workflow = NexusWorkflow(deps, worktree_root=worktree_root, close_pool=False, close_llm=True)
    state = await workflow.run(ISSUE, tmp_git_repo, task_id="t-shared-pool")

    assert state["status"] == "SUCCESS"
    assert calls == [(False, True)]


def test_sync_container_mount_rejects_warmed_pool(tmp_path):
    """容器池已预热时不允许更改挂载根（已创建容器 bind 的是旧路径）。"""

    class _WarmPool:
        backend = "docker"

        def __init__(self) -> None:
            self.container_mount_root = tmp_path / "old-root"

        def has_active_containers(self) -> bool:
            return True

    deps = WorkflowDeps(pool=_WarmPool())  # type: ignore[arg-type]
    workflow = NexusWorkflow(deps, use_langgraph=False, worktree_root=tmp_path / "new-root")
    with pytest.raises(ValueError, match="已预热"):
        workflow._sync_container_mount()


def test_sync_container_mount_updates_root_before_warmup(tmp_path):
    """未预热（容器尚未创建）时允许对齐挂载根。"""

    class _ColdPool:
        backend = "docker"

        def __init__(self) -> None:
            self.container_mount_root = tmp_path / "old-root"

        def has_active_containers(self) -> bool:
            return False

    pool = _ColdPool()
    deps = WorkflowDeps(pool=pool)  # type: ignore[arg-type]
    workflow = NexusWorkflow(deps, use_langgraph=False, worktree_root=tmp_path / "new-root")
    workflow._sync_container_mount()
    assert pool.container_mount_root == (tmp_path / "new-root").resolve()


async def test_workflow_lsp_blocks_bad_patch(tmp_git_repo, worktree_root):
    """LSP 静态拦截路径：构造语法错误候选不应进入沙箱执行。"""
    from src.mcp_tools.lsp_bridge import LSPBridge

    bad = tmp_git_repo / "broken.py"
    bad.write_text("def f(:\n    pass\n", encoding="utf-8")
    report = await LSPBridge(pyright_cmd=None).check([bad], root=tmp_git_repo, use_pyright=False)
    assert not report.passed


async def test_workflow_auto_inits_plain_repo(tmp_path, worktree_root):
    """对非 Git 目录应自动初始化基线并完成修复（无需手工 git init）。"""
    repo = tmp_path / "plain_repo"
    repo.mkdir()
    (repo / "calculator.py").write_text(
        '"""示例计算器。"""\n\nfrom __future__ import annotations\n\n\n'
        "def add(a: int, b: int) -> int:\n"
        '    """返回两数之和。"""\n'
        "    return a - b\n",
        encoding="utf-8",
    )
    (repo / "conftest.py").write_text('"""test path bootstrap"""\n', encoding="utf-8")
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    assert not (repo / ".git").exists()

    deps = WorkflowDeps.from_repo(repo)
    workflow = NexusWorkflow(deps, worktree_root=worktree_root)
    state = await workflow.run(ISSUE, repo, task_id="t-autoinit")

    assert state["status"] == "SUCCESS"
    assert (repo / ".git").exists()  # 已自动初始化
    assert "return a + b" in state["final_diff"]


async def test_workflow_state_history_time_travel(tmp_git_repo, worktree_root):
    """启用 Checkpointer 后应可回溯每个节点的状态快照。"""
    from src.agent.workflow import build_checkpointer

    checkpointer = build_checkpointer()
    if checkpointer is None:  # pragma: no cover - langgraph 缺失
        return

    deps = WorkflowDeps.from_repo(tmp_git_repo)
    workflow = NexusWorkflow(deps, checkpointer=checkpointer, worktree_root=worktree_root)
    state = await workflow.run(ISSUE, tmp_git_repo, task_id="t-tt")
    assert state["status"] == "SUCCESS"

    history = await workflow.state_history("t-tt")
    assert history, "Checkpointer 应记录状态快照"
    statuses = {snap["status"] for snap in history}
    assert "SUCCESS" in statuses
    assert all(snap["checkpoint_id"] for snap in history)
    # 未启用 Checkpointer 的工作流返回空历史（不抛异常）
    bare = NexusWorkflow(WorkflowDeps.from_repo(tmp_git_repo))
    assert await bare.state_history("t-tt") == []


async def test_critic_escalates_after_replan_still_stuck(tmp_path):
    """重新规划后仍连续相同报错：熔断须升级人工（判据使用本轮新计数）。"""
    from src.agent.nodes.critic import critic_node
    from src.agent.state import error_fingerprint, initial_state

    trace = "Traceback (most recent call last):\nAssertionError: boom"
    state = initial_state("issue", str(tmp_path))
    state["current_iteration"] = 1
    state["replanned"] = True
    state["max_iterations"] = 10
    state["same_error_streak"] = 1
    state["last_fingerprint"] = error_fingerprint(trace)
    state["test_output"] = trace

    update = await critic_node(state)
    assert update["status"] == "HUMAN_ESCALATION"
    assert update["same_error_streak"] >= 2


async def test_coder_prefers_least_tried_candidates(tmp_path):
    """候选补丁调度优先「尝试次数最少」：仍有未试候选时不得重放已失败补丁。"""
    from src.agent.nodes.coder import coder_node
    from src.agent.state import initial_state

    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / "calc.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    def _patch(value: int) -> str:
        return (
            "<<<<<<< SEARCH\n    return 1\n=======\n"
            f"    return {value}\n>>>>>>> REPLACE"
        )

    p0, p1, p2 = _patch(2), _patch(3), _patch(4)
    state = initial_state("issue", str(worktree))
    state.update(
        {
            "worktree_path": str(worktree),
            "suspected_files": ["calc.py"],
            "candidate_patches": [p0, p1, p2],
            "current_iteration": 2,
            "patch_history": [
                {"iteration": 1, "diff_content": p0},
                {"iteration": 2, "diff_content": p2},
            ],
        }
    )

    update = await coder_node(state)
    # p0 / p2 已试过 → 本轮应选择从未试过的 p1，而不是按下标轮转重放 p0
    assert update["last_patch"] == p1
    assert update["candidate_patches"] == [p0, p1, p2]


async def test_coder_rejects_paths_escaping_worktree(tmp_path):
    """suspected_files 可能来自 LLM 输出：绝对路径 / .. 逃逸必须被拒绝。

    回归背景：旧实现用 ``(worktree / candidate).is_file()`` 做校验，而 pathlib
    在拼接绝对路径时会直接丢弃左侧根目录，导致 ``worktree / "/tmp/x.py"``
    等于 ``/tmp/x.py``——补丁会写到 Worktree 之外（沙箱隔离失效，且可用
    GitHub Issue 正文提示注入影响模型输出）。
    """
    from src.agent.nodes.coder import _select_target_file, coder_node
    from src.agent.state import initial_state

    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / "calc.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    victim = tmp_path / "outside.py"
    victim.write_text("def f():\n    return 1\n", encoding="utf-8")
    sibling = tmp_path / "sibling.py"
    sibling.write_text("def f():\n    return 1\n", encoding="utf-8")

    # 绝对路径与 .. 逃逸都必须被丢弃，回退到工作区内文件
    escaped = {"worktree_path": str(worktree), "suspected_files": [str(victim)]}
    assert _select_target_file(escaped, worktree) == "calc.py"
    relative_escape = {"worktree_path": str(worktree), "suspected_files": ["../sibling.py"]}
    assert _select_target_file(relative_escape, worktree) == "calc.py"
    # 缺少 worktree 上下文时不猜目标
    assert _select_target_file({"suspected_files": ["calc.py"]}, worktree) is None

    state = initial_state("issue", str(tmp_path))
    state.update({"worktree_path": str(worktree), "suspected_files": [str(victim)]})
    update = await coder_node(state)

    assert update["last_patch_target"] == "calc.py"
    assert update["changed_files"] == ["calc.py"]
    assert "return 2" in (worktree / "calc.py").read_text(encoding="utf-8")
    assert victim.read_text(encoding="utf-8") == "def f():\n    return 1\n"  # 未被改写


async def test_index_build_does_not_block_event_loop(tmp_git_repo, worktree_root, monkeypatch):
    """索引构建必须放到线程池，不能整段阻塞事件循环。

    回归背景：``WorkflowDeps.from_repo``（全仓 AST 解析 + 建图 + 向量化）是同步
    CPU/IO 操作，直接在事件循环里执行会卡住同进程的其他任务（实测 400 文件
    仓库约 1s，期间事件循环调度次数为 0），高并发下表现为「并发度上不去 /
    SSE 心跳停摆」。
    """
    original = WorkflowDeps.from_repo
    ticks = 0
    observed: dict[str, int] = {}

    def slow_from_repo(*args, **kwargs):  # type: ignore[no-untyped-def]
        observed["before"] = ticks
        time.sleep(0.4)  # 模拟大仓库索引耗时
        observed["after"] = ticks
        return original(*args, **kwargs)

    monkeypatch.setattr(WorkflowDeps, "from_repo", slow_from_repo)

    async def _ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    ticker = asyncio.create_task(_ticker())
    try:
        from src.mcp_tools.lsp_bridge import LSPBridge

        deps = WorkflowDeps(lsp=LSPBridge(pyright_cmd=None))  # 只跑语法引擎，保证测试离线可复现
        workflow = NexusWorkflow(deps, worktree_root=worktree_root)
        state = await workflow.run(ISSUE, tmp_git_repo, task_id="t-nonblock", max_iterations=1)
    finally:
        ticker.cancel()

    assert state["status"] in {"SUCCESS", "HUMAN_ESCALATION"}
    # 阻塞期间事件循环仍应被调度多次（直接同步调用时为 0）
    assert observed["after"] - observed["before"] >= 5, (
        f"索引构建阻塞了事件循环（期间仅 {observed['after'] - observed['before']} 次调度）"
    )
