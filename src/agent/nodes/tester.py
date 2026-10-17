"""沙箱动态验证节点：LSP 静态拦截 + pytest 动态测试（双层校验）。

流程（对应文档 §2 工作流第 4~5 步）：

1. 补丁应用后**立即**触发 LSP 静态检查（秒级），未通过则直接打回，不消耗单测算力；
2. 静态检查通过后，从容器池获取预热沙箱执行相关 pytest 套件；
3. 输出经正则压缩后只保留关键 Traceback，防止上下文膨胀。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agent.state import AgentState, event
from src.config import get_logger
from src.mcp_tools.lsp_bridge import LSPBridge
from src.sandbox.pool_manager import PoolManager

logger = get_logger(__name__)


async def tester_node(
    state: AgentState,
    *,
    pool: PoolManager,
    lsp: LSPBridge | None = None,
) -> dict[str, Any]:
    """执行静态与动态双层校验。"""
    worktree = Path(state.get("worktree_path", ""))
    changed = state.get("changed_files", [])

    if not state.get("patch_applied", False):
        reason = state.get("test_output") or "补丁未能应用到目标文件"
        return {
            "status": "CRITIQUING",
            "lsp_passed": False,
            "test_passed": False,
            "test_output": reason,
            "events": [event("tester", "补丁未应用，跳过校验", level="warning")],
        }

    # 1) LSP 静态拦截
    bridge = lsp or LSPBridge()
    abs_files = [str((worktree / f).resolve()) if not Path(f).is_absolute() else f for f in changed]
    lsp_report = await bridge.check(abs_files, root=worktree)
    if not lsp_report.passed:
        summary = lsp_report.summary()
        logger.info("LSP 拦截：跳过沙箱测试", extra={"errors": lsp_report.error_count})
        return {
            "status": "CRITIQUING",
            "lsp_passed": False,
            "test_passed": False,
            "test_output": summary,
            "events": [
                event(
                    "tester",
                    f"LSP 拦截 {lsp_report.error_count} 个错误，未消耗单测算力",
                    engine=lsp_report.engine,
                    diagnostics=[d.render() for d in lsp_report.diagnostics[:10]],
                )
            ],
        }

    # 2) 沙箱动态测试
    targets = _test_targets(worktree)
    async with pool.session(worktree) as session:
        result = await session.run_pytest(targets)
    summary = result.summary()
    logger.info(
        "沙箱测试完成",
        extra={"ok": result.ok, "exit": result.returncode, "duration_s": round(result.duration_s, 2)},
    )
    return {
        "status": "CRITIQUING",
        "lsp_passed": True,
        "test_passed": result.ok,
        "test_output": summary,
        "events": [
            event(
                "tester",
                f"沙箱测试 {'通过' if result.ok else '失败'}（exit={result.returncode}, {result.duration_s:.1f}s）",
                returncode=result.returncode,
                timed_out=result.timed_out,
                oom=result.oom,
                backend=pool.backend,
                output=summary[:4000],
            )
        ],
    }


def _test_targets(worktree: Path) -> list[str]:
    """有 tests/ 目录则只跑该目录，否则在仓库根目录做全量收集。"""
    return ["tests"] if (worktree / "tests").is_dir() else ["."]
