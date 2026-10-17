"""报错堆栈归因与死循环熔断判别器。

职责（对应文档 §4 防劣化机制）：

* **归因**：解析 Traceback，判断根因类别并给出下一步建议；
* **死循环熔断**：计算报错堆栈 Hash 指纹，若连续 N 次相同报错或修改同一代码行
  无进展，则强制切换路由（回退 Architect 重新规划）；超过最大迭代阈值则升级人工介入；
* **终局**：测试通过时导出标准 Git Patch，流程结束。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from src.agent.llm import LLMClient, parse_json_block
from src.agent.prompts import critic_messages
from src.agent.state import (
    AgentState,
    add_tokens,
    error_fingerprint,
    event,
    fold_patch_history,
    record,
    register_failure,
    should_escalate,
)
from src.config import get_logger
from src.config.settings import settings as global_settings
from src.sandbox.executor import extract_traceback
from src.sandbox.git_snapshot import worktree_changed_files, worktree_diff

logger = get_logger(__name__)


async def critic_node(
    state: AgentState,
    *,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """归因 + 熔断判别 + 路由决策。"""
    iteration = int(state.get("current_iteration", 0)) + 1
    passed = bool(state.get("test_passed", False))
    worktree = Path(state.get("worktree_path", ""))
    test_output = state.get("test_output", "")

    patch = state.get("last_patch", "")
    history = state.get("patch_history", [])
    stalled = bool(history) and history[-1].get("diff_content") == patch

    fingerprint = error_fingerprint(test_output) if not passed else ""
    traceback = extract_traceback(test_output)
    counters = register_failure(state, fingerprint, stalled) if not passed else {
        "same_error_streak": 0,
        "stall_streak": 0,
    }

    tokens = 0
    attribution = "测试通过，修复成功。"
    if not passed and llm is not None and llm.enabled:
        try:
            _, history_summary = fold_patch_history(history)
            raw = await llm.complete(
                critic_messages(traceback, history_summary, iteration), json_mode=True
            )
            payload = parse_json_block(raw)
            attribution = payload.get("root_cause") or attribution
            tokens = add_tokens(state, raw)
        except Exception as exc:  # pragma: no cover
            logger.warning("critic LLM 调用失败", extra={"error": str(exc)})
    elif not passed:
        attribution = _heuristic_attribution(traceback or test_output)

    record_entry = record(
        iteration,
        diff_content=patch,
        lsp_passed=bool(state.get("lsp_passed", False)),
        test_passed=passed,
        error_summary=test_output,
        error_text=test_output,
        strategy=state.get("last_patch_target", ""),
    )

    update: dict[str, Any] = {
        "current_iteration": iteration,
        "last_fingerprint": fingerprint,
        "patch_history": [record_entry],
        "events": [
            event(
                "critic",
                attribution,
                iteration=iteration,
                test_passed=passed,
                fingerprint=fingerprint,
                **counters,
            )
        ],
        **counters,
    }
    if tokens:
        update["token_usage"] = tokens

    if passed:
        diff = await worktree_diff(worktree) if worktree.exists() else ""
        changed = await worktree_changed_files(worktree) if worktree.exists() else []
        update.update(
            {
                "status": "SUCCESS",
                "final_diff": diff,
                "changed_files": changed or state.get("changed_files", []),
                "termination_reason": f"修复成功（第 {iteration} 轮，{len(changed)} 个文件变更）",
            }
        )
        logger.info("critic: 修复成功", extra={"iteration": iteration, "files": changed})
        return update

    cfg = global_settings.agent
    # 熔断判据必须基于本轮刚算出的计数：state 中的 streak 还是上一轮的旧值，
    # 直接传给 should_escalate 会依赖第二处同步检查（脆弱），这里先合并再判断。
    effective = cast(AgentState, {**state, **counters})
    escalate, reason = should_escalate(effective, reason="")
    same_streak = int(counters["same_error_streak"])
    stall_streak = int(counters["stall_streak"])
    streak_exceeded = same_streak >= cfg.same_error_streak or stall_streak >= cfg.stall_streak
    if iteration >= int(state.get("max_iterations", cfg.max_iterations)):
        update.update(
            {
                "status": "HUMAN_ESCALATION",
                "termination_reason": f"达到最大迭代阈值（{iteration}），升级人工介入",
            }
        )
    elif streak_exceeded and not state.get("replanned"):
        update.update(
            {
                "status": "PLANNING",
                "same_error_streak": 0,
                "stall_streak": 0,
                "replanned": True,
                "termination_reason": "熔断触发：连续相同报错/无进展，回退 Architect 重新规划",
            }
        )
    elif escalate:
        update.update(
            {
                "status": "HUMAN_ESCALATION",
                "termination_reason": reason
                or "熔断触发：重新规划后仍连续出现相同报错/无进展，升级人工介入",
            }
        )
    else:
        update.update({"status": "CODING"})

    logger.info(
        "critic 路由决策",
        extra={
            "iteration": iteration,
            "status": update["status"],
            "same_error_streak": counters["same_error_streak"],
            "stall_streak": counters["stall_streak"],
        },
    )
    return update


def _heuristic_attribution(text: str) -> str:
    lowered = text.lower()
    if "syntaxerror" in lowered or "indentationerror" in lowered:
        return "根因类别：语法错误（补丁破坏了代码结构）"
    if "modulenotfounderror" in lowered or "importerror" in lowered:
        return "根因类别：导入错误（依赖或路径解析失败）"
    if "assertionerror" in lowered or "assert" in lowered:
        return "根因类别：逻辑错误（断言未通过，修复方向不正确）"
    if "typeerror" in lowered:
        return "根因类别：类型错误（参数或返回值类型不匹配）"
    if "timeout" in lowered:
        return "根因类别：执行超时（疑似死循环或测试过重）"
    return "根因类别：未归类（需人工或更换修复策略）"
