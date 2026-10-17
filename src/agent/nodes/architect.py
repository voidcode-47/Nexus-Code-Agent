"""修复方案规划与多文件影响评估 Agent。

分析调用图上下游影响面，产出分步修改计划（Thought & Plan）。启用真实 LLM 时
由模型规划；离线时给出确定性的、可执行的最小改动计划。
"""

from __future__ import annotations

from typing import Any

from src.agent.llm import LLMClient
from src.agent.prompts import architect_messages
from src.agent.state import AgentState, add_tokens, event
from src.config import get_logger

logger = get_logger(__name__)


async def architect_node(
    state: AgentState,
    *,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """产出分步修复计划。"""
    issue = state.get("issue_description", "")
    files = state.get("suspected_files", [])
    symbols = state.get("suspected_symbols", [])
    context = state.get("call_graph_context", "")
    previous = state.get("repair_plan", "")

    tokens = 0
    plan = ""
    if llm is not None and llm.enabled:
        try:
            messages = architect_messages(issue, _localized_text(files, symbols), context)
            plan = (await llm.complete(messages)).strip()
            tokens = add_tokens(state, plan)
        except Exception as exc:  # pragma: no cover
            logger.warning("architect LLM 调用失败", extra={"error": str(exc)})

    if not plan:
        plan = _deterministic_plan(issue, files, symbols, previous)

    update: dict[str, Any] = {
        "repair_plan": plan,
        "status": "CODING",
        "events": [event("architect", plan[:800], plan=plan)],
    }
    if tokens:
        update["token_usage"] = tokens
    logger.info("architect 完成", extra={"plan_chars": len(plan)})
    return update


def _localized_text(files: list[str], symbols: list[str]) -> str:
    return f"files={files}\nsymbols={symbols}"


def _deterministic_plan(
    issue: str,
    files: list[str],
    symbols: list[str],
    previous: str,
) -> str:
    target = files[0] if files else "(未知文件)"
    fn = symbols[0] if symbols else "(未知符号)"
    revision_note = "\n> 注意：上一轮计划未通过测试，本次请更换修改思路（如调整运算符/边界条件）。" if previous else ""
    return (
        f"目标：修复 Issue「{issue.strip() or '未提供描述'}」。\n"
        f"步骤：\n"
        f"1. 定位到 `{target}` 中的 `{fn}`，确认缺陷语义；\n"
        f"2. 对该函数实施最小必要改动（优先检查运算符、比较符、边界常量与返回值）；\n"
        f"3. 保持签名与既有调用方不变，仅修改函数体；\n"
        f"4. 通过 LSP 静态检查后，运行相关 pytest 用例验证。"
        f"{revision_note}"
    )
