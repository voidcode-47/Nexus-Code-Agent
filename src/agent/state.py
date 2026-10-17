"""强类型状态定义、Token 计数器与滑动窗口裁剪器。

强类型状态设计是保障多智能体协同「不乱套」的核心（对应文档第 4 节）。
除文档定义的字段外，补充了运行期所需的临时字段（补丁内容、LSP 报告、
测试输出、熔断计数等），所有字段均可选，便于节点返回「局部更新」。
"""

from __future__ import annotations

import operator
import re
import uuid
from typing import Annotated, Any, Literal, TypedDict

from src.config.settings import settings as global_settings
from src.sandbox.executor import fingerprint_error

AgentStatus = Literal[
    "LOCATING",
    "PLANNING",
    "CODING",
    "TESTING",
    "CRITIQUING",
    "SUCCESS",
    "HUMAN_ESCALATION",
]


class PatchRecord(TypedDict, total=False):
    """单次迭代的补丁与校验记录。"""

    iteration: int
    diff_content: str
    lsp_passed: bool
    test_passed: bool
    error_fingerprint: str  # 报错堆栈特征哈希，用于检测死循环
    error_summary: str
    strategy: str


class AgentState(TypedDict, total=False):
    """自愈智能体的全局状态。"""

    # --- 任务标识 ---
    task_id: str
    issue_description: str
    repo_path: str
    worktree_id: str
    worktree_path: str

    # --- 检索与规划上下文 ---
    suspected_files: list[str]
    suspected_symbols: list[str]
    call_graph_context: str
    repair_plan: str

    # --- 迭代与熔断控制 ---
    current_iteration: int
    max_iterations: int
    patch_history: Annotated[list[PatchRecord], operator.add]
    token_usage: int
    same_error_streak: int
    stall_streak: int
    last_fingerprint: str
    replanned: bool

    # --- 运行期临时字段 ---
    candidate_patches: list[str]
    last_patch: str
    last_patch_target: str
    patch_applied: bool
    changed_files: list[str]
    lsp_passed: bool
    test_passed: bool
    test_output: str
    final_diff: str
    termination_reason: str

    # --- 路由状态 ---
    status: AgentStatus
    events: Annotated[list[dict[str, Any]], operator.add]


_TOKEN_DIVISOR = 4  # 英文/代码近似：1 token ≈ 4 字符


def count_tokens(text: str) -> int:
    """粗略估算 token 数（离线可用，避免强依赖 tokenizer）。"""
    if not text:
        return 0
    # 中文按 1 字 ≈ 1 token，其余按 4 字符 ≈ 1 token
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = len(text) - cjk
    return cjk + max(1, other // _TOKEN_DIVISOR)


def error_fingerprint(text: str) -> str:
    """对报错输出计算稳定指纹（委托自 sandbox.executor）。"""
    return fingerprint_error(text)


def initial_state(
    issue_description: str,
    repo_path: str,
    *,
    task_id: str | None = None,
    max_iterations: int | None = None,
) -> AgentState:
    """构造初始状态。"""
    return AgentState(
        task_id=task_id or f"task-{uuid.uuid4().hex[:12]}",
        issue_description=issue_description,
        repo_path=repo_path,
        worktree_id="",
        worktree_path="",
        suspected_files=[],
        suspected_symbols=[],
        call_graph_context="",
        repair_plan="",
        current_iteration=0,
        max_iterations=max_iterations or global_settings.agent.max_iterations,
        patch_history=[],
        token_usage=0,
        same_error_streak=0,
        stall_streak=0,
        last_fingerprint="",
        replanned=False,
        candidate_patches=[],
        last_patch="",
        last_patch_target="",
        patch_applied=False,
        changed_files=[],
        lsp_passed=False,
        test_passed=False,
        test_output="",
        final_diff="",
        termination_reason="",
        status="LOCATING",
        events=[],
    )


def record(
    iteration: int,
    *,
    diff_content: str = "",
    lsp_passed: bool = False,
    test_passed: bool = False,
    error_summary: str = "",
    error_text: str = "",
    strategy: str = "",
) -> PatchRecord:
    """构造一条补丁历史记录。"""
    return PatchRecord(
        iteration=iteration,
        diff_content=diff_content,
        lsp_passed=lsp_passed,
        test_passed=test_passed,
        error_fingerprint=error_fingerprint(error_text) if error_text else "",
        error_summary=error_summary[:2000],
        strategy=strategy,
    )


def event(node: str, message: str, **extra: Any) -> dict[str, Any]:
    """构造一条 SSE 事件。"""
    return {"node": node, "message": message, **extra}


def add_tokens(state: AgentState, *texts: str) -> int:
    """累加 token 用量，返回新的总量。"""
    delta = sum(count_tokens(t) for t in texts)
    return int(state.get("token_usage", 0)) + delta


def fold_patch_history(
    history: list[PatchRecord],
    *,
    keep_recent: int = 3,
    max_summary_chars: int = 1500,
) -> tuple[list[PatchRecord], str]:
    """滑动窗口裁剪：保留最近 N 次迭代明细，早期记录折叠为一行摘要。

    这是防止上下文膨胀（Context Bloat）的关键：冗长的失败日志只保留摘要。
    """
    if len(history) <= keep_recent:
        return list(history), ""
    folded = history[:-keep_recent]
    recent = history[-keep_recent:]
    lines: list[str] = []
    for rec in folded:
        outcome = "pass" if rec.get("test_passed") else ("lsp-fail" if not rec.get("lsp_passed") else "test-fail")
        summary = (rec.get("error_summary") or "").splitlines()
        head = summary[0] if summary else ""
        lines.append(
            f"- iter{rec.get('iteration')} [{outcome}] {rec.get('strategy', '')}: {head}"
        )
    text = "\n".join(lines)[:max_summary_chars]
    return recent, text


def trim_context(text: str, *, budget_tokens: int | None = None) -> str:
    """按 token 预算裁剪长文本（保留头部，尾部加省略提示）。"""
    budget = budget_tokens or global_settings.agent.context_window
    if count_tokens(text) <= budget:
        return text
    # 近似保留前 budget*4 字符
    cutoff = max(200, budget * _TOKEN_DIVISOR)
    return text[:cutoff] + "\n... (context trimmed)"


def should_escalate(state: AgentState, *, reason: str = "") -> tuple[bool, str]:
    """判断是否应触发熔断降级 / 人工介入。"""
    cfg = global_settings.agent
    if state.get("current_iteration", 0) >= state.get("max_iterations", cfg.max_iterations):
        return True, reason or "达到最大重试阈值"
    if state.get("same_error_streak", 0) >= cfg.same_error_streak:
        return True, reason or "连续出现相同报错指纹（疑似死循环）"
    if state.get("stall_streak", 0) >= cfg.stall_streak:
        return True, reason or "修改无进展（同一代码行反复修改）"
    if state.get("token_usage", 0) > cfg.token_budget:
        return True, reason or "Token 预算超限"
    return False, ""


def register_failure(state: AgentState, fingerprint: str, stalled: bool) -> dict[str, Any]:
    """更新连续相同错误 / 无进展计数，返回局部状态更新。

    Args:
        fingerprint: 本轮报错的堆栈特征哈希。
        stalled: 本轮补丁是否与上一轮完全相同（无进展）。
    """
    same = int(state.get("same_error_streak", 0))
    prev_fp = state.get("last_fingerprint", "")
    same = same + 1 if fingerprint and fingerprint == prev_fp else (1 if fingerprint else 0)
    stall = int(state.get("stall_streak", 0)) + 1 if stalled else 0
    return {
        "same_error_streak": same,
        "stall_streak": stall,
        "last_fingerprint": fingerprint,
    }
