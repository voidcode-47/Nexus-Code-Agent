"""补丁生成 Agent。

优先使用 LLM 依据修复计划生成 Search-Replace / Unified Diff 补丁；
无 LLM 时退化为基于 AST 变异的确定性候选补丁。补丁统一经
:mod:`src.mcp_tools.diff_patcher` 的模糊匹配引擎应用到 Worktree。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from src.agent.llm import LLMClient
from src.agent.prompts import coder_messages
from src.agent.repair_heuristics import (
    CandidatePatch,
    generate_candidate_patches,
    is_test_only_path,
)
from src.agent.state import AgentState, add_tokens, event, trim_context
from src.config import get_logger
from src.config.settings import settings as global_settings
from src.indexer.call_graph import CallGraph
from src.mcp_tools.diff_patcher import apply_patch
from src.sandbox.git_snapshot import reset_workdir

logger = get_logger(__name__)


async def coder_node(
    state: AgentState,
    *,
    llm: LLMClient | None = None,
    graph: CallGraph | None = None,
) -> dict[str, Any]:
    """生成并应用补丁，返回状态局部更新。"""
    worktree = Path(state.get("worktree_path", ""))
    target_rel = _select_target_file(state, worktree)
    if target_rel is None:
        return {
            "status": "HUMAN_ESCALATION",
            "termination_reason": "未能在工作区定位到可修改的源文件",
            "events": [event("coder", "定位失败：无可用目标文件", level="error")],
        }

    target_path = worktree / target_rel
    # 写入前再校验一次（写入本身不可回滚，这里拒绝任何逃出 Worktree 的路径）
    if _confine_to_worktree(worktree, target_rel) is None:
        return {
            "status": "HUMAN_ESCALATION",
            "termination_reason": f"目标路径越出 Worktree，已拒绝写入：{target_rel}",
            "events": [
                event("coder", "目标路径越界，已拒绝写入", level="error", target=target_rel)
            ],
        }
    iteration = int(state.get("current_iteration", 0))
    plan = state.get("repair_plan", "")

    # 每轮从基线状态开始，避免候选补丁叠加在已修改内容上
    if iteration > 0 and state.get("changed_files"):
        try:
            await reset_workdir(worktree)
        except Exception as exc:  # pragma: no cover
            logger.warning("worktree 回滚失败", extra={"error": str(exc)})

    # 文件读取 / 补丁匹配 / 写回都是同步阻塞操作（补丁匹配在极端输入下可达
    # 百毫秒级），放线程池执行，避免拖住同进程的其他任务与 SSE 心跳。
    source = await asyncio.to_thread(target_path.read_text, encoding="utf-8")
    # 送入 LLM 的源码按上下文预算裁剪；apply_patch 仍使用完整源码，避免截断错位
    prompt_source = trim_context(
        source, budget_tokens=max(2000, global_settings.agent.context_window // 4)
    )

    patch_text = ""
    strategy = "llm"
    tokens = 0
    stored: list[str] = list(state.get("candidate_patches", []))

    if llm is not None and llm.enabled:
        feedback = state.get("test_output", "")
        messages = coder_messages(plan, target_rel, prompt_source, feedback=feedback, attempt=iteration)
        try:
            patch_text = (await llm.complete(messages)).strip()
            tokens = add_tokens(state, patch_text)
        except Exception as exc:  # pragma: no cover
            logger.warning("coder LLM 调用失败，回退确定性策略", extra={"error": str(exc)})
            patch_text = ""

    if not patch_text:
        if not stored:
            candidates = generate_candidate_patches(source)
            candidates = _filter_by_symbols(candidates, state.get("suspected_symbols", []), graph)
            stored = [c.to_patch() for c in candidates]
            if not stored:
                return {
                    "status": "HUMAN_ESCALATION",
                    "termination_reason": "无法为可疑函数生成候选补丁",
                    "events": [event("coder", "候选补丁为空", level="error")],
                }
        # 已试候选统计：优先尝试次数最少者（0 次优先）。纯轮转 `iteration % n`
        # 在「LLM 轮与候选轮交错」时会对齐到错误下标，重放已证明失败的补丁白耗
        # 一轮；按次数调度则保证仍有未试候选时绝不重放。
        attempts: dict[str, int] = {}
        for rec in state.get("patch_history", []):
            diff = rec.get("diff_content", "")
            if diff:
                attempts[diff] = attempts.get(diff, 0) + 1
        index = min(range(len(stored)), key=lambda i: (attempts.get(stored[i], 0), i))
        patch_text = stored[index]
        strategy = f"candidate#{index}"
        logger.info(
            "coder 采用确定性候选补丁",
            extra={"index": index, "total": len(stored)},
        )

    outcome = await asyncio.to_thread(apply_patch, source, patch_text)
    update: dict[str, Any] = {
        "last_patch": patch_text,
        "last_patch_target": target_rel,
        "patch_applied": outcome.applied,
        "changed_files": [target_rel] if outcome.applied else [],
        "candidate_patches": stored,
        "status": "TESTING",
        "events": [
            event(
                "coder",
                f"补丁 {'应用成功' if outcome.applied else '应用失败'}（{outcome.method}/{strategy}）",
                target=target_rel,
                applied=outcome.applied,
                detail=outcome.describe(),
                patch=patch_text[:4000],
                strategy=strategy,
            )
        ],
    }
    if outcome.applied:
        await asyncio.to_thread(target_path.write_text, outcome.new_content, encoding="utf-8")
    else:
        update["test_output"] = f"补丁应用失败：{outcome.error}"
    if tokens:
        update["token_usage"] = tokens
    logger.info("coder 完成", extra={"target": target_rel, "applied": outcome.applied})
    return update


def _select_target_file(state: AgentState, worktree: Path) -> str | None:
    """选择目标文件，返回**收敛在 worktree 内**的相对 POSIX 路径。

    候选路径来自 ``suspected_files``，其中可能包含 LLM 输出的文件清单
    （localizer 会把模型返回的路径优先并入）。不加约束时，绝对路径
    （``/etc/...``、``C:\\...``）或 ``..`` 逃逸会让补丁写到 Worktree 之外——
    沙箱隔离失效，且 GitHub Issue 正文可经提示注入影响模型输出。
    """
    if not state.get("worktree_path"):
        logger.warning("缺少 worktree_path，拒绝选择目标文件")
        return None

    existing: list[str] = []
    for candidate in state.get("suspected_files", []):
        rel = _confine_to_worktree(worktree, candidate)
        if rel is not None and (worktree / rel).is_file():
            existing.append(rel)
    business = [p for p in existing if not is_test_only_path(p)]
    if business:
        return business[0]
    if existing:
        return existing[0]
    # 兜底：扫描工作区，取第一个业务 Python 文件
    for path in sorted(worktree.rglob("*.py")):
        rel = path.relative_to(worktree).as_posix()
        if is_test_only_path(rel) or ".git" in rel or "__pycache__" in rel:
            continue
        return rel
    return None


def _confine_to_worktree(worktree: Path, candidate: str) -> str | None:
    """把候选路径收敛为 worktree 内的相对路径；逃逸（绝对路径 / ``..`` / 符号链接）返回 None。"""
    if not candidate:
        return None
    raw = Path(candidate)
    if raw.is_absolute() or raw.drive:
        return None
    root = worktree.resolve()
    resolved = (worktree / raw).resolve()
    if resolved != root and root not in resolved.parents:
        return None
    return resolved.relative_to(root).as_posix()


def _filter_by_symbols(
    candidates: list[CandidatePatch],
    symbols: list[str],
    graph: CallGraph | None,
) -> list[CandidatePatch]:
    """若能把候选补丁约束到可疑符号的行区间内，则优先保留。"""
    if graph is None or not symbols:
        return candidates
    ranges: list[tuple[int, int]] = []
    for name in symbols:
        for sym in graph.table.lookup(name, limit=3):
            ranges.append((sym.start_line, sym.end_line))
    if not ranges:
        return candidates
    inside = [
        c for c in candidates if any(lo <= c.line <= hi for lo, hi in ranges)
    ]
    return inside or candidates
