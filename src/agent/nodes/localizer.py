"""故障定位 Agent (Fault Localization)。

将报错堆栈 ``(file, line)`` 帧与自然语言 Issue 描述融合，借助混合检索与调用图
溯源，锁定最可疑的文件与核心函数签名。
"""

from __future__ import annotations

from typing import Any

from src.agent.llm import LLMClient, parse_json_block
from src.agent.prompts import localizer_messages, render_snippets
from src.agent.repair_heuristics import is_test_only_path
from src.agent.state import AgentState, add_tokens, event
from src.config import get_logger
from src.indexer.call_graph import CallGraph
from src.indexer.hybrid_retriever import HybridRetriever
from src.sandbox.executor import extract_frames

logger = get_logger(__name__)


async def localizer_node(
    state: AgentState,
    *,
    retriever: HybridRetriever,
    graph: CallGraph | None = None,
    llm: LLMClient | None = None,
    n_hop: int = 2,
) -> dict[str, Any]:
    """执行故障定位，返回状态局部更新。"""
    issue = state.get("issue_description", "")
    frames = extract_frames(issue)
    hits = retriever.search(issue, top_k=retriever.retrieval.top_k)

    seed_symbols: list[str] = []
    if graph is not None and frames:
        seed_symbols = graph.seeds_from_trace(frames)
    if not seed_symbols:
        seed_symbols = [h.symbol for h in hits[:3]]

    subgraph_context = ""
    if graph is not None and seed_symbols:
        nodes = graph.subgraph_nodes(seed_symbols, n_hop=n_hop)
        subgraph_context = graph.render_context(nodes)

    # 可疑文件：优先业务代码，测试文件排后
    ordered_files = sorted(
        dict.fromkeys(h.path for h in hits),
        key=lambda p: (is_test_only_path(p), next((i for i, h in enumerate(hits) if h.path == p), 99)),
    )
    suspected_files = ordered_files[:5]
    suspected_symbols = list(dict.fromkeys([*seed_symbols, *(h.symbol for h in hits)]))[:8]
    retrieved_context = render_snippets(hits)

    reason = (
        f"混合检索命中 {len(hits)} 个切片，调用图锚定 {len(seed_symbols)} 个种子符号；"
        f"最可疑文件：{suspected_files[:3]}"
    )

    tokens = 0
    if llm is not None and llm.enabled:
        messages = localizer_messages(issue, _frames_text(frames), retrieved_context)
        try:
            raw = await llm.complete(messages, json_mode=True)
            payload = parse_json_block(raw)
            if payload.get("suspected_files"):
                suspected_files = list(
                    dict.fromkeys([*payload["suspected_files"], *suspected_files])
                )[:6]
            if payload.get("suspected_symbols"):
                suspected_symbols = list(
                    dict.fromkeys([*payload["suspected_symbols"], *suspected_symbols])
                )[:10]
            reason = payload.get("reason", reason)
            tokens = add_tokens(state, raw)
        except Exception as exc:  # pragma: no cover - 网络异常
            logger.warning("localizer LLM 调用失败，使用确定性结果", extra={"error": str(exc)})

    update: dict[str, Any] = {
        "suspected_files": suspected_files,
        "suspected_symbols": suspected_symbols,
        "call_graph_context": subgraph_context,
        "status": "PLANNING",
        "events": [
            event(
                "localizer",
                reason,
                suspected_files=suspected_files,
                suspected_symbols=suspected_symbols,
                hits=[
                    {"symbol": h.symbol, "path": h.path, "score": round(h.score, 4)}
                    for h in hits
                ],
            )
        ],
    }
    if tokens:
        update["token_usage"] = tokens
    logger.info("localizer 完成", extra={"files": suspected_files[:3]})
    return update


def _frames_text(frames: list[tuple[str, int, str]]) -> str:
    if not frames:
        return ""
    return "\n".join(f'  File "{f}", line {ln}, in {fn}' for f, ln, fn in frames)
