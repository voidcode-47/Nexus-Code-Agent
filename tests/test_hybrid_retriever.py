"""混合检索器测试（离线内存模式）。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from src.indexer.ast_parser import parse_source
from src.indexer.call_graph import CallGraph
from src.indexer.hybrid_retriever import HashingEmbedder, HybridRetriever, tokenize
from src.indexer.symbol_table import build_symbol_table

SOURCE = '''"""检索样例。"""


def calculate_interest(principal, rate):
    return principal * rate


def render_report(rows):
    return "\\n".join(rows)


def handle_error(code):
    return f"E{code}"
'''


def _chunks():
    result = parse_source(SOURCE, "billing.py")
    return result.chunks


def test_tokenize_splits_identifiers():
    tokens = tokenize("calculateInterest snake_case HTTPError")
    assert "calculateinterest" in tokens
    assert "calculate" in tokens or "interest" in tokens
    assert "snake" in tokens and "case" in tokens


def test_embedding_is_deterministic():
    emb = HashingEmbedder(dim=64)
    a = emb.embed(["hello world"])[0]
    b = emb.embed(["hello world"])[0]
    assert a == b
    assert len(a) == 64


def test_search_returns_relevant_symbol():
    retriever = HybridRetriever()
    retriever.index(_chunks())
    hits = retriever.search("interest principal rate", top_k=3)
    assert hits
    assert hits[0].symbol.endswith("calculate_interest")


def test_search_uses_bm25_for_exact_identifier():
    retriever = HybridRetriever()
    retriever.index(_chunks())
    hits = retriever.search("handle_error", top_k=3)
    assert any("handle_error" in h.symbol for h in hits)


def test_rrf_ranks_recorded():
    retriever = HybridRetriever()
    retriever.index(_chunks())
    hits = retriever.search("report rows join", top_k=3)
    assert hits[0].dense_rank is not None or hits[0].sparse_rank is not None


def test_index_repo_with_call_graph(tmp_path):
    (tmp_path / "mod.py").write_text(SOURCE, encoding="utf-8")
    table = build_symbol_table(tmp_path)
    graph = CallGraph.build(table)
    retriever = HybridRetriever(table=table, call_graph=graph)
    count = retriever.index_repo(table, graph)
    assert count >= 3
    hits = retriever.search("interest", top_k=2)
    assert hits[0].pagerank >= 0.0


def test_empty_index_search():
    retriever = HybridRetriever()
    retriever.index([])
    assert retriever.search("anything") == []


def test_sparse_channel_works_without_rank_bm25():
    """rank_bm25 缺失时走内置 _BM25 兜底，稀疏通道必须仍然可用。

    回归背景：调用方统一按 ``get_scores`` 取分，而内置兜底实现只提供了
    ``scores``，导致「未安装 rank-bm25」的环境在检索阶段直接抛
    AttributeError（文档承诺的兜底路径实际不可用）。
    """
    import src.indexer.hybrid_retriever as hr

    retriever = hr.HybridRetriever()
    retriever.index(_chunks())
    # 用内置兜底实现替换索引（等价于环境未安装 rank_bm25）
    texts = [f"{c.symbol}\n{c.source}" for c in retriever._chunks]
    fallback = hr._BM25([tokenize(t) for t in texts])
    assert fallback.get_scores(["handle_error"])  # 与 BM25Okapi 接口对齐
    retriever._bm25 = fallback

    hits = retriever.search("handle_error", top_k=3)
    assert any("handle_error" in h.symbol for h in hits)


def test_dense_channel_ranks_semantic_match_first():
    retriever = HybridRetriever()
    retriever.index(_chunks())
    hits = retriever.search("interest principal rate", top_k=3)
    assert hits[0].symbol.endswith("calculate_interest")
    assert hits[0].dense_rank == 1  # Dense 通道必须把语义最相关的切片排第一


def test_embedding_stable_across_hash_seeds():
    """HashingEmbedder 不得依赖 PYTHONHASHSEED（跨进程 / 持久化索引稳定）。"""
    root = Path(__file__).resolve().parents[1]
    code = (
        "from src.indexer.hybrid_retriever import HashingEmbedder; "
        "print(HashingEmbedder(dim=32).embed(['hello world billing invoice'])[0])"
    )

    def embed_with(seed: str) -> str:
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip()

    assert embed_with("0") == embed_with("12345")
