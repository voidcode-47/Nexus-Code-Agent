"""Qdrant Dense 向量 + BM25 稀疏向量 + RRF 重排序的混合检索器。

检索流水线：

1. **Dense 语义召回**：代码切片向量化后在 Qdrant 中做近邻检索，理解「意图」；
2. **BM25 稀疏召回**：精确匹配变量名 / 函数名 / 错误码 / 报错关键字；
3. **RRF 倒数秩融合**：``score = Σ 1 / (k + rank)`` 融合两路结果，避免分数尺度差异；
4. **PageRank 核心度加权**：优先召回高核心度依赖（被大量调用的「改动中心」）。

Qdrant 不可用时自动降级为内存向量索引（NumPy 余弦相似度），保证离线可跑。
"""

from __future__ import annotations

import hashlib
import math
import re
import zlib
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.config import get_logger
from src.config.settings import RetrievalSettings
from src.config.settings import settings as global_settings
from src.indexer.ast_parser import CodeChunk
from src.indexer.call_graph import CallGraph

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[A-Z]{2,}")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    """代码感知分词：保留下划线标识符，并拆分 camelCase / snake_case。"""
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(text):
        low = match.lower()
        tokens.append(low)
        for part in _CAMEL_RE.split(match):
            if part and part.lower() != low:
                tokens.append(part.lower())
        if "_" in match:
            tokens.extend(p.lower() for p in match.split("_") if p)
    return tokens


class Embedder(Protocol):
    """文本向量化协议。"""

    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """确定性哈希词袋向量器（离线默认实现，无需外部模型）。

    采用 hashing trick + L2 归一化。散列使用 CRC32 而非内置 ``hash()``，
    跨进程稳定（不受 ``PYTHONHASHSEED`` 影响），保证相同文本在任意进程
    中得到相同向量（持久化 Qdrant 索引 / 多进程评测也安全）；
    生产可通过配置替换为真实 embedding 模型。
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            counts = Counter(tokenize(text))
            for token, count in counts.items():
                idx = zlib.crc32(token.encode("utf-8")) % self.dim
                vec[idx] += 1.0 + math.log(count)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            out.append(vec.tolist())
        return out


@dataclass(slots=True)
class RetrievalHit:
    """一条检索命中。"""

    chunk_id: str
    path: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    source: str
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    pagerank: float = 0.0

    def snippet(self, max_lines: int = 12) -> str:
        lines = self.source.splitlines()
        head = lines[:max_lines]
        if len(lines) > max_lines:
            head.append("    # ... (truncated)")
        return "\n".join(head)


class _BM25:
    """纯 Python BM25（``rank_bm25`` 缺失时的兜底实现）。"""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus_tokens
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.doc_freqs: list[Counter[str]] = [Counter(d) for d in corpus_tokens]
        df: Counter[str] = Counter()
        for freqs in self.doc_freqs:
            df.update(freqs.keys())
        n = len(corpus_tokens)
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def scores(self, query: Sequence[str]) -> list[float]:
        result = [0.0] * len(self.corpus)
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                result[i] += idf * (tf * (self.k1 + 1)) / denom
        return result

    def get_scores(self, query: Sequence[str]) -> list[float]:
        """与 ``rank_bm25.BM25Okapi`` 的接口对齐。

        调用方（:meth:`HybridRetriever._sparse_search`）统一按 ``get_scores``
        取分；旧实现只有 ``scores``，导致 ``rank_bm25`` 缺失时走兜底实现的分支
        直接抛 AttributeError（稀疏通道整体不可用）。
        """
        return self.scores(query)


@dataclass
class HybridRetriever:
    """代码切片混合检索器。"""

    retrieval: RetrievalSettings = field(default_factory=lambda: global_settings.retrieval)
    embedder: Embedder | None = None
    table: object | None = None  # SymbolTable，弱引用以避免循环导入
    call_graph: CallGraph | None = None
    pagerank_weight: float = 0.3
    repo_root: str = ""

    _chunks: list[CodeChunk] = field(default_factory=list, init=False)
    _matrix: np.ndarray | None = field(default=None, init=False, repr=False)
    _bm25: Any = field(default=None, init=False, repr=False)
    _updated: bool = field(default=False, init=False)
    _qdrant: Any = field(default=None, init=False, repr=False)
    _by_chunk_id: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _pr_scale: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.embedder is None:
            self.embedder = HashingEmbedder(self.retrieval.embedding_dim)

    # -- 索引 ------------------------------------------------------------- #

    def index(self, chunks: Iterable[CodeChunk]) -> int:
        """重建索引。返回被索引的切片数量。"""
        self._chunks = [c for c in chunks if c.kind != "module"]
        if not self._chunks:
            self._chunks = list(chunks)
        texts = ["\n".join([c.symbol, c.signature, c.docstring, c.source]) for c in self._chunks]
        if texts and self.embedder is not None:
            self._matrix = np.asarray(self.embedder.embed(texts), dtype=np.float32)
        else:
            self._matrix = None
        self._bm25 = self._build_bm25(texts)
        self._by_chunk_id = {c.chunk_id: i for i, c in enumerate(self._chunks)}
        self._updated = True
        if self.retrieval.use_qdrant:
            self._upsert_qdrant(texts)
        logger.info("retriever indexed", extra={"chunks": len(self._chunks)})
        return len(self._chunks)

    def index_repo(self, table, call_graph: CallGraph | None = None) -> int:
        """从符号表（复用其 AST 切片）构建索引。"""
        self.table = table
        self.call_graph = call_graph
        self._pr_scale = None  # 调用图变更后 PageRank 归一化尺度需重算
        chunks: list[CodeChunk] = []
        for file_chunks in getattr(table, "chunks", {}).values():
            chunks.extend(file_chunks)
        return self.index(chunks)

    def _build_bm25(self, texts: list[str]) -> Any:
        corpus = [tokenize(t) for t in texts]
        if not corpus:
            return None
        try:
            from rank_bm25 import BM25Okapi

            return BM25Okapi(corpus)
        except Exception:  # pragma: no cover - fallback path
            return _BM25(corpus)

    # -- 检索 ------------------------------------------------------------- #

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        """执行混合检索并返回融合排序后的命中。"""
        if not self._chunks:
            return []
        k = top_k or self.retrieval.top_k
        dense_ranked = self._dense_search(query)
        sparse_ranked = self._sparse_search(query)
        fused = self._rrf_fuse(dense_ranked, sparse_ranked)

        hits: list[RetrievalHit] = []
        scale = self._pagerank_scale()
        for idx, score in fused[: k * 3]:
            chunk = self._chunks[idx]
            pr = self._pagerank_of(chunk.symbol)
            # PageRank 与 RRF 分数量纲不同（RRF 单项 ~1/(k+rank)，PageRank 可达 0.x，
            # 直接相加会让排序被调用图中心度主导）。归一化后改为乘性微调：
            # 混合检索排序保持主体地位，中心度只贡献 ≤pagerank_weight 的提升。
            pr_norm = (pr / scale) if scale > 0 else 0.0
            final = score * (1.0 + self.pagerank_weight * pr_norm)
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    path=chunk.path,
                    symbol=chunk.symbol,
                    kind=chunk.kind,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    source=chunk.source,
                    score=final,
                    dense_rank=_rank_of(idx, dense_ranked),
                    sparse_rank=_rank_of(idx, sparse_ranked),
                    pagerank=pr,
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def _dense_search(self, query: str) -> list[tuple[int, float]]:
        if self._matrix is None or self.embedder is None:
            return []
        if self.retrieval.use_qdrant:
            remote = self._qdrant_search(query)
            if remote:
                return remote
        qvec = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        sims = self._matrix @ qvec
        order = np.argsort(-sims)[: max(self.retrieval.top_k * 4, 20)]
        return [(int(i), float(sims[i])) for i in order]

    def _sparse_search(self, query: str) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        pairs = [(i, float(s)) for i, s in enumerate(scores) if s > 0]
        pairs.sort(key=lambda kv: kv[1], reverse=True)
        return pairs[: max(self.retrieval.top_k * 4, 20)]

    def _rrf_fuse(
        self,
        dense: list[tuple[int, float]],
        sparse: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        k = self.retrieval.rrf_k
        scores: dict[int, float] = {}
        for rank, (idx, _) in enumerate(dense):
            scores[idx] = scores.get(idx, 0.0) + self.retrieval.dense_weight / (k + rank + 1)
        for rank, (idx, _) in enumerate(sparse):
            scores[idx] = scores.get(idx, 0.0) + self.retrieval.bm25_weight / (k + rank + 1)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    def _pagerank_of(self, symbol: str) -> float:
        if self.call_graph is None:
            return 0.0
        return self.call_graph.pagerank(symbol)

    def _pagerank_scale(self) -> float:
        """PageRank 全网最大值（归一化用；惰性计算并缓存）。"""
        if self._pr_scale is None:
            getter = getattr(self.call_graph, "max_pagerank", None)
            self._pr_scale = float(getter()) if callable(getter) else 0.0
        return self._pr_scale

    # -- Qdrant（可选） --------------------------------------------------- #

    def _client(self) -> Any:
        if self._qdrant is not None:
            return self._qdrant
        try:
            from qdrant_client import QdrantClient

            self._qdrant = QdrantClient(
                url=self.retrieval.qdrant_url,
                api_key=self.retrieval.qdrant_api_key or None,
                timeout=5,
            )
            return self._qdrant
        except Exception as exc:  # pragma: no cover - 外部依赖
            logger.warning("qdrant unavailable, fallback to in-memory", extra={"error": str(exc)})
            return None

    def _collection_name(self) -> str:
        """按仓库指纹隔离 Qdrant 集合。

        单集合共享 + ``recreate_collection`` 会让两个仓库（或并发进程）
        互相清空对方索引。指纹优先取「仓库根路径」：同一仓库反复索引集合名
        稳定（增删文件不换名，不产生无人清理的孤儿集合）；未提供仓库路径时
        退回「切片路径集合」指纹。

        不做自动清理：同一 Qdrant 可能被多仓库共享，按 ``base__`` 前缀清扫
        会误删其他仓库的集合。历史孤儿集合（旧版本指纹、仓库迁移路径）由
        运维按需手动删除。
        """
        base = self.retrieval.collection
        if self.repo_root:
            key = str(Path(self.repo_root).resolve()).lower()
            return f"{base}__{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}"
        if not self._chunks:
            return base
        digest = hashlib.sha1(
            "\n".join(sorted({c.path for c in self._chunks})).encode("utf-8")
        ).hexdigest()[:12]
        return f"{base}__{digest}"

    def _upsert_qdrant(self, texts: list[str]) -> None:  # pragma: no cover - 外部依赖
        client = self._client()
        if client is None or self.embedder is None:
            return
        collection = self._collection_name()
        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams

            vectors = self.embedder.embed(texts)
            client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
            )
            points = [
                PointStruct(
                    id=i,
                    vector=vec,
                    # _idx 与 chunk_id 双写：跨进程/跨仓库检索时优先用 chunk_id 反查
                    payload={"_idx": i, "chunk_id": self._chunks[i].chunk_id},
                )
                for i, vec in enumerate(vectors)
            ]
            client.upsert(collection_name=collection, points=points)
        except Exception as exc:
            logger.warning("qdrant upsert failed", extra={"error": str(exc)})

    def _qdrant_search(self, query: str) -> list[tuple[int, float]]:  # pragma: no cover
        client = self._client()
        if client is None or self.embedder is None:
            return []
        try:
            qvec = self.embedder.embed([query])[0]
            response = client.query_points(
                collection_name=self._collection_name(),
                query=qvec,
                limit=max(self.retrieval.top_k * 4, 20),
            )
            hits: list[tuple[int, float]] = []
            for point in response.points:
                payload = point.payload or {}
                # 优先用 chunk_id 反查本地下标（_idx / 点 ID 仅在集合由本进程
                # 按相同顺序建立时才与 self._chunks 对应，不能盲目信任）
                idx = self._by_chunk_id.get(str(payload.get("chunk_id", "")))
                if idx is None:
                    raw = payload.get("_idx", point.id)
                    idx = int(raw) if isinstance(raw, int) else None
                    if idx is None or not (0 <= idx < len(self._chunks)):
                        continue
                hits.append((idx, float(point.score)))
            return hits
        except Exception as exc:
            logger.warning("qdrant search failed", extra={"error": str(exc)})
            return []


def _rank_of(idx: int, ranked: list[tuple[int, float]]) -> int | None:
    for rank, (i, _) in enumerate(ranked):
        if i == idx:
            return rank + 1
    return None
