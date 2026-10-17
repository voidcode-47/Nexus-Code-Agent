"""NetworkX 函数调用图、类继承图与 PageRank 核心度计算。

* **调用图**：节点为函数/方法限定名，边 ``caller → callee`` 表示静态调用；
* **继承图**：边 ``class → base`` 表示类继承关系；
* **PageRank 核心度**：高入度（被大量调用）的节点权重更高，检索时优先召回
  高核心度依赖，帮助定位「真正的改动中心」；
* **N-hop 子图**：给定报错堆栈锚定的种子符号，取出上下游 N 跳邻域，
  突破超长上下文限制，只把相关符号送进 LLM。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from src.config import get_logger
from src.indexer.symbol_table import SymbolTable, build_symbol_table

logger = get_logger(__name__)


def pagerank_power_iteration(
    graph: nx.DiGraph,
    *,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1.0e-6,
    weight: str = "weight",
) -> dict[str, float]:
    """纯 Python 幂迭代 PageRank（无 scipy 依赖）。

    语义与 ``networkx.pagerank`` 对齐：按边权归一化分配权重，悬挂节点
    （出度为 0）的权重平均回灌全图，收敛判据为 L1 变化量 < ``n * tol``。

    Raises:
        nx.PowerIterationFailedConvergence: 超过 ``max_iter`` 仍未收敛。
    """
    nodes = list(graph)
    n = len(nodes)
    if n == 0:
        return {}
    index = {node: i for i, node in enumerate(nodes)}
    out_weight = [0.0] * n
    edges: list[tuple[int, int, float]] = []
    for source, target, data in graph.edges(data=True):
        raw = data.get(weight, 1.0)
        w = 1.0 if raw is None else float(raw)
        if w <= 0:
            w = 0.0
        i, j = index[source], index[target]
        out_weight[i] += w
        edges.append((i, j, w))

    ranks = [1.0 / n] * n
    for _ in range(max_iter):
        previous = ranks
        ranks = [(1.0 - alpha) / n] * n
        dangling = sum(previous[i] for i in range(n) if out_weight[i] == 0.0)
        bonus = alpha * dangling / n
        for i, j, w in edges:
            if out_weight[i] > 0.0:
                ranks[j] += alpha * previous[i] * (w / out_weight[i])
        if bonus:
            ranks = [value + bonus for value in ranks]
        if sum(abs(ranks[i] - previous[i]) for i in range(n)) < n * tol:
            return {node: ranks[index[node]] for node in nodes}
    raise nx.PowerIterationFailedConvergence(max_iter)


@dataclass(slots=True)
class CallGraph:
    """仓库级调用图与继承图。"""

    table: SymbolTable
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    inheritance: nx.DiGraph = field(default_factory=nx.DiGraph)
    _pagerank: dict[str, float] = field(default_factory=dict)

    # -- 构建 ------------------------------------------------------------- #

    @classmethod
    def build(cls, table: SymbolTable) -> CallGraph:
        cg = cls(table=table)
        cg._build_call_edges()
        cg._build_inheritance_edges()
        cg._compute_pagerank()
        logger.info(
            "call graph built",
            extra={"nodes": cg.graph.number_of_nodes(), "edges": cg.graph.number_of_edges()},
        )
        return cg

    @classmethod
    def from_repo(cls, root: str | Path) -> CallGraph:
        return cls.build(build_symbol_table(root))

    def _build_call_edges(self) -> None:
        for symbols in self.table.by_path.values():
            for sym in symbols:
                self.graph.add_node(sym.qualified_name, kind=sym.kind)

        for chunks in self.table.chunks.values():
            for chunk in chunks:
                if chunk.kind not in {"function", "method"}:
                    continue
                caller = chunk.symbol
                self.graph.add_node(caller, path=chunk.path, line=chunk.start_line)
                for call in chunk.calls:
                    target = self._resolve_call(call, chunk)
                    if target and target != caller:
                        weight = self.graph.get_edge_data(caller, target, {}).get("weight", 0)
                        self.graph.add_edge(caller, target, weight=weight + 1)

    def _resolve_call(self, name: str, chunk) -> str | None:
        # 1) 限定名精确命中
        if name in self.table.symbols:
            return name
        # 2) 同一模块内的短名
        module = chunk.symbol.rsplit(".", 1)[0] if "." in chunk.symbol else ""
        candidates = self.table.by_name.get(name.split(".")[0], [])
        if not candidates:
            candidates = self.table.by_name.get(name.rsplit(".", 1)[-1], [])
        same_module = [c for c in candidates if c.module == module]
        if same_module:
            return same_module[0].qualified_name
        # 3) 方法调用 ``obj.method`` → 匹配同名方法
        method_name = name.rsplit(".", 1)[-1]
        methods = [c for c in self.table.by_name.get(method_name, []) if c.kind == "method"]
        if methods:
            return methods[0].qualified_name
        if candidates:
            return candidates[0].qualified_name
        return None

    def _build_inheritance_edges(self) -> None:
        for chunks in self.table.chunks.values():
            for chunk in chunks:
                if chunk.kind != "class":
                    continue
                self.inheritance.add_node(chunk.symbol)
                for base in chunk.bases:
                    base_name = base.split("[")[0].strip()
                    resolved = self._resolve_call(base_name, chunk)
                    if resolved:
                        self.inheritance.add_edge(chunk.symbol, resolved)

    def _compute_pagerank(self) -> None:
        node_count = self.graph.number_of_nodes()
        if node_count == 0:
            self._pagerank = {}
            return
        try:
            self._pagerank = nx.pagerank(self.graph, alpha=0.85, weight="weight")
            return
        except nx.PowerIterationFailedConvergence:  # pragma: no cover - 收敛失败
            logger.warning("PageRank 未收敛，退化为均匀权重")
        except Exception as exc:  # 典型原因：环境未安装 scipy
            # networkx 自 3.x 起 ``pagerank`` 内部固定调用 ``_pagerank_scipy``，
            # 缺少 scipy 时抛 ModuleNotFoundError——而 scipy 并非本项目依赖，
            # 干净环境（仅 requirements.txt）下建图会在这一步直接失败，
            # 连带索引 / 定位 / 评测全链路不可用。这里改用内置幂迭代兜底。
            logger.info("networkx.pagerank 不可用，改用内置幂迭代", extra={"error": str(exc)})
            try:
                self._pagerank = pagerank_power_iteration(
                    self.graph, alpha=0.85, weight="weight"
                )
                return
            except nx.PowerIterationFailedConvergence:  # pragma: no cover - 收敛失败
                logger.warning("内置 PageRank 幂迭代未收敛，退化为均匀权重")
            except Exception as inner:  # pragma: no cover - 兜底
                logger.warning(
                    "内置 PageRank 计算失败，退化为均匀权重", extra={"error": str(inner)}
                )
        self._pagerank = {node: 1.0 / node_count for node in self.graph}

    # -- 查询 ------------------------------------------------------------- #

    def pagerank(self, node: str) -> float:
        return self._pagerank.get(node, 0.0)

    def max_pagerank(self) -> float:
        """全网最高 PageRank（供混合检索归一化加权使用）。"""
        return max(self._pagerank.values(), default=0.0)

    def top_core(self, limit: int = 20) -> list[tuple[str, float]]:
        ranked = sorted(self._pagerank.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:limit]

    def callers_of(self, node: str) -> list[str]:
        if node not in self.graph:
            return []
        return list(self.graph.predecessors(node))

    def callees_of(self, node: str) -> list[str]:
        if node not in self.graph:
            return []
        return list(self.graph.successors(node))

    def subgraph_nodes(self, seeds: list[str], n_hop: int = 2) -> list[str]:
        """提取种子节点上下游 N 跳邻域内的节点（去重、保留核心度排序）。"""
        visited: set[str] = set()
        frontier = [s for s in seeds if s in self.graph]
        visited.update(frontier)
        for _ in range(max(0, n_hop)):
            new_frontier: list[str] = []
            for node in frontier:
                new_frontier.extend(self.callers_of(node))
                new_frontier.extend(self.callees_of(node))
            new_frontier = [n for n in new_frontier if n not in visited]
            visited.update(new_frontier)
            frontier = new_frontier
            if not frontier:
                break
        return sorted(visited, key=self.pagerank, reverse=True)

    def seeds_from_trace(self, frames: list[tuple], path_prefix: str = "") -> list[str]:
        """将报错堆栈 ``(file, line[, func])`` 帧转换为种子符号。"""
        seeds: list[str] = []
        for frame in frames:
            file, line = frame[0], int(frame[1])
            rel = file
            if path_prefix:
                rel = file.split(path_prefix, 1)[-1].lstrip("/\\")
            sym = self.table.symbol_at(rel.replace("\\", "/"), line)
            if sym:
                seeds.append(sym.qualified_name)
        return list(dict.fromkeys(seeds))

    def render_context(self, nodes: list[str], *, max_chars: int = 4000) -> str:
        """将子图节点渲染为紧凑的上下文文本，供规划 / 编码提示词使用。"""
        lines: list[str] = []
        for node in nodes:
            info = self.table.symbols.get(node)
            if info is None:
                continue
            rank = self.pagerank(node)
            callees = self.callees_of(node)[:5]
            lines.append(
                f"- {node} [{info.kind}] @ {info.location} "
                f"core={rank:.4f} → calls {callees}"
            )
        text = "\n".join(lines)
        return text[:max_chars]
