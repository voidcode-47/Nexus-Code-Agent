"""符号表与调用图测试。"""

from __future__ import annotations

from src.indexer.call_graph import CallGraph
from src.indexer.symbol_table import build_symbol_table

PKG_INIT = ""
MODULE_A = '''"""A 模块。"""

from src.indexer import symbol_table


def alpha():
    return beta(1)


def beta(x):
    return x + 1
'''

MODULE_B = '''"""B 模块。"""
from .module_a import alpha


class Base:
    def run(self):
        return alpha()


class Child(Base):
    def run(self):
        return super().run()
'''


def _make_pkg(tmp_path):
    root = tmp_path / "proj"
    (root / "src" / "indexer").mkdir(parents=True)
    (root / "src" / "__init__.py").write_text(PKG_INIT, encoding="utf-8")
    (root / "src" / "indexer" / "__init__.py").write_text(PKG_INIT, encoding="utf-8")
    (root / "src" / "indexer" / "module_a.py").write_text(MODULE_A, encoding="utf-8")
    (root / "src" / "indexer" / "module_b.py").write_text(MODULE_B, encoding="utf-8")
    return root


def test_build_symbol_table(tmp_path):
    root = _make_pkg(tmp_path)
    table = build_symbol_table(root)
    assert table.lookup("alpha")
    assert table.lookup("src.indexer.module_a.alpha")
    assert table.lookup("Base")[0].kind == "class"


def test_symbol_at(tmp_path):
    root = _make_pkg(tmp_path)
    table = build_symbol_table(root)
    sym = table.symbol_at("src/indexer/module_a.py", 7)
    assert sym is not None
    assert sym.name == "alpha"


def test_resolve_relative_import(tmp_path):
    root = _make_pkg(tmp_path)
    table = build_symbol_table(root)
    imports = table.imports["src/indexer/module_b.py"]
    relative = [i for i in imports if i.is_relative]
    assert relative
    resolved = table.resolve_import("src/indexer/module_b.py", relative[0])
    assert resolved.endswith("module_a")


def test_call_graph_edges(tmp_path):
    root = _make_pkg(tmp_path)
    table = build_symbol_table(root)
    graph = CallGraph.build(table)
    assert graph.graph.has_node("src.indexer.module_a.alpha")
    callees = graph.callees_of("src.indexer.module_a.alpha")
    assert any("beta" in c for c in callees)


def test_pagerank_and_subgraph(tmp_path):
    root = _make_pkg(tmp_path)
    graph = CallGraph.from_repo(root)
    top = graph.top_core(5)
    assert top
    assert all(score >= 0 for _, score in top)
    nodes = graph.subgraph_nodes(["src.indexer.module_a.alpha"], n_hop=2)
    assert "src.indexer.module_a.alpha" in nodes


def test_inheritance(tmp_path):
    root = _make_pkg(tmp_path)
    table = build_symbol_table(root)
    graph = CallGraph.build(table)
    assert graph.inheritance.number_of_nodes() >= 2


def test_seeds_from_trace(tmp_path):
    root = _make_pkg(tmp_path)
    graph = CallGraph.from_repo(root)
    seeds = graph.seeds_from_trace([("src/indexer/module_a.py", 7, "alpha")])
    assert any("alpha" in s for s in seeds)


def test_render_context(tmp_path):
    root = _make_pkg(tmp_path)
    graph = CallGraph.from_repo(root)
    text = graph.render_context(["src.indexer.module_a.alpha"])
    assert "alpha" in text
    assert "core=" in text


def test_pagerank_falls_back_without_scipy(tmp_path, monkeypatch):
    """networkx.pagerank 不可用（环境无 scipy）时必须自动降级为内置幂迭代。

    回归背景：networkx 自 3.x 起 ``pagerank`` 内部固定走 ``_pagerank_scipy``，
    缺少 scipy 时抛 ModuleNotFoundError——而 scipy 并不是本项目依赖，
    于是「只装 requirements.txt」的干净环境在建图这一步就整体失败。
    """
    import networkx as nx

    from src.indexer import call_graph as cg_module

    def _boom(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'scipy'")

    monkeypatch.setattr(nx, "pagerank", _boom)
    root = _make_pkg(tmp_path)
    table = build_symbol_table(root)
    graph = CallGraph.build(table)  # 不应抛异常

    scores = {name: graph.pagerank(name) for name in table.symbols}
    assert scores, "兜底实现必须给出核心度"
    assert all(score > 0 for score in scores.values())
    # beta 被 alpha 调用，核心度应高于 alpha
    assert graph.pagerank("src.indexer.module_a.beta") > graph.pagerank(
        "src.indexer.module_a.alpha"
    )
    assert cg_module.pagerank_power_iteration(graph.graph)  # 兜底实现可直接调用
