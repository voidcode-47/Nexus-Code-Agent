"""确定性修复策略：基于 AST 变异的候选补丁生成。

当未配置真实 LLM（``provider=heuristic``）时，编码 Agent 会退化为
**基于搜索的程序修复（Search-Based Program Repair）**：

1. 对可疑函数的 AST 施加一组「常见缺陷」变异（运算符、比较符、常量、
   ``not`` 取反、缺失 return 等）；
2. 每个变异生成一个标准的 Search-Replace 补丁；
3. 由沙箱测试逐轮验证候选补丁，测试通过者即最终修复。

这既是离线兜底，也是真实可用的 APS 引擎，确保无 API Key 也能跑通端到端闭环。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

# 运算符 → 候选替换（按工程经验排序，越靠前越常见）
_BINOP_CANDIDATES: dict[type, list[str]] = {
    ast.Add: ["-"],
    ast.Sub: ["+"],
    ast.Mult: ["+", "/"],
    ast.Div: ["*", "//"],
    ast.FloorDiv: ["/"],
    ast.Mod: ["/"],
    ast.BitOr: ["^", "&"],
    ast.BitAnd: ["|"],
    ast.BitXor: ["|"],
    ast.LShift: [">>"],
    ast.RShift: ["<<"],
}

_CMPOP_CANDIDATES: dict[type, list[str]] = {
    ast.Eq: ["!="],
    ast.NotEq: ["=="],
    ast.Lt: [">"],
    ast.LtE: [">="],
    ast.Gt: ["<"],
    ast.GtE: ["<="],
}


@dataclass(slots=True)
class CandidatePatch:
    """一个候选补丁。"""

    strategy: str
    line: int
    search: str
    replace: str

    def to_patch(self) -> str:
        return (
            "<<<<<<< SEARCH\n"
            f"{self.search}\n"
            "=======\n"
            f"{self.replace}\n"
            ">>>>>>> REPLACE"
        )


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    mapping: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            mapping[child] = node
    return mapping


def _nearest_statement(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.stmt | None:
    cur: ast.AST | None = node
    while cur is not None:
        if isinstance(cur, ast.stmt) and not isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return cur
        cur = parents.get(cur)
    return None


def _candidate(search: str, replace: str, strategy: str, line: int) -> CandidatePatch | None:
    if not search or not replace or search == replace:
        return None
    return CandidatePatch(strategy=strategy, line=line, search=search, replace=replace)


def generate_candidate_patches(
    source: str,
    *,
    max_candidates: int = 16,
) -> list[CandidatePatch]:
    """从源码中生成有序的候选补丁列表。"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    parents = _parents(tree)
    candidates: list[CandidatePatch] = []
    seen: set[tuple[str, str]] = set()

    def add(node: ast.AST, replacement: str, strategy: str) -> None:
        stmt = _nearest_statement(node, parents)
        if stmt is None or not (hasattr(stmt, "lineno") and hasattr(stmt, "end_lineno")):
            return
        seg_lines = source.splitlines()[stmt.lineno - 1 : stmt.end_lineno]
        seg = "\n".join(seg_lines)
        node_seg = ast.get_source_segment(source, node)
        if not node_seg or node_seg not in seg:
            return
        # 精确替换 node 出现的位置：同语句文本重复出现（如 `x - x`）时，
        # seg.replace(..., 1) 可能替换到更早的错误位置。用节点行号 + 列偏移
        # 定位后从该处向后搜索，保证替换到实际节点。
        node_line = int(getattr(node, "lineno", stmt.lineno) or stmt.lineno)
        rel_line = max(0, node_line - stmt.lineno)
        line_start = sum(len(line) + 1 for line in seg_lines[:rel_line])
        col = int(getattr(node, "col_offset", 0) or 0)
        # ast 的 col_offset 是 UTF-8 字节偏移，seg.find 按字符计——行首含中文时
        # 直接相加会越位。先按字节切片再解码，换算成字符偏移。
        line_text = seg_lines[rel_line] if rel_line < len(seg_lines) else ""
        col_chars = len(line_text.encode("utf-8")[:col].decode("utf-8", errors="ignore"))
        pos = seg.find(node_seg, line_start + col_chars)
        if pos < 0:
            pos = seg.find(node_seg, line_start)
        if pos < 0:  # pragma: no cover - 与上方 `node_seg in seg` 校验保持一致
            pos = seg.find(node_seg)
        if pos < 0:
            return
        new_seg = seg[:pos] + replacement + seg[pos + len(node_seg) :]
        cand = _candidate(seg, new_seg, strategy, stmt.lineno)
        if cand is not None and (cand.search, cand.replace) not in seen:
            seen.add((cand.search, cand.replace))
            candidates.append(cand)

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            for symbol in _BINOP_CANDIDATES.get(type(node.op), []):
                left = ast.get_source_segment(source, node.left)
                right = ast.get_source_segment(source, node.right)
                if left is not None and right is not None:
                    add(node, f"{left} {symbol} {right}", f"binop:{type(node.op).__name__}->{symbol}")
        elif isinstance(node, ast.Compare) and node.ops:
            for symbol in _CMPOP_CANDIDATES.get(type(node.ops[0]), []):
                left = ast.get_source_segment(source, node.left)
                right = ast.get_source_segment(source, node.comparators[0]) if node.comparators else None
                if left is not None and right is not None:
                    add(node, f"{left} {symbol} {right}", f"cmp:{type(node.ops[0]).__name__}->{symbol}")
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            operand = ast.get_source_segment(source, node.operand)
            if operand is not None:
                add(node, operand, "unary:drop-not")
        elif isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            for delta in (1, -1):
                add(node, str(node.value + delta), f"const:{node.value}->{node.value + delta}")
        elif isinstance(node, ast.Return) and node.value is not None:
            add(node, "return None", "return:->None")

    # 按「缺陷常见度」重排：运算符/比较符 > 常量/取反 > 缺失返回值
    priority = {"binop": 0, "cmp": 1, "const": 2, "unary": 2, "return": 9}
    candidates.sort(key=lambda c: priority.get(c.strategy.split(":", 1)[0], 5))
    return candidates[:max_candidates]


def is_test_only_path(path: str) -> bool:
    """判断路径是否属于测试代码（避免修改测试来「作弊」）。"""
    lowered = path.replace("\\", "/").lower()
    return (
        "/test_" in lowered
        or lowered.startswith("test_")
        or "/tests/" in lowered
        or lowered.endswith("_test.py")
        or "conftest.py" in lowered
    )
