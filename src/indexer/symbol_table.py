"""跨文件符号表 (Symbol Table) 与 import 路径解析。

符号表是调用图构建与故障定位（将 Traceback 的 ``file:line`` 映射回符号）的
基础设施。提供：

* 全仓符号索引（限定名 / 短名 / 按文件三种索引维度）；
* ``import`` 路径解析（含相对导入 ``from . import x``）；
* ``file:line → Symbol`` 反查，用于把报错堆栈直接锚定到可疑函数。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from src.config import get_logger
from src.indexer.ast_parser import (
    CodeChunk,
    ImportRecord,
    iter_python_files,
    parse_source,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class Symbol:
    """一个具名代码实体（模块 / 类 / 函数 / 方法）。"""

    qualified_name: str
    name: str
    kind: str
    path: str
    start_line: int
    end_line: int
    signature: str = ""
    module: str = ""
    docstring: str = ""

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}"


@dataclass(slots=True)
class SymbolTable:
    """全仓符号表。"""

    symbols: dict[str, Symbol] = field(default_factory=dict)
    by_name: dict[str, list[Symbol]] = field(default_factory=dict)
    by_path: dict[str, list[Symbol]] = field(default_factory=dict)
    imports: dict[str, list[ImportRecord]] = field(default_factory=dict)
    #: 相对路径 → 该文件的 AST 切片（供调用图 / 检索复用，避免重复解析）
    chunks: dict[str, list[CodeChunk]] = field(default_factory=dict)
    #: 顶层包名 → 仓库内路径，用于 import 解析
    roots: set[str] = field(default_factory=set)

    # -- 构建 ------------------------------------------------------------- #

    def add_symbol(self, symbol: Symbol) -> None:
        if symbol.kind == "module":
            return
        self.symbols[symbol.qualified_name] = symbol
        self.by_name.setdefault(symbol.name, []).append(symbol)
        self.by_path.setdefault(symbol.path, []).append(symbol)

    def add_imports(self, path: str, records: list[ImportRecord]) -> None:
        self.imports[path] = records

    # -- 查询 ------------------------------------------------------------- #

    def lookup(self, name: str, *, limit: int = 10) -> list[Symbol]:
        """按限定名或短名查找符号。"""
        if name in self.symbols:
            return [self.symbols[name]]
        matches = list(self.by_name.get(name, []))
        if not matches and "." in name:
            short = name.rsplit(".", 1)[-1]
            matches = list(self.by_name.get(short, []))
        return matches[:limit]

    def symbol_at(self, path: str, line: int) -> Symbol | None:
        """返回覆盖指定 ``path:line`` 的最内层符号。"""
        key = path.replace("\\", "/")
        candidates = self.by_path.get(key)
        if not candidates:
            # 允许相对于仓库根的尾部匹配
            key_tail = key.lstrip("./")
            for p, syms in self.by_path.items():
                if p.endswith(key_tail) or key_tail.endswith(p):
                    candidates = syms
                    break
        if not candidates:
            return None
        covering = [s for s in candidates if s.start_line <= line <= s.end_line]
        if not covering:
            return None
        return min(covering, key=lambda s: (s.end_line - s.start_line))

    def enclosing_module(self, path: str) -> str | None:
        key = path.replace("\\", "/")
        for sym in self.by_path.get(key, []):
            return sym.module
        return None

    def resolve_import(self, importer_path: str, record: ImportRecord) -> str:
        """将 import 记录解析为仓库内模块限定名（尽力而为）。"""
        if not record.is_relative:
            return record.module
        pkg_parts = importer_path.replace("\\", "/").split("/")[:-1]
        up = record.level - 1
        base = pkg_parts[: len(pkg_parts) - up] if up <= len(pkg_parts) else []
        tail = [record.module] if record.module else []
        return ".".join([*base, *tail]).strip(".")

    def dependencies_of(self, qualified_name: str) -> list[ImportRecord]:
        """返回定义该符号文件所依赖的 import 记录。"""
        symbol = self.symbols.get(qualified_name)
        if symbol is None:
            return []
        return self.imports.get(symbol.path, [])

    def modules(self) -> set[str]:
        return {s.module for s in self.symbols.values() if s.module}

    def __len__(self) -> int:  # pragma: no cover - 便捷
        return len(self.symbols)


def build_symbol_table(
    root: str | Path,
    *,
    files: Iterable[Path] | None = None,
    ignored: Iterable[str] = (),
) -> SymbolTable:
    """扫描仓库并构建符号表。"""
    table = SymbolTable()
    root_path = Path(root)
    source_files = list(files) if files is not None else list(iter_python_files(root, ignored))

    for file_path in source_files:
        try:
            source = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = _relpath(file_path, root_path)
        result = parse_source(source, rel)
        if not result.ok:
            logger.debug("skip unparsable file", extra={"path": rel, "error": result.error})
            continue
        table.add_imports(rel, result.imports)
        table.chunks[rel] = result.chunks
        for chunk in result.chunks:
            module_name = _module_name(rel)
            table.add_symbol(
                Symbol(
                    qualified_name=chunk.symbol,
                    name=chunk.name,
                    kind=chunk.kind,
                    path=chunk.path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    signature=chunk.signature,
                    module=module_name,
                    docstring=chunk.docstring,
                )
            )
            if chunk.kind == "module":
                table.roots.add(_module_name(rel).split(".")[0])

    logger.info(
        "symbol table built",
        extra={"files": len(source_files), "symbols": len(table.symbols)},
    )
    return table


def _module_name(rel_path: str) -> str:
    parts = list(Path(rel_path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _relpath(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
