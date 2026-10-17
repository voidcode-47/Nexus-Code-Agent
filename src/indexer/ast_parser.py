"""多语言 Tree-sitter AST 解析器。

设计要点：

* **策略模式（Strategy Pattern）**：每种语言实现一个 :class:`LanguageParser`，
  通过 :func:`register_parser` 注册，第一阶段 100% 聚焦 Python（stdlib ``ast`` 实现，
  零外部依赖且健壮），同时预留 ``tree-sitter`` 驱动的 TS/Go 扩展位。
* **AST 结构化切片**：摒弃按字符固定切分，按类/函数边界提取代码块，
  保留完整签名、Docstring、装饰器与调用点，便于后续图谱构建与混合检索。
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ImportRecord:
    """单条 import 记录。"""

    module: str  # 被导入模块 / 包；相对导入时可能为空
    names: list[str]  # 导入的符号名（``import a.b`` 时为 ["a.b"]）
    level: int = 0  # 相对导入层级（``from . import x`` → level=1）
    alias: str = ""  # 顶层绑定别名（用于符号表解析）
    line: int = 0

    @property
    def is_relative(self) -> bool:
        return self.level > 0


@dataclass(slots=True)
class CodeChunk:
    """AST 结构化代码切片。"""

    chunk_id: str
    path: str  # 相对仓库根的 POSIX 路径
    language: str
    kind: str  # module | class | function | method
    symbol: str  # 限定名，例如 ``pkg.mod.Class.method``
    name: str  # 短名
    start_line: int
    end_line: int
    source: str
    signature: str = ""
    docstring: str = ""
    decorators: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)  # 类继承
    calls: list[str] = field(default_factory=list)  # 调用的（可能未限定）符号名（含重复，反映调用次数）

    @property
    def text_for_embedding(self) -> str:
        """构造用于向量化的文本（签名 + 文档 + 源码）。"""
        parts = [f"# {self.kind} {self.symbol}"]
        if self.signature:
            parts.append(self.signature)
        if self.docstring:
            parts.append(self.docstring)
        parts.append(self.source)
        return "\n".join(p for p in parts if p)


@dataclass(slots=True)
class ParseResult:
    """单个文件的解析结果。"""

    path: str
    language: str
    ok: bool
    chunks: list[CodeChunk] = field(default_factory=list)
    imports: list[ImportRecord] = field(default_factory=list)
    error: str = ""


# --------------------------------------------------------------------------- #
# 解析器接口与注册表
# --------------------------------------------------------------------------- #


class LanguageParser(Protocol):
    """语言解析器协议。"""

    language: str
    extensions: tuple[str, ...]

    def parse(self, source: str, path: str) -> ParseResult: ...


_REGISTRY: dict[str, LanguageParser] = {}


def register_parser(parser: LanguageParser, *, override: bool = False) -> None:
    """注册一个语言解析器。"""
    if parser.language in _REGISTRY and not override:
        return
    _REGISTRY[parser.language] = parser


def supported_languages() -> list[str]:
    return sorted(_REGISTRY)


def parser_for_path(path: str | Path) -> LanguageParser | None:
    suffix = Path(path).suffix.lower()
    for parser in _REGISTRY.values():
        if suffix in parser.extensions:
            return parser
    return None


# --------------------------------------------------------------------------- #
# Python 解析器（stdlib ast 实现）
# --------------------------------------------------------------------------- #


def _call_name(node: ast.AST) -> str:
    """尽力还原一个调用表达式的点分名称。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    return ""


def _unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - ast.unparse 极少失败
        return ""


class PythonParser:
    """基于标准库 ``ast`` 的 Python 解析器。"""

    language = "python"
    extensions: tuple[str, ...] = (".py", ".pyi")

    def parse(self, source: str, path: str) -> ParseResult:
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError as exc:
            return ParseResult(path=path, language=self.language, ok=False, error=str(exc))

        lines = source.splitlines()
        chunks: list[CodeChunk] = []
        imports: list[ImportRecord] = []

        module_name = _module_name_from_path(path)
        chunks.append(
            CodeChunk(
                chunk_id=f"{path}::<module>",
                path=path,
                language=self.language,
                kind="module",
                symbol=module_name,
                name=Path(path).stem,
                start_line=1,
                end_line=max(1, len(lines)),
                source="\n".join(lines[:1]) if lines else "",
                docstring=ast.get_docstring(tree) or "",
            )
        )

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(_import_record(node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append(self._function_chunk(node, path, module_name, lines))
            elif isinstance(node, ast.ClassDef):
                chunks.append(self._class_chunk(node, path, module_name, lines))
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        chunks.append(
                            self._function_chunk(
                                item,
                                path,
                                f"{module_name}.{node.name}",
                                lines,
                                kind="method",
                            )
                        )

        return ParseResult(
            path=path,
            language=self.language,
            ok=True,
            chunks=chunks,
            imports=imports,
        )

    # -- helpers ---------------------------------------------------------- #

    def _slice(self, node: ast.AST, lines: list[str]) -> str:
        start = getattr(node, "lineno", 1) - 1
        end = getattr(node, "end_lineno", start + 1) or start + 1
        return "\n".join(lines[start:end])

    def _function_chunk(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        path: str,
        parent: str,
        lines: list[str],
        kind: str = "function",
    ) -> CodeChunk:
        symbol = f"{parent}.{node.name}" if parent else node.name
        decorators = [_unparse(d) for d in node.decorator_list]
        # 保留重复调用名（不去重）：调用图边权统计的是真实调用次数
        calls = sorted(
            call
            for c in ast.walk(node)
            if isinstance(c, ast.Call) and (call := _call_name(c.func))
        )
        return CodeChunk(
            chunk_id=f"{path}::{symbol}",
            path=path,
            language=self.language,
            kind=kind,
            symbol=symbol,
            name=node.name,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            source=self._slice(node, lines),
            signature=_function_signature(node),
            docstring=ast.get_docstring(node) or "",
            decorators=decorators,
            calls=calls,
        )

    def _class_chunk(
        self,
        node: ast.ClassDef,
        path: str,
        parent: str,
        lines: list[str],
    ) -> CodeChunk:
        symbol = f"{parent}.{node.name}" if parent else node.name
        bases = [_unparse(b) for b in node.bases]
        methods = sorted(
            m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        return CodeChunk(
            chunk_id=f"{path}::{symbol}",
            path=path,
            language=self.language,
            kind="class",
            symbol=symbol,
            name=node.name,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            source=self._slice(node, lines),
            signature=f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}",
            docstring=ast.get_docstring(node) or "",
            decorators=[_unparse(d) for d in node.decorator_list],
            bases=bases,
            calls=[f"{symbol}.{m}" for m in methods],
        )


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {_unparse(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({_unparse(node.args)}){returns}"


def _module_name_from_path(path: str) -> str:
    p = Path(path)
    parts = list(p.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _import_record(node: ast.Import | ast.ImportFrom) -> ImportRecord:
    if isinstance(node, ast.Import):
        names = [a.name for a in node.names]
        alias = node.names[0].asname or node.names[0].name.split(".")[0]
        return ImportRecord(module=names[0], names=names, alias=alias, line=node.lineno)
    module = node.module or ""
    names = [a.name for a in node.names]
    alias = node.names[0].asname or node.names[0].name
    return ImportRecord(
        module=module, names=names, level=node.level, alias=alias, line=node.lineno
    )


# --------------------------------------------------------------------------- #
# 可选：Tree-sitter 驱动的多语言解析器扩展位
# --------------------------------------------------------------------------- #


class TreeSitterParser:
    """通用 Tree-sitter 解析器（TS/JS/Go 扩展位）。

    仅在对应 grammar 包安装时可用；否则 :func:`parser_for_path` 不会命中。
    """

    #: node type → chunk kind
    _SYMBOL_NODES = {
        "function_declaration": "function",
        "function_definition": "function",
        "method_definition": "method",
        "method_declaration": "method",
        "class_declaration": "class",
        "class_definition": "class",
        "arrow_function": "function",
    }

    def __init__(self, language: str, extensions: tuple[str, ...], module_name: str) -> None:
        import importlib

        from tree_sitter import Language, Parser

        grammar = importlib.import_module(module_name)
        self._language = Language(grammar.language())
        self._parser = Parser(self._language)
        self.language = language
        self.extensions = extensions

    def _name_of(self, node) -> str:
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return name_node.text.decode("utf-8", "replace")
        return ""

    def _signature_of(self, node, data: bytes, text: str) -> str:
        """取「头部到函数体之间」的文本作为签名（保留参数与返回类型）。"""
        body = node.child_by_field_name("body")
        if body is not None and body.start_byte > node.start_byte:
            raw = data[node.start_byte : body.start_byte].decode("utf-8", "replace")
        else:
            raw = text.splitlines()[0] if text else ""
        return " ".join(raw.split()).rstrip("{:").strip()

    def _docstring_of(self, node, data: bytes) -> str:
        """紧邻声明前的注释块视为 Docstring（JSDoc / Go doc / Python 注释）。"""
        prev = node.prev_sibling
        if prev is not None and prev.type in {"comment", "line_comment", "block_comment"}:
            raw = data[prev.start_byte : prev.end_byte].decode("utf-8", "replace")
            return _clean_comment(raw)
        return ""

    def parse(self, source: str, path: str) -> ParseResult:  # noqa: D102
        data = source.encode("utf-8")
        tree = self._parser.parse(data)
        chunks: list[CodeChunk] = []
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            kind = self._SYMBOL_NODES.get(node.type)
            if kind:
                symbol = self._name_of(node) or node.type
                text = data[node.start_byte : node.end_byte].decode("utf-8", "replace")
                chunks.append(
                    CodeChunk(
                        chunk_id=f"{path}::{symbol}",
                        path=path,
                        language=self.language,
                        kind=kind,
                        symbol=symbol,
                        name=symbol,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        signature=self._signature_of(node, data, text),
                        docstring=self._docstring_of(node, data),
                        source=text,
                    )
                )
            stack.extend(node.children)
        return ParseResult(path=path, language=self.language, ok=True, chunks=chunks)


_COMMENT_PREFIXES = ("///", "//!", "//", "/*!", "/*", "#")


def _clean_comment(text: str) -> str:
    """去除注释标记，压缩为纯文本。"""
    cleaned = text.strip()
    for prefix in _COMMENT_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    if cleaned.rstrip().endswith("*/"):
        cleaned = cleaned.rstrip()[:-2]
    lines = [ln.strip().lstrip("*").strip() for ln in cleaned.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


def _register_defaults() -> None:
    register_parser(PythonParser())
    # 预留多语言扩展位（grammar 缺失时静默跳过）
    for lang, exts, mod in (
        ("typescript", (".ts", ".tsx"), "tree_sitter_typescript"),
        ("javascript", (".js", ".jsx"), "tree_sitter_javascript"),
        ("go", (".go",), "tree_sitter_go"),
    ):
        try:
            register_parser(TreeSitterParser(lang, exts, mod))
        except Exception:  # pragma: no cover - grammar 未安装
            continue


_register_defaults()


# --------------------------------------------------------------------------- #
# 公共 API
# --------------------------------------------------------------------------- #


def parse_source(source: str, path: str) -> ParseResult:
    """解析给定源码字符串。"""
    parser = parser_for_path(path)
    if parser is None:
        return ParseResult(
            path=path, language="unknown", ok=False, error=f"unsupported file type: {path}"
        )
    return parser.parse(source, path)


def parse_file(path: str | Path) -> ParseResult:
    """解析磁盘文件。"""
    p = Path(path)
    try:
        source = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return ParseResult(path=str(p), language="unknown", ok=False, error=str(exc))
    return parse_source(source, str(p).replace("\\", "/"))


def iter_python_files(root: str | Path, ignored: Iterable[str] = ()) -> Iterable[Path]:
    """遍历仓库中的 Python 源文件（跳过常见噪声目录）。"""
    root_path = Path(root)
    default_skip = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".nexus",
        "node_modules",
        "build",
        "dist",
    }
    skip = default_skip | set(ignored)
    for p in root_path.rglob("*.py"):
        # 只用相对 root 的路径参与过滤：绝对路径包含仓库的祖先目录，
        # 检出路径中出现 build/dist/venv 等名字时会把整个仓库静默判空
        if any(part in skip for part in p.relative_to(root_path).parts):
            continue
        yield p
