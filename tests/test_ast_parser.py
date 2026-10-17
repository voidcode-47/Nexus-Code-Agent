"""AST 解析器测试。"""

from __future__ import annotations

from src.indexer.ast_parser import parse_file, parse_source, supported_languages

SOURCE = '''
"""模块文档。"""
import os
from pathlib import Path


class Greeter:
    """打招呼。"""

    def greet(self, name: str) -> str:
        return f"hi {name}"


async def fetch(url):
    return await load(url)


def load(url):
    return Path(url).read_text()
'''


def test_python_supported():
    assert "python" in supported_languages()


def test_parse_source_extracts_symbols():
    result = parse_source(SOURCE, "pkg/mod.py")
    assert result.ok
    kinds = {c.symbol: c.kind for c in result.chunks}
    assert kinds["pkg.mod.Greeter"] == "class"
    assert kinds["pkg.mod.Greeter.greet"] == "method"
    assert kinds["pkg.mod.fetch"] == "function"
    assert kinds["pkg.mod.load"] == "function"
    assert any(c.kind == "module" for c in result.chunks)


def test_docstring_and_signature():
    result = parse_source(SOURCE, "pkg/mod.py")
    greeter = next(c for c in result.chunks if c.symbol.endswith("Greeter"))
    assert greeter.docstring == "打招呼。"
    greet = next(c for c in result.chunks if c.symbol.endswith("greet"))
    assert greet.signature.startswith("def greet(")
    assert "name: str" in greet.signature
    fetch = next(c for c in result.chunks if c.symbol.endswith("fetch"))
    assert fetch.signature.startswith("async def fetch(")


def test_function_docstring_extracted():
    source = 'def add(a, b):\n    """两数之和。"""\n    return a + b\n'
    result = parse_source(source, "calc.py")
    add = next(c for c in result.chunks if c.symbol.endswith("add"))
    assert add.docstring == "两数之和。"


def test_calls_extracted():
    result = parse_source(SOURCE, "pkg/mod.py")
    fetch = next(c for c in result.chunks if c.symbol.endswith("fetch"))
    assert "load" in fetch.calls


def test_imports_recorded():
    result = parse_source(SOURCE, "pkg/mod.py")
    modules = {imp.module for imp in result.imports}
    assert "os" in modules
    assert "pathlib" in modules


def test_syntax_error_returned():
    result = parse_source("def broken(:\n  pass", "bad.py")
    assert result.ok is False
    assert result.error


def test_unsupported_extension():
    result = parse_source("hello", "notes.txt")
    assert result.ok is False


def test_parse_file(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("def f():\n    return 1\n", encoding="utf-8")
    result = parse_file(p)
    assert result.ok
    assert any(c.name == "f" for c in result.chunks)


def test_treesitter_parser_extracts_signature_and_docstring():
    """Tree-sitter 扩展位：签名与紧邻注释（Docstring）均应提取。

    使用 tree-sitter-python 语法包驱动多语言解析器的通用机制；
    grammar 未安装时跳过（可选依赖）。
    """
    import pytest

    pytest.importorskip("tree_sitter_python")
    from src.indexer.ast_parser import TreeSitterParser

    parser = TreeSitterParser("python", (".py",), "tree_sitter_python")
    source = (
        "# Adds two numbers together\n"
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n\n\n"
        "class Calc:\n"
        "    def mul(self, x):\n"
        "        return x * 2\n"
    )
    result = parser.parse(source, "calc.py")
    chunks = {c.name: c for c in result.chunks}
    assert chunks["add"].kind == "function"
    assert chunks["add"].signature == "def add(a: int, b: int) -> int"
    assert chunks["add"].docstring == "Adds two numbers together"
    assert chunks["Calc"].kind == "class"
    assert chunks["mul"].kind == "function"
    assert chunks["mul"].signature == "def mul(self, x)"
