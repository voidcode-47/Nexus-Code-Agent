"""MCP 工具服务端测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.mcp_tools.server import _safe_path, build_server


def test_safe_path_allows_inside(tmp_path: Path):
    target = _safe_path(tmp_path, "pkg/mod.py")
    assert str(target).startswith(str(tmp_path.resolve()))


def test_safe_path_blocks_escape(tmp_path: Path):
    with pytest.raises(ValueError):
        _safe_path(tmp_path, "../outside.py")


def test_build_server_registers_tools(tmp_path: Path):
    pytest.importorskip("fastmcp")
    server = build_server(tmp_path)
    assert server is not None
    # FastMCP 实例应暴露工具注册信息
    tool_names = set()
    for attr in ("_tools", "tools"):
        tools = getattr(server, attr, None)
        if isinstance(tools, dict):
            tool_names.update(tools.keys())
    if tool_names:
        assert {"apply_patch", "read_file", "run_tests"} & tool_names
        assert "create_pull_request" in tool_names
