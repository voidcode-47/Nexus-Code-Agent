"""LSP 桥接器测试（语法引擎路径，无需 Pyright）。"""

from __future__ import annotations

from src.mcp_tools.lsp_bridge import Diagnostic, LSPBridge, LSPReport, run_static_checks


async def test_check_valid_file(tmp_path):
    good = tmp_path / "good.py"
    good.write_text("def f():\n    return 1\n", encoding="utf-8")
    bridge = LSPBridge(pyright_cmd=None)
    report = await bridge.check([good], root=tmp_path, use_pyright=False)
    assert report.passed
    assert report.error_count == 0
    assert report.engine == "syntax"


async def test_check_syntax_error_file(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def f(:\n    return 1\n", encoding="utf-8")
    report = await run_static_checks([bad], root=tmp_path)
    assert not report.passed
    assert report.error_count >= 1
    assert "bad" in report.summary() or report.errors


async def test_check_missing_file(tmp_path):
    report = await run_static_checks(["__nope__.py"], root=tmp_path)
    assert not report.passed
    assert report.errors


async def test_check_mixed_existing_and_missing(tmp_path):
    """部分文件缺失时：缺失文件报 error，存在文件仍正常检查（pyright 路径同样生效）。"""
    good = tmp_path / "good.py"
    good.write_text("def f():\n    return 1\n", encoding="utf-8")
    bridge = LSPBridge()  # 不禁用 pyright：CI 上装有 pyright 也必须拦截缺失文件
    report = await bridge.check([good, "__nope__.py"], root=tmp_path)
    assert not report.passed
    assert any(d.message == "文件不存在" for d in report.errors)
    assert not any("good" in d.message for d in report.errors)


def test_report_summary_formats():
    report = LSPReport(
        passed=False,
        diagnostics=[Diagnostic("a.py", 1, 2, "error", "boom", "rule-x")],
        engine="pyright",
    )
    text = report.summary()
    assert "pyright" in text
    assert "boom" in text
    assert report.error_count == 1
    assert report.errors[0].render().startswith("a.py:1:2")


def test_last_duration_recorded(tmp_path):
    good = tmp_path / "g.py"
    good.write_text("x = 1\n", encoding="utf-8")
    import asyncio

    bridge = LSPBridge(pyright_cmd=None)
    asyncio.run(bridge.check([good], root=tmp_path, use_pyright=False))
    assert bridge.last_duration >= 0.0
