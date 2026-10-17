"""CLI 冒烟测试（Typer CliRunner，离线）。"""

from __future__ import annotations

from typer.testing import CliRunner

from src.cli import app

runner = CliRunner()


def test_cli_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("heal", "index", "evaluate", "serve", "mcp"):
        assert cmd in result.output


def test_cli_heal_requires_issue():
    result = runner.invoke(app, ["heal", "--repo", "."])
    assert result.exit_code != 0  # 缺少必填的 --issue


def test_cli_index_prints_core_ranking(tmp_path):
    (tmp_path / "calc.py").write_text(
        'def add(a, b):\n    """两数之和。"""\n    return a + b\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["index", "--repo", str(tmp_path), "--query", "add"])
    assert result.exit_code == 0
    assert "PageRank" in result.output
    assert "add" in result.output
