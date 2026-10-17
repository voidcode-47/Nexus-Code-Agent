"""LLM 客户端与提示词测试。"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from src.agent.llm import HeuristicClient, OpenAICompatClient, get_llm, parse_json_block
from src.agent.prompts import (
    architect_messages,
    coder_messages,
    critic_messages,
    localizer_messages,
    render_snippets,
)
from src.config.settings import LLMSettings


async def test_heuristic_client_disabled():
    client = HeuristicClient()
    assert client.enabled is False
    assert await client.complete([{"role": "user", "content": "hi"}]) == ""


def test_get_llm_defaults_to_heuristic():
    client = get_llm(LLMSettings(provider="heuristic", api_key=""))
    assert client.enabled is False


def test_get_llm_without_key_falls_back():
    client = get_llm(LLMSettings(provider="openai", api_key=""))
    assert client.enabled is False


def test_parse_json_block_plain():
    assert parse_json_block('{"a": 1}') == {"a": 1}


def test_parse_json_block_fenced():
    raw = '前言\n```json\n{"root_cause": "x"}\n```\n后记'
    assert parse_json_block(raw)["root_cause"] == "x"


def test_parse_json_block_invalid():
    assert parse_json_block("not json at all") == {}


def test_prompt_builders():
    loc = localizer_messages("issue", "tb", "candidates")
    assert loc[0]["role"] == "system"
    assert "JSON" in loc[0]["content"]

    arch = architect_messages("issue", "loc", "graph")
    assert "修复计划" in arch[1]["content"]

    coder = coder_messages("plan", "calc.py", "def f(): pass", feedback="failed", attempt=1)
    assert "calc.py" in coder[1]["content"]
    assert "第 2 次" in coder[1]["content"]

    critic = critic_messages("tb", "hist", 2)
    assert "第 2 次" in critic[1]["content"]


def test_render_snippets_truncates():
    class _Hit:
        symbol = "s"
        path = "p.py"
        start_line = 1
        end_line = 2

        def snippet(self, max_lines: int = 12) -> str:
            return "code\n" * 100

    text = render_snippets([_Hit()], max_chars=200)
    assert len(text) <= 400


# --------------------------------------------------------------------------- #
# OpenAICompatClient（httpx.MockTransport，无真实网络）
# --------------------------------------------------------------------------- #


def _install_mock_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """把 httpx.AsyncClient 替换为挂载 MockTransport 的等价客户端。"""
    real = httpx.AsyncClient

    def factory(**kwargs: Any) -> httpx.AsyncClient:
        return real(
            transport=httpx.MockTransport(handler),
            base_url=kwargs.get("base_url", ""),
            timeout=kwargs.get("timeout"),
            headers=kwargs.get("headers"),
        )

    monkeypatch.setattr(httpx, "AsyncClient", factory)


async def test_openai_client_complete_success(monkeypatch):
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": "补丁已生成"}}]})

    _install_mock_transport(monkeypatch, handler)
    client = OpenAICompatClient(LLMSettings(provider="openai", api_key="sk-test"))
    try:
        text = await client.complete([{"role": "user", "content": "hi"}], json_mode=True)
    finally:
        await client.aclose()
    assert text == "补丁已生成"
    assert seen["url"].endswith("/chat/completions")


async def test_openai_client_http_error_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    _install_mock_transport(monkeypatch, handler)
    client = OpenAICompatClient(LLMSettings(provider="openai", api_key="sk-test"))
    try:
        with pytest.raises(httpx.HTTPError):
            await client.complete([{"role": "user", "content": "hi"}])
    finally:
        await client.aclose()
