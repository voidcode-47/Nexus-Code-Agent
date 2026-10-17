"""LLM 客户端抽象：真实模型（OpenAI 兼容）与离线启发式降级。

* :class:`OpenAICompatClient` —— 通过 ``/chat/completions`` 调用任意 OpenAI 兼容
  服务（OpenAI / DeepSeek / 本地 vLLM 等）；
* :class:`HeuristicClient` —— 离线占位实现，``enabled=False``，节点会自动切换到
  **确定性修复策略**（基于 AST 变异的程序修复），保证无 API Key 时也能跑通闭环。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from src.config import get_logger
from src.config.settings import LLMSettings
from src.config.settings import settings as global_settings

logger = get_logger(__name__)


class LLMClient(Protocol):
    """LLM 客户端协议。"""

    enabled: bool
    model: str

    async def complete(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str: ...


@dataclass
class HeuristicClient:
    """离线启发式客户端（不发起任何网络请求）。"""

    model: str = "heuristic"
    enabled: bool = False

    async def complete(  # noqa: D102
        self,
        messages: Sequence[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        return ""


@dataclass
class OpenAICompatClient:
    """OpenAI Chat Completions 兼容客户端。"""

    settings: LLMSettings
    model: str = ""
    enabled: bool = True

    def __post_init__(self) -> None:
        self.model = self.model or self.settings.model
        self._client = httpx.AsyncClient(
            base_url=self.settings.base_url.rstrip("/"),
            timeout=self.settings.request_timeout_s,
            headers={"Authorization": f"Bearer {self.settings.api_key}"},
        )

    async def complete(  # noqa: D102
        self,
        messages: Sequence[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.settings.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = await self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("LLM 调用失败", extra={"error": str(exc)})
            raise
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def aclose(self) -> None:
        await self._client.aclose()


def get_llm(settings: LLMSettings | None = None) -> LLMClient:
    """按配置构造 LLM 客户端。"""
    cfg = settings or global_settings.llm
    if cfg.provider == "heuristic" or not cfg.api_key:
        if cfg.provider != "heuristic":
            logger.warning("provider=%s 但缺少 api_key，降级为启发式客户端", cfg.provider)
        return HeuristicClient()
    return OpenAICompatClient(cfg)


def parse_json_block(text: str) -> dict[str, Any]:
    """从可能包含 Markdown 围栏的 LLM 输出中稳健地解析 JSON。"""
    if not text:
        return {}
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
    return {}
