"""配置与日志测试。"""

from __future__ import annotations

import json
import logging

from src.config.logging_config import JsonFormatter, get_trace_id, new_trace_id, set_trace_id
from src.config.settings import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.app_name == "Nexus-Code-Agent"
    assert s.agent.max_iterations >= 1
    assert s.sandbox.backend in {"local", "docker"}


def test_env_override(monkeypatch):
    monkeypatch.setenv("NEXUS_ENV", "prod")
    monkeypatch.setenv("NEXUS_API__PORT", "9999")
    monkeypatch.setenv("NEXUS_SANDBOX__POOL_SIZE", "7")
    s = Settings(_env_file=None)
    assert s.env == "prod"
    assert s.is_prod
    assert s.api.port == 9999
    assert s.sandbox.pool_size == 7


def test_cors_list():
    s = Settings(_env_file=None, api={"cors_origins": "a.com, b.com"})
    assert s.api.cors_origin_list == ["a.com", "b.com"]
    s2 = Settings(_env_file=None, api={"cors_origins": "*"})
    assert s2.api.cors_origin_list == ["*"]


def test_resolve_path(tmp_path, monkeypatch):
    s = Settings(_env_file=None)
    resolved = s.resolve_path("relative/dir")
    assert resolved.is_absolute()


def test_trace_id_context():
    tid = new_trace_id()
    assert get_trace_id() == tid
    set_trace_id("fixed-trace")
    assert get_trace_id() == "fixed-trace"


def test_json_formatter_emits_trace_id():
    set_trace_id("trace-123")
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello %s", args=("world",), exc_info=None,
    )
    record.trace_id = get_trace_id()
    record.task_id = "-"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "hello world"
    assert payload["trace_id"] == "trace-123"
    assert payload["level"] == "INFO"
