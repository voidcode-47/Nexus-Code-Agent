"""结构化 JSON 日志与 TraceID 链路追踪注入。

使用 :func:`setup_logging` 初始化根日志器；使用 :func:`get_logger` 获取带
TraceID 自动注入能力的模块日志器。TraceID 通过 :class:`contextvars.ContextVar`
在异步任务间安全传播。
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import uuid
from typing import Any

_TRACE_ID: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")
_TASK_ID: contextvars.ContextVar[str] = contextvars.ContextVar("task_id", default="-")


def new_trace_id() -> str:
    """生成并绑定一个新的 TraceID 到当前上下文。"""
    tid = uuid.uuid4().hex[:16]
    _TRACE_ID.set(tid)
    return tid


def set_trace_id(trace_id: str | None) -> str:
    """显式设置当前上下文的 TraceID（不存在则自动生成）。"""
    tid = trace_id or uuid.uuid4().hex[:16]
    _TRACE_ID.set(tid)
    return tid


def get_trace_id() -> str:
    return _TRACE_ID.get()


def set_task_id(task_id: str) -> None:
    _TASK_ID.set(task_id or "-")


def get_task_id() -> str:
    return _TASK_ID.get()


class _ContextFilter(logging.Filter):
    """向日志记录注入 trace_id / task_id 字段。"""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.trace_id = _TRACE_ID.get()
        record.task_id = _TASK_ID.get()
        return True


class JsonFormatter(logging.Formatter):
    """输出单行结构化 JSON。"""

    _RESERVED = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
        "trace_id",
        "task_id",
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": getattr(record, "trace_id", "-"),
            "task_id": getattr(record, "task_id", "-"),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """人类可读的彩色控制台格式（dev 环境）。"""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        base = (
            f"{self.formatTime(record, '%H:%M:%S')} "
            f"{record.levelname:<7} "
            f"[{getattr(record, 'trace_id', '-')}] "
            f"{record.name}: {record.getMessage()}"
        )
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging(
    level: str = "INFO",
    *,
    json_output: bool = True,
    force: bool = False,
) -> None:
    """初始化根日志器。

    Args:
        level: 日志等级字符串。
        json_output: True 输出结构化 JSON，False 输出人类可读格式。
        force: 是否强制重建已存在的 handler。
    """
    root = logging.getLogger()
    if root.handlers and not force:
        root.setLevel(level.upper())
        return

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_ContextFilter())
    handler.setFormatter(JsonFormatter() if json_output else ConsoleFormatter())

    root.addHandler(handler)
    root.setLevel(level.upper())

    # 降低三方库噪声
    for noisy in ("httpx", "httpcore", "urllib3", "docker", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str, **initial: Any) -> logging.LoggerAdapter:
    """返回带默认字段的 LoggerAdapter。"""
    return logging.LoggerAdapter(logging.getLogger(name), initial)
