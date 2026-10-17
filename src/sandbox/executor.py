"""异步命令执行、超时强杀、内存溢出 (OOM) 监控与日志压缩。

工程要点（对应文档《避坑指南》第 2 条）：

* **异步流式读取** stdout/stderr，避免管道写满导致死锁；
* **超时强杀**：超过 ``timeout`` 秒杀死整个进程树（含子进程）；
* **OOM 监控**：通过 psutil 轮询进程树 RSS，超过内存上限 + 宽限值即强杀并标记；
* **正则过滤**：``pytest`` 失败时终端输出可能数万行，内置过滤器只保留栈顶/栈底
  关键 ``AssertionError`` / ``Traceback`` 核心帧，防止上下文膨胀。
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import get_logger

logger = get_logger(__name__)

try:  # psutil 为可选增强
    import psutil
except Exception:  # pragma: no cover
    psutil = None


@dataclass(slots=True)
class CommandResult:
    """命令执行结果。"""

    command: str
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    oom: bool = False
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.oom

    @property
    def output(self) -> str:
        return self.stdout + ("\n" + self.stderr if self.stderr else "")

    def summary(self, max_chars: int = 4000) -> str:
        body = compress_test_output(self.output)
        body = body[:max_chars]
        flag = ""
        if self.timed_out:
            flag = " [TIMEOUT]"
        elif self.oom:
            flag = " [OOM]"
        return f"$ {self.command}\n(exit={self.returncode}{flag})\n{body}".strip()


async def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """强杀进程及其全部子进程。"""
    if proc.returncode is not None:
        return
    if psutil is not None:
        try:
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    continue
            parent.kill()
            return
        except Exception:
            pass
    try:
        if os.name == "nt":  # pragma: no cover - Windows
            proc.kill()
        else:
            proc.send_signal(getattr(signal, "SIGKILL", signal.SIGTERM))
    except ProcessLookupError:
        pass


def _close_transport(proc: asyncio.subprocess.Process) -> None:
    """显式关闭子进程传输层，释放滞留的管道。

    孙进程持有管道写端时读端永远等不到 EOF，传输对象会滞留到 GC；
    若期间事件循环已关闭，``__del__`` 会在已关闭的 loop 上 ``call_soon``
    抛 RuntimeError（pytest 报 PytestUnraisableExceptionWarning）。
    """
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        try:
            transport.close()
        except Exception:  # pragma: no cover - 关闭失败可忽略
            pass


def _rss_mb(proc: asyncio.subprocess.Process) -> float:
    if psutil is None:
        return 0.0
    try:
        process = psutil.Process(proc.pid)
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except Exception:
                continue
        return total / (1024 * 1024)
    except Exception:
        return 0.0


async def run_command(
    command: str | Sequence[str],
    *,
    cwd: str | Path | None = None,
    timeout: float = 120.0,
    env: Mapping[str, str] | None = None,
    max_output_bytes: int = 200_000,
    memory_limit_mb: int | None = None,
    oom_grace_mb: int = 512,
    shell: bool = False,
    poll_interval: float = 0.5,
    stall_grace_s: float = 5.0,
) -> CommandResult:
    """异步执行命令并返回结构化结果。

    约定：

    * **字符串命令**：通过 shell 执行（等价 ``shell=True``）；
    * **序列命令**：直接 exec 不经 shell；此形态再传 ``shell=True`` 会抛
      ``ValueError``（避免 ``" ".join`` 丢失引号导致含空格路径执行错乱）；
    * **超时兜底**：``timeout`` 由看门狗强杀进程树；``stall_grace_s`` 是
      「子进程已退出、但孙进程仍持有 stdout 管道」导致读取永久挂起的兜底，
      超时后取消读取器并强杀残余进程，已捕获的输出保留并置
      ``timed_out=True`` 标记结果不完整。
    """
    if not isinstance(command, str) and shell:
        raise ValueError("shell=True 仅支持字符串命令；序列命令请直接传入参数列表")
    cmd_str = command if isinstance(command, str) else " ".join(command)
    merged_env = {**os.environ, "PYTHONUNBUFFERED": "1", "LC_ALL": "C.UTF-8"}
    if env:
        merged_env.update(env)

    start = time.monotonic()
    if isinstance(command, str):
        proc = await asyncio.create_subprocess_shell(
            cmd_str,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    else:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    state: dict[str, Any] = {
        "timed_out": False,
        "oom": False,
        "truncated": False,
        "out": bytearray(),
        "err": bytearray(),
    }

    async def _reader(stream: Any, key: str) -> None:
        """持续读取输出到 ``state[key]``；任务被取消时已读内容仍然保留。"""
        if stream is None:
            return
        buf: bytearray = state[key]
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
            room = max_output_bytes - len(buf)
            if room > 0:
                buf.extend(chunk[:room])
            if len(chunk) > room:
                # 只要有任何字节被丢弃就置位（含「单块输出恰好超出上限后
                # 流结束」的路径，旧实现会把尾部最多 64KB 静默截掉而无标记）。
                state["truncated"] = True

    async def _watchdog() -> None:
        limit = (memory_limit_mb + oom_grace_mb) if memory_limit_mb else None
        deadline = start + timeout
        while proc.returncode is None:
            await asyncio.sleep(poll_interval)
            if proc.returncode is not None:
                return
            if time.monotonic() >= deadline:
                state["timed_out"] = True
                await _kill_tree(proc)
                return
            if limit is not None and _rss_mb(proc) > limit:
                state["oom"] = True
                await _kill_tree(proc)
                return

    out_task = asyncio.create_task(_reader(proc.stdout, "out"))
    err_task = asyncio.create_task(_reader(proc.stderr, "err"))
    # 1) 等待进程退出 / 超时强杀 / OOM 强杀
    await asyncio.gather(_watchdog(), return_exceptions=True)
    # 2) 输出读取兜底宽限：防止孙进程持有管道导致 read() 永远等不到 EOF
    try:
        await asyncio.wait_for(
            asyncio.gather(out_task, err_task, return_exceptions=True),
            timeout=stall_grace_s,
        )
    except asyncio.TimeoutError:
        # 孙进程持有管道写端时读端永不 EOF：取消读取并释放传输层管道，
        # 避免传输对象滞留到事件循环关闭后触发 __del__ 报错。
        state["timed_out"] = True
        await _kill_tree(proc)
        out_task.cancel()
        err_task.cancel()
        await asyncio.gather(out_task, err_task, return_exceptions=True)
        _close_transport(proc)
    # 3) 进程句柄收尾（正常路径毫秒级返回）
    try:
        await asyncio.wait_for(proc.wait(), timeout=stall_grace_s)
    except asyncio.TimeoutError:  # pragma: no cover - 极端场景
        state["timed_out"] = True
        await _kill_tree(proc)
        _close_transport(proc)

    duration = time.monotonic() - start
    reader_out = bytes(state["out"][:max_output_bytes]).decode("utf-8", "replace")
    reader_err = bytes(state["err"][:max_output_bytes]).decode("utf-8", "replace")

    result = CommandResult(
        command=cmd_str,
        returncode=proc.returncode if proc.returncode is not None else -1,
        stdout=reader_out,
        stderr=reader_err,
        duration_s=duration,
        timed_out=state["timed_out"],
        oom=state["oom"],
        truncated=state["truncated"],
    )
    logger.debug(
        "command finished",
        extra={
            "command": cmd_str,
            "exit": result.returncode,
            "duration_s": round(duration, 3),
            "timed_out": result.timed_out,
            "oom": result.oom,
        },
    )
    return result


# --------------------------------------------------------------------------- #
# 日志压缩 / Traceback 提取
# --------------------------------------------------------------------------- #

_TRACEBACK_HEAD = re.compile(r"^Traceback \(most recent call last\):", re.MULTILINE)
_FRAME_RE = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>.+)$')
_ERROR_LINE_RE = re.compile(
    r"^\s*(?P<exc>[A-Za-z_][\w.]*(?:Error|Exception|Warning|Exit|Interrupt|Failure))"
    r"(?::\s*(?P<msg>.*))?$"
)
_PYTEST_FAIL_RE = re.compile(r"^(FAILED|ERROR)\s+(?P<test>\S+)", re.MULTILINE)
_PYTEST_SUMMARY_RE = re.compile(r"^=+\s+(?P<summary>.*?(passed|failed|error).*?)\s+=+$", re.MULTILINE | re.IGNORECASE)
_ASSERTION_RE = re.compile(r"^E\s+.*$", re.MULTILINE)


def extract_traceback(text: str, *, max_frames: int = 12, max_chars: int = 3000) -> str:
    """从原始输出中提取关键 Traceback 片段（栈顶 + 栈底 + 异常行）。"""
    if not text:
        return ""
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "Traceback (most recent call last):" in line:
            header_idx = i
            break

    collected: list[str] = []
    if header_idx is not None:
        block = lines[header_idx:]
        frames = [ln for ln in block if _FRAME_RE.match(ln)]
        errors = [ln for ln in block if _ERROR_LINE_RE.match(ln.strip())]
        # 栈顶前 3 帧 + 栈底后 3 帧
        head = frames[:3]
        tail = frames[-3:] if len(frames) > 3 else []
        collected.append("Traceback (most recent call last):")
        collected.extend(f"  {ln.strip()}" for ln in head)
        if len(frames) > 6:
            collected.append(f"  ... ({len(frames) - 6} frames omitted) ...")
        collected.extend(f"  {ln.strip()}" for ln in tail if ln not in head)
        if errors:
            collected.append(errors[-1].strip())
    else:
        collected.extend(ln.strip() for ln in lines if _ERROR_LINE_RE.match(ln.strip()))

    if not collected:
        collected = lines[-max_frames:]

    result = "\n".join(collected[:max_frames])
    if len(result) > max_chars:
        result = result[:max_chars] + "\n... (truncated)"
    return result


def extract_frames(text: str) -> list[tuple[str, int, str]]:
    """提取 Traceback 中所有 ``(file, line, func)`` 帧，供故障定位使用。"""
    frames: list[tuple[str, int, str]] = []
    for line in text.splitlines():
        m = _FRAME_RE.match(line)
        if m:
            frames.append((m.group("file"), int(m.group("line")), m.group("func")))
    return frames


def compress_test_output(text: str, *, max_chars: int = 4000) -> str:
    """压缩 pytest / 终端输出：只保留失败清单、Traceback 与总结行。"""
    if not text:
        return ""
    if len(text) <= max_chars and "FAILURES" not in text:
        return text

    parts: list[str] = []

    tb = extract_traceback(text)
    if tb:
        parts.append(tb)

    failures = [f"{m.group(1)} {m.group('test')}" for m in _PYTEST_FAIL_RE.finditer(text)]
    if failures:
        parts.insert(0, "Failed tests:\n" + "\n".join(f"  {f}" for f in dedupe(failures)))

    assertion_lines = dedupe(_ASSERTION_RE.findall(text))[:10]
    if assertion_lines:
        parts.append("Assertions:\n" + "\n".join(assertion_lines))

    # 注意：该正则含两个捕获组（summary + 状态词），findall 会返回元组，
    # 必须用 finditer 取命名组，否则 .strip() 会在 tuple 上崩溃。
    summaries = dedupe([m.group("summary").strip() for m in _PYTEST_SUMMARY_RE.finditer(text)])
    if summaries:
        parts.append("Summary: " + "; ".join(summaries[-2:]))

    if not parts:
        return text[:max_chars]

    joined = "\n\n".join(parts)
    return joined[:max_chars]


def dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def fingerprint_error(text: str, *, max_frames: int = 6) -> str:
    """对报错堆栈计算稳定指纹哈希，用于死循环检测。

    指纹刻意**忽略行号**（代码修改会导致行号漂移），仅保留文件名、函数名与
    规范化后的异常信息，从而稳定识别「同一种错误是否在重复出现」。
    """
    import hashlib

    frames = extract_frames(text)[-max_frames:]
    core = extract_traceback(text)
    frame_sig = "|".join(f"{f}:{fn}" for f, _l, fn in frames)
    payload = frame_sig + "||" + _normalize(core)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _normalize(text: str) -> str:
    """去除易变内容（行号、内存地址、耗时）以稳定指纹。"""
    text = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
    text = re.sub(r"\b\d+(\.\d+)?\s*(s|ms|seconds)\b", "TIME", text)
    text = re.sub(r"\b\d+\b", "N", text)
    return text.strip()
