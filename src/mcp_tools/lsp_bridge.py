"""对接 Pyright / LSP 静态语法检查（在跑单测前先拦截语法与类型错误）。

**静态左移（Shift-Left）**：每次打补丁后立即触发静态检查，平均 200ms 内返回
未定义变量、括号不匹配、类型不兼容等诊断信息。约 90% 的低级错误在消耗沙箱
与单测算力之前就被拦截，显著降低 LLM 重试 Token 成本。

* Pyright 可用时：``pyright --outputjson`` 获取结构化诊断；
* 不可用时：降级为 ``ast`` / ``py_compile`` 语法检查（"syntax-only 护栏"）。
"""

from __future__ import annotations

import ast
import json
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.config import get_logger
from src.sandbox.executor import run_command

logger = get_logger(__name__)

_SEVERITY = {"error": "error", "warning": "warning", "information": "info", "hint": "info"}


@dataclass(slots=True)
class Diagnostic:
    """单条静态诊断。"""

    file: str
    line: int
    column: int
    severity: str  # error | warning | info
    message: str
    rule: str = ""

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    def render(self) -> str:
        rule = f" [{self.rule}]" if self.rule else ""
        return f"{self.file}:{self.line}:{self.column} {self.severity}: {self.message}{rule}"


@dataclass(slots=True)
class LSPReport:
    """一次静态检查的汇总结果。"""

    passed: bool
    diagnostics: list[Diagnostic] = field(default_factory=list)
    engine: str = "none"
    duration_s: float = 0.0

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.is_error]

    @property
    def error_count(self) -> int:
        return len(self.errors)

    def summary(self, limit: int = 15) -> str:
        if self.passed:
            return f"[{self.engine}] 静态检查通过（{len(self.diagnostics)} 条非阻断告警）"
        head = self.errors[:limit]
        lines = [f"[{self.engine}] 静态检查失败：{self.error_count} 个错误"]
        lines.extend(f"  - {d.render()}" for d in head)
        if self.error_count > limit:
            lines.append(f"  ... 另有 {self.error_count - limit} 个错误")
        return "\n".join(lines)


class LSPBridge:
    """常驻静态检查桥接器。"""

    def __init__(self, *, pyright_cmd: str | None = None) -> None:
        self.pyright_cmd = pyright_cmd or self._detect_pyright()
        self._last_duration = 0.0

    @staticmethod
    def _detect_pyright() -> str | None:
        for name in ("pyright", "pyright-langserver", "basedpyright"):
            path = shutil.which(name)
            if path:
                return path
        return None

    @property
    def engine(self) -> str:
        return "pyright" if self.pyright_cmd else "syntax"

    def available(self) -> bool:
        return True  # 语法检查引擎始终可用

    async def check(
        self,
        files: Sequence[str | Path],
        *,
        root: str | Path = ".",
        use_pyright: bool = True,
        timeout: float = 60.0,
    ) -> LSPReport:
        """对指定文件执行静态检查。"""
        start = time.monotonic()
        paths = [str(Path(f)) for f in files]
        # pyright 对不存在的路径静默返回 0 错误（--outputjson 无诊断），
        # 缺失文件必须在此拦截，否则「补丁指向已删除文件」会被误判为通过
        resolved = [(f, Path(f) if Path(f).is_absolute() else Path(root) / f) for f in paths]
        missing = [orig for orig, p in resolved if not p.is_file()]
        existing = [orig for orig, p in resolved if p.is_file()]

        diagnostics = [Diagnostic(f, 0, 0, "error", "文件不存在", "io") for f in missing]
        engine = self.engine
        if existing:
            sub: LSPReport
            if use_pyright and self.pyright_cmd:
                sub = await self._check_pyright(existing, root=root, timeout=timeout)
            else:
                sub = await self._check_syntax(existing, root=root)
            diagnostics.extend(sub.diagnostics)
            engine = sub.engine
        report = LSPReport(
            passed=not any(d.is_error for d in diagnostics),
            diagnostics=diagnostics,
            engine=engine,
        )
        report.duration_s = time.monotonic() - start
        self._last_duration = report.duration_s
        logger.info(
            "static check finished",
            extra={
                "engine": report.engine,
                "errors": report.error_count,
                "duration_s": round(report.duration_s, 3),
            },
        )
        return report

    # -- Pyright ---------------------------------------------------------- #

    async def _check_pyright(
        self, paths: list[str], *, root: str | Path, timeout: float
    ) -> LSPReport:  # pragma: no cover - 需要 pyright
        cmd = [self.pyright_cmd or "pyright", "--outputjson", "--level", "error", *paths]
        result = await run_command(cmd, cwd=root, timeout=timeout, max_output_bytes=2_000_000)
        if result.timed_out:
            return LSPReport(passed=False, engine="pyright", diagnostics=[
                Diagnostic("", 0, 0, "error", "pyright 执行超时")
            ])
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            logger.warning("pyright output not JSON, fallback to syntax check")
            return await self._check_syntax(paths, root=root)

        diagnostics: list[Diagnostic] = []
        summary = payload.get("summary", {})
        for diag in payload.get("generalDiagnostics", []):
            rng = diag.get("range", {}).get("start", {})
            diagnostics.append(
                Diagnostic(
                    file=diag.get("file", ""),
                    line=int(rng.get("line", 0)) + 1,
                    column=int(rng.get("character", 0)) + 1,
                    severity=_SEVERITY.get(diag.get("severity", "error"), "error"),
                    message=diag.get("message", ""),
                    rule=diag.get("rule", "") or "",
                )
            )
        error_count = summary.get("errorCount", len([d for d in diagnostics if d.is_error]))
        return LSPReport(
            passed=error_count == 0, diagnostics=diagnostics, engine="pyright"
        )

    # -- 语法兜底引擎 ------------------------------------------------------ #

    async def _check_syntax(self, paths: list[str], *, root: str | Path) -> LSPReport:
        diagnostics: list[Diagnostic] = []
        for path in paths:
            p = Path(path)
            if not p.is_absolute():
                p = Path(root) / p
            try:
                source = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                diagnostics.append(
                    Diagnostic(str(path), 0, 0, "error", f"无法读取文件: {exc}", "io")
                )
                continue
            try:
                ast.parse(source, filename=str(path))
            except SyntaxError as exc:
                diagnostics.append(
                    Diagnostic(
                        file=str(path),
                        line=exc.lineno or 0,
                        column=exc.offset or 0,
                        severity="error",
                        message=exc.msg,
                        rule="syntax",
                    )
                )
        return LSPReport(passed=not diagnostics, diagnostics=diagnostics, engine="syntax")

    @property
    def last_duration(self) -> float:
        return self._last_duration


async def run_static_checks(
    files: Sequence[str | Path],
    *,
    root: str | Path = ".",
    bridge: LSPBridge | None = None,
) -> LSPReport:
    """便捷函数：构造默认桥接器并检查文件。"""
    bridge = bridge or LSPBridge()
    return await bridge.check(files, root=root)
