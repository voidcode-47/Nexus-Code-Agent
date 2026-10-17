"""多进程/高并发评测执行器。

* :meth:`EvaluationRunner.run` —— 基于 ``asyncio.Semaphore`` 的高并发评测；
* :meth:`EvaluationRunner.run_multiprocess` —— 基于 ``ProcessPoolExecutor`` 的多进程
  隔离评测（避免 LLM/沙箱状态互相污染），适配大规模跑分。

**Pass@k**：``attempts=k`` 时每道题至多尝试 k 轮（成功即早停），
每轮结果记录在 :attr:`EvalResult.attempt_outcomes`，供报表计算 Pass@k 曲线。

每道题产出 :class:`EvalResult`，包含是否修复、迭代次数、Token 消耗、耗时与
LSP 拦截情况，供 :mod:`src.evaluation.metrics_report` 汇总。
"""

from __future__ import annotations

import asyncio
import pickle
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.agent.workflow import NexusWorkflow, WorkflowDeps
from src.config import get_logger
from src.evaluation.swe_bench_loader import SweBenchInstance

logger = get_logger(__name__)


@dataclass
class EvalResult:
    """单题评测结果。"""

    instance_id: str
    resolved: bool
    status: str = "PENDING"
    iterations: int = 0
    tokens: int = 0
    duration_s: float = 0.0
    lsp_blocked: bool = False
    predicted_patch: str = ""
    termination_reason: str = ""
    error: str = ""
    #: 每轮尝试是否成功（Pass@k 计算依据）；单轮时长度为 1
    attempt_outcomes: list[bool] = field(default_factory=list)
    #: 实际执行的尝试轮数（成功早停时可能小于配置的 k）
    attempts_made: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["predicted_patch"] = ""  # 序列化时剔除大字段（单独持久化）
        return data


RepoResolver = Callable[[SweBenchInstance], str | None]


class EvaluationRunner:
    """评测执行器。"""

    def __init__(
        self,
        *,
        repo_resolver: RepoResolver | None = None,
        concurrency: int = 2,
        timeout_s: float = 900.0,
        attempts: int = 1,
        workflow_factory: Callable[[str], NexusWorkflow] | None = None,
    ) -> None:
        self.repo_resolver = repo_resolver
        self.concurrency = max(1, concurrency)
        self.timeout_s = timeout_s
        self.attempts = max(1, attempts)
        self.workflow_factory = workflow_factory or _default_workflow

    # -- 单题 ------------------------------------------------------------- #

    def resolve_repo(self, instance: SweBenchInstance) -> str | None:
        if instance.local_repo and Path(instance.local_repo).exists():
            return instance.local_repo
        if self.repo_resolver is not None:
            return self.repo_resolver(instance)
        return None

    async def solve(self, instance: SweBenchInstance) -> EvalResult:
        """求解一道题：至多 ``attempts`` 轮尝试，任意一轮成功即早停。

        每轮结果记录在 ``attempt_outcomes``，供 Pass@k 曲线计算。
        """
        outcomes: list[bool] = []
        result: EvalResult | None = None
        total_iterations = 0
        total_tokens = 0
        total_duration = 0.0

        for attempt in range(self.attempts):
            result = await self._solve_once(instance, attempt)
            outcomes.append(result.resolved)
            total_iterations += result.iterations
            total_tokens += result.tokens
            total_duration += result.duration_s
            if result.resolved:
                break

        assert result is not None  # attempts >= 1 保证至少执行一轮
        result.iterations = total_iterations
        result.tokens = total_tokens
        result.duration_s = total_duration
        result.attempt_outcomes = outcomes
        result.attempts_made = len(outcomes)
        result.resolved = any(outcomes)
        return result

    async def _solve_once(self, instance: SweBenchInstance, attempt: int = 0) -> EvalResult:
        repo_path = self.resolve_repo(instance)
        if not repo_path:
            return EvalResult(
                instance_id=instance.instance_id,
                resolved=False,
                status="SKIPPED",
                error="未找到本地仓库路径（需配置 local_repo 或 repo_resolver）",
            )

        task_id = (
            instance.instance_id if attempt == 0 else f"{instance.instance_id}-a{attempt + 1}"
        )
        start = time.monotonic()
        try:
            workflow = self.workflow_factory(repo_path)
            state = await asyncio.wait_for(
                workflow.run(instance.problem_statement, repo_path, task_id=task_id),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            return EvalResult(
                instance_id=instance.instance_id,
                resolved=False,
                status="TIMEOUT",
                duration_s=time.monotonic() - start,
                error=f"超出单题时限 {self.timeout_s}s",
            )
        except Exception as exc:  # pragma: no cover - 运行时异常
            logger.exception("题目求解异常", extra={"instance": instance.instance_id})
            return EvalResult(
                instance_id=instance.instance_id,
                resolved=False,
                status="ERROR",
                duration_s=time.monotonic() - start,
                error=str(exc),
            )

        events = state.get("events", [])
        lsp_blocked = any(
            e.get("node") == "tester" and "拦截" in str(e.get("message", "")) for e in events
        )
        resolved = bool(state.get("test_passed", False)) and state.get("status") == "SUCCESS"
        return EvalResult(
            instance_id=instance.instance_id,
            resolved=resolved,
            status=str(state.get("status", "UNKNOWN")),
            iterations=int(state.get("current_iteration", 0)),
            tokens=int(state.get("token_usage", 0)),
            duration_s=time.monotonic() - start,
            lsp_blocked=lsp_blocked,
            predicted_patch=state.get("final_diff", ""),
            termination_reason=state.get("termination_reason", ""),
        )

    # -- 批量（异步并发） -------------------------------------------------- #

    async def run(self, instances: Iterable[SweBenchInstance]) -> list[EvalResult]:
        semaphore = asyncio.Semaphore(self.concurrency)

        async def _guarded(inst: SweBenchInstance) -> EvalResult:
            async with semaphore:
                return await self.solve(inst)

        results = await asyncio.gather(*(_guarded(i) for i in instances))
        logger.info(
            "评测完成",
            extra={"n": len(results), "resolved": sum(r.resolved for r in results)},
        )
        return list(results)

    def run_sync(self, instances: Iterable[SweBenchInstance]) -> list[EvalResult]:
        return asyncio.run(self.run(instances))

    # -- 批量（多进程隔离） ------------------------------------------------ #

    def run_multiprocess(
        self, instances: Iterable[SweBenchInstance], *, workers: int = 2
    ) -> list[EvalResult]:
        items = list(instances)
        factory_blob: bytes | None = None
        if self.workflow_factory is not _default_workflow:
            # 子进程无法继承闭包：尝试 pickle 自定义工厂；失败则显式报错，
            # 而不是静默退化到默认工作流（那会造成单/多进程结果不一致）。
            try:
                factory_blob = pickle.dumps(self.workflow_factory)
            except Exception as exc:
                raise ValueError(
                    "自定义 workflow_factory 无法序列化到子进程；"
                    "请改用模块级函数工厂，或设置 workers=1 走异步并发"
                ) from exc
        payloads = [
            {
                "instance": i.to_dict(),
                "timeout_s": self.timeout_s,
                "attempts": self.attempts,
                "workflow_factory": factory_blob,
            }
            for i in items
            if self.resolve_repo(i)
        ]
        skipped = [
            EvalResult(
                instance_id=i.instance_id,
                resolved=False,
                status="SKIPPED",
                error="未找到本地仓库路径",
            )
            for i in items
            if not self.resolve_repo(i)
        ]
        if not payloads:
            return skipped

        with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
            raw = list(executor.map(_solve_in_process, payloads))
        results = [EvalResult(**r) for r in raw] + skipped
        return results


def _default_workflow(repo_path: str) -> NexusWorkflow:
    """默认工作流工厂：每道题独立构建索引与依赖。"""
    deps = WorkflowDeps.from_repo(repo_path)
    # 每题依赖独占，运行结束即释放容器池 / LLM 连接池
    return NexusWorkflow(deps, close_pool=True, close_llm=True)


def _solve_in_process(payload: dict[str, Any]) -> dict[str, Any]:
    """多进程 worker 入口（需为模块级函数以保证可 pickle）。"""
    instance = SweBenchInstance.from_dict(payload["instance"])
    blob = payload.get("workflow_factory")
    factory = pickle.loads(blob) if blob else None
    runner = EvaluationRunner(
        timeout_s=float(payload.get("timeout_s", 900)),
        attempts=int(payload.get("attempts", 1)),
        workflow_factory=factory,
    )
    result = asyncio.run(runner.solve(instance))
    return result.to_dict()


def workflow_factory_from_deps(deps: WorkflowDeps) -> Callable[[str], NexusWorkflow]:
    """复用同一套依赖（检索/沙箱）的工厂，适合单仓库多题评测。"""

    def _factory(_: str) -> NexusWorkflow:
        return NexusWorkflow(deps)

    return _factory
