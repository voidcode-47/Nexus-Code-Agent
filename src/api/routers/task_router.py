"""异步任务提交、SSE 实时思考流推送与 Time-Travel 人工介入接口。

* **任务存储**：进程内 :class:`TaskStore` 保存任务快照、事件历史与订阅者队列；
* **SSE 思考流**：``GET /tasks/{id}/events`` 以 Server-Sent Events 实时推送
  Agent 当前节点、检索命中、Diff 预览与沙箱终端输出；
* **Time-Travel**：``POST /tasks/{id}/intervention`` 注入人工反馈并可续跑；
  ``GET /tasks/{id}/checkpoints`` 返回 Checkpoint 历史（状态回溯），
  Checkpointer 由应用生命周期注入（Postgres 优先，内存兜底）。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.agent.workflow import NexusWorkflow, WorkflowDeps, build_checkpointer
from src.api.schemas import (
    InterventionRequest,
    InterventionResponse,
    PatchRecordSchema,
    TaskStatusResponse,
    TaskSubmitRequest,
    TaskSubmitResponse,
)
from src.config import get_logger
from src.config.settings import settings

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])

#: 进程级 Checkpointer（由 ``main.lifespan`` 注入；Postgres 不可用时为内存实现）
_checkpointer: Any | None = None

#: 进程级共享容器池（由 ``main.lifespan`` 注入/预热/回收；未注入时任务自建自毁）
_shared_pool: Any | None = None


def set_checkpointer(checkpointer: Any | None) -> None:
    """注入进程级 Checkpointer（应用启动时调用）。"""
    global _checkpointer
    _checkpointer = checkpointer


def set_shared_pool(pool: Any | None) -> None:
    """注入进程级共享容器池（应用启动时调用；None 恢复任务私有池）。"""
    global _shared_pool
    _shared_pool = pool


def _resolve_checkpointer() -> Any | None:
    return _checkpointer if _checkpointer is not None else build_checkpointer()

TERMINAL_STATUSES = {"SUCCESS", "HUMAN_ESCALATION", "ERROR"}

#: 「状态已终态但 done 事件尚未发布」时的收流宽限（秒）。
#: 正常路径下 done 在几百毫秒内（Worktree 清理）就会到达；该宽限只用于
#: 兜底异常路径，避免客户端被永久挂住。
_TERMINAL_GRACE_S = 5.0

EventSourceResponse: Any = None
try:  # pragma: no cover - 可选依赖
    from sse_starlette.sse import EventSourceResponse

    _HAS_SSE_STARLETTE = True
except Exception:  # pragma: no cover
    _HAS_SSE_STARLETTE = False


@dataclass
class TaskRecord:
    """任务运行记录。"""

    task_id: str
    repo_path: str
    issue_description: str
    max_iterations: int | None = None
    status: str = "PENDING"
    attempt: int = 0
    state: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    handle: asyncio.Task | None = field(default=None, repr=False)
    workflow: Any = field(default=None, repr=False)
    created_at: float = field(default_factory=time.time)
    #: done 事件是否已发布。与 ``publish(done)`` 之间无 await，二者原子。
    finished: bool = False

    @property
    def thread_id(self) -> str:
        """Checkpoint 线程 ID：每次干预续跑使用独立线程，避免历史交叉污染。"""
        return self.task_id if self.attempt == 0 else f"{self.task_id}-r{self.attempt}"

    def publish(self, payload: dict[str, Any]) -> None:
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:  # pragma: no cover
                continue


class TaskStore:
    """进程内任务存储（生产可替换为 Redis / Postgres）。

    带容量上限：超出时按创建时间优先淘汰**已终结**任务，防止长跑服务
    任务字典与事件列表无限增长。
    """

    def __init__(self, *, max_tasks: int = 200, max_events: int = 500) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = asyncio.Lock()
        self.max_tasks = max_tasks
        self.max_events = max_events

    @property
    def size(self) -> int:
        return len(self._tasks)

    def _evict_locked(self) -> None:
        """持锁调用：为新任务腾出容量，按创建时间优先淘汰已终结任务。

        在插入**前**调用并淘汰到 ``len < max_tasks``，稳态容量恰好为
        ``max_tasks``（旧实现允许多出 1 个）。全部任务都在运行中时无法
        淘汰（软上限，允许暂时超出）。
        """
        for record in sorted(self._tasks.values(), key=lambda r: r.created_at):
            if len(self._tasks) < self.max_tasks:
                break
            if record.status in TERMINAL_STATUSES:
                self._tasks.pop(record.task_id, None)

    async def create(
        self,
        repo_path: str,
        issue_description: str,
        *,
        task_id: str | None = None,
        max_iterations: int | None = None,
    ) -> TaskRecord:
        async with self._lock:
            if task_id is not None and task_id in self._tasks:
                raise ValueError(f"task_id 已存在：{task_id}（请换用新 ID，或省略以自动生成）")
            self._evict_locked()
            tid = task_id or f"task-{uuid.uuid4().hex[:12]}"
            record = TaskRecord(
                task_id=tid,
                repo_path=repo_path,
                issue_description=issue_description,
                max_iterations=max_iterations,
            )
            self._tasks[tid] = record
            return record

    def get(self, task_id: str) -> TaskRecord | None:
        return self._tasks.get(task_id)

    def list(self) -> list[TaskRecord]:
        return list(self._tasks.values())

    def subscribe(self, record: TaskRecord) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        record.subscribers.append(queue)
        return queue

    def unsubscribe(self, record: TaskRecord, queue: asyncio.Queue) -> None:
        if queue in record.subscribers:
            record.subscribers.remove(queue)


store = TaskStore()


async def _run_task_workflow(record: TaskRecord) -> None:
    """构建依赖并消费工作流事件流（整体受 ``_execute`` 的任务级超时约束）。"""
    pool_kwargs: dict[str, Any] = {"pool": _shared_pool} if _shared_pool is not None else {}
    # 索引构建与容器池探测都是同步阻塞操作：直接在事件循环里执行会让
    # 同进程的其他任务 / SSE 心跳整段停摆（高并发时表现为「并发度上不去」）。
    deps = await asyncio.to_thread(WorkflowDeps.from_repo, record.repo_path, **pool_kwargs)
    # 容器池可能共享（由应用生命周期关闭）；LLM 客户端始终是每任务新建的，
    # 必须随任务释放，否则真实 api_key 路径下每任务泄漏一个连接池。
    workflow = NexusWorkflow(
        deps,
        checkpointer=_resolve_checkpointer(),
        close_pool=_shared_pool is None,
        close_llm=True,
    )
    record.workflow = workflow
    async for chunk in workflow.astream(
        record.issue_description,
        record.repo_path,
        task_id=record.thread_id,
        max_iterations=record.max_iterations,
    ):
        for node, update in chunk.items():
            if not isinstance(update, dict):
                continue
            record.state = _merge(record.state, update)
            for ev in update.get("events", []) or []:
                enriched = {"type": "event", **ev}
                record.events.append(enriched)
                if len(record.events) > store.max_events:
                    del record.events[: len(record.events) - store.max_events]
                record.publish(enriched)
            if update.get("status"):
                record.status = str(update["status"])
                record.publish(
                    {"type": "status", "node": node, "status": record.status,
                     "message": f"{node} → {record.status}"}
                )


async def _execute(record: TaskRecord) -> None:
    """后台执行工作流并把事件推送给订阅者（带任务级超时）。

    ``NEXUS_API__REQUEST_TIMEOUT_S`` 是整条自愈任务的上限：无论卡在沙箱、
    LLM 还是图执行的哪一环，都不允许永久挂起。超时任务终止为 ``ERROR``
    并写入可检索的 ``termination_reason``（0 = 显式关闭超时，仅限本地调试）。
    """
    record.finished = False  # 续跑复用同一 record：新一轮必须重新等待 done
    record.status = "RUNNING"
    record.publish({"type": "status", "node": "system", "status": record.status,
                    "message": "任务开始执行"})
    timeout = settings.api.request_timeout_s
    try:
        await asyncio.wait_for(
            _run_task_workflow(record), timeout=timeout if timeout > 0 else None
        )
        record.status = str(record.state.get("status", "UNKNOWN"))
    except asyncio.TimeoutError:
        # 各 HTTP / 子进程客户端内部均已自行兜底超时；能冒泡到这一层的
        # TimeoutError 几乎必然来自 wait_for 的任务级超时。捕获
        # ``asyncio.TimeoutError`` 而非内置类型：Python 3.10 下两者不是同一
        # 类型，写内置类型会漏捕获，任务级超时退化成普通异常路径。
        logger.error(
            "任务执行超时", extra={"task_id": record.task_id, "timeout_s": timeout}
        )
        record.status = "ERROR"
        record.state["termination_reason"] = (
            f"任务执行超时：超过 NEXUS_API__REQUEST_TIMEOUT_S={timeout}s 未完成"
        )
        record.publish(
            {"type": "error", "node": "system", "message": "任务执行超时，已中止"}
        )
    except Exception as exc:  # pragma: no cover - 运行时异常
        logger.exception("任务执行失败", extra={"task_id": record.task_id})
        record.status = "ERROR"
        record.state["termination_reason"] = str(exc)
        record.publish({"type": "error", "node": "system", "message": str(exc)})
    finally:
        # finished 与 done 的发布之间不能有 await：SSE 侧据此判断「队列空 +
        # finished」即可安全收流，不会漏掉 done。
        record.finished = True
        record.publish(
            {
                "type": "done",
                "node": "system",
                "status": record.status,
                "message": record.state.get("termination_reason", "任务结束"),
                "final_diff": record.state.get("final_diff", ""),
            }
        )


def _merge(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(state)
    for key, value in update.items():
        if key in {"patch_history", "events"} and isinstance(value, list):
            merged[key] = [*merged.get(key, []), *value]
        else:
            merged[key] = value
    return merged


# --------------------------------------------------------------------------- #
# 路由
# --------------------------------------------------------------------------- #


def _validate_repo_path(repo_path: str) -> None:
    """校验 repo_path：必须存在；配置白名单时必须落在白名单目录内。"""
    path = Path(repo_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"repo_path 不存在或不是目录：{repo_path}")
    roots = settings.api.allowed_repo_root_list
    if roots:
        allowed = any(path.is_relative_to(Path(r).expanduser().resolve()) for r in roots)
        if not allowed:
            raise HTTPException(
                status_code=403,
                detail="repo_path 不在 NEXUS_API__ALLOWED_REPO_ROOTS 白名单内",
            )


@router.post("", response_model=TaskSubmitResponse, summary="提交自愈任务")
async def submit_task(payload: TaskSubmitRequest) -> TaskSubmitResponse:
    """提交任务并异步执行（返回后可通过 SSE 订阅实时思考流）。"""
    _validate_repo_path(payload.repo_path)
    try:
        record = await store.create(
            payload.repo_path,
            payload.issue_description,
            task_id=payload.task_id,
            max_iterations=payload.max_iterations,
        )
    except ValueError as exc:
        # 重复 task_id 会覆盖仍在运行的原记录（事件/终态丢失、SSE 订阅错位）
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    record.handle = asyncio.create_task(_execute(record))
    return TaskSubmitResponse(task_id=record.task_id, status=record.status)


@router.get("", summary="列出全部任务")
async def list_tasks() -> list[dict[str, Any]]:
    return [
        {
            "task_id": r.task_id,
            "status": r.status,
            "repo_path": r.repo_path,
            "iteration": r.state.get("current_iteration", 0),
            "created_at": r.created_at,
        }
        for r in store.list()
    ]


@router.get("/{task_id}", response_model=TaskStatusResponse, summary="查询任务状态")
async def get_task(task_id: str) -> TaskStatusResponse:
    record = _require(task_id)
    state = record.state
    return TaskStatusResponse(
        task_id=record.task_id,
        status=record.status,
        repo_path=record.repo_path,
        issue_description=record.issue_description,
        current_iteration=int(state.get("current_iteration", 0)),
        max_iterations=int(state.get("max_iterations", 0)),
        suspected_files=list(state.get("suspected_files", [])),
        repair_plan=str(state.get("repair_plan", "")),
        worktree_path=str(state.get("worktree_path", "")),
        token_usage=int(state.get("token_usage", 0)),
        termination_reason=str(state.get("termination_reason", "")),
        final_diff=str(state.get("final_diff", "")),
        patch_history=[
            PatchRecordSchema(
                iteration=rec.get("iteration", 0),
                lsp_passed=rec.get("lsp_passed", False),
                test_passed=rec.get("test_passed", False),
                error_fingerprint=rec.get("error_fingerprint", ""),
                error_summary=rec.get("error_summary", ""),
                strategy=rec.get("strategy", ""),
            )
            for rec in state.get("patch_history", [])
        ],
        events=record.events[-100:],
    )


@router.get("/{task_id}/events", summary="SSE 实时思考流")
async def stream_events(task_id: str) -> Any:
    """以 Server-Sent Events 推送 Agent 思考流。"""
    record = _require(task_id)
    queue = store.subscribe(record)

    async def _gen() -> AsyncIterator[dict[str, str]]:
        # 先回放历史事件，保证晚订阅者也能看到完整过程
        for ev in record.events[-50:]:
            yield {"event": "message", "data": json.dumps(ev, ensure_ascii=False, default=str)}
        # 收流判据必须基于 finished（done 是否已发布），而不是终态 status：
        # critic 节点返回时 record.status 就已变成 SUCCESS，而 done 要等
        # Worktree 清理（git worktree remove/prune，数百毫秒）之后才发布；
        # 只看 status 会在该窗口内提前收流，客户端永远拿不到携带 final_diff
        # 的 done 事件（实测可稳定复现）。
        grace_deadline: float | None = None
        try:
            while True:
                if record.finished and queue.empty():
                    break
                if not record.finished and record.status in TERMINAL_STATUSES:
                    # 兜底：终态已置但 done 迟迟未发布（异常路径），限时收尾
                    if grace_deadline is None:
                        grace_deadline = time.monotonic() + _TERMINAL_GRACE_S
                    elif time.monotonic() >= grace_deadline:
                        break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # 保持长连接：发心跳，收流条件由循环顶部统一判定
                    yield {"event": "ping", "data": "{}"}
                    continue
                yield {
                    "event": "message",
                    "data": json.dumps(payload, ensure_ascii=False, default=str),
                }
                if payload.get("type") == "done":
                    break
        finally:
            store.unsubscribe(record, queue)

    if _HAS_SSE_STARLETTE:
        return EventSourceResponse(_gen())
    return StreamingResponse(_sse_bytes(_gen()), media_type="text/event-stream")


@router.get("/{task_id}/checkpoints", summary="Time-Travel 状态回溯")
async def list_checkpoints(task_id: str) -> list[dict[str, Any]]:
    """返回任务当前线程的 Checkpoint 历史（最新在前），支持状态回溯。"""
    record = _require(task_id)
    if record.workflow is None:
        return []
    return await record.workflow.state_history(record.thread_id)


@router.post(
    "/{task_id}/intervention",
    response_model=InterventionResponse,
    summary="Time-Travel 人工介入",
)
async def intervene(task_id: str, payload: InterventionRequest) -> InterventionResponse:
    """注入人工反馈；当任务处于 HUMAN_ESCALATION 时可用反馈续跑。"""
    record = _require(task_id)
    feedback = payload.human_feedback.strip()
    if feedback:
        # 反馈经 issue_description 注入工作流（这是唯一生效通道）
        record.issue_description = f"{record.issue_description}\n\n[人工介入] {feedback}"

    if payload.resume and record.status in {"HUMAN_ESCALATION", "ERROR", "SUCCESS"}:
        record.attempt += 1  # 新线程：保留上一轮 Checkpoint 供回溯
        record.status = "PENDING"
        record.events = []
        record.state = {}
        record.handle = asyncio.create_task(_execute(record))
        return InterventionResponse(
            task_id=task_id,
            status=record.status,
            message=f"已携带人工反馈重新执行（第 {record.attempt + 1} 轮，线程 {record.thread_id}）",
        )
    return InterventionResponse(
        task_id=task_id, status=record.status, message="已记录人工反馈（未触发续跑）"
    )


def _require(task_id: str) -> TaskRecord:
    record = store.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
    return record


async def _sse_bytes(gen: AsyncIterator[dict[str, str]]) -> AsyncIterator[bytes]:
    async for item in gen:
        event = item.get("event", "message")
        data = item.get("data", "")
        yield f"event: {event}\ndata: {data}\n\n".encode()
