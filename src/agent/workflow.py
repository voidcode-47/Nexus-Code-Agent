"""带 PostgresCheckpointer 持久化的主图编排。

图结构::

    localizer → architect → coder → tester → critic ─┬─ SUCCESS / HUMAN_ESCALATION → END
                                                     ├─ CODING  → coder   （修正补丁）
                                                     └─ PLANNING → architect（熔断重规划）

* **Checkpointer 持久化**：配置 ``postgres.enabled=true`` 时接入 ``AsyncPostgresSaver``，
  支持断点续跑与状态回溯（Time-Travel）；不可用时降级为内存 Checkpointer / 无 Checkpointer。
* **递归上限**：由 ``agent.graph_recursion_limit`` 控制，配合 Critic 熔断双保险。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from src.agent.llm import LLMClient, get_llm
from src.agent.nodes import (
    architect_node,
    coder_node,
    critic_node,
    localizer_node,
    tester_node,
)
from src.agent.state import AgentState, event, initial_state
from src.config import get_logger, set_task_id
from src.config.settings import settings as global_settings
from src.indexer.call_graph import CallGraph
from src.indexer.hybrid_retriever import HybridRetriever
from src.indexer.symbol_table import build_symbol_table
from src.mcp_tools.lsp_bridge import LSPBridge
from src.sandbox.git_snapshot import WorktreeSnapshot, ensure_git_repo
from src.sandbox.pool_manager import PoolManager

logger = get_logger(__name__)

END: Any = "__end__"
StateGraph: Any = None
try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - 依赖缺失
    LANGGRAPH_AVAILABLE = False

ADDITIVE_FIELDS = {"patch_history", "events"}


@dataclass
class WorkflowDeps:
    """工作流运行时依赖。"""

    retriever: HybridRetriever | None = None
    graph: CallGraph | None = None
    llm: LLMClient | None = None
    pool: PoolManager = field(default_factory=PoolManager)
    lsp: LSPBridge = field(default_factory=LSPBridge)

    @classmethod
    def from_repo(cls, repo_root: str | Path, **kwargs: Any) -> WorkflowDeps:
        """构建索引（符号表 / 调用图 / 混合检索器）并组装依赖。"""
        table = build_symbol_table(repo_root)
        graph = CallGraph.build(table)
        retriever = HybridRetriever(
            table=table, call_graph=graph, repo_root=str(Path(repo_root).resolve())
        )
        retriever.index_repo(table, graph)
        return cls(retriever=retriever, graph=graph, llm=get_llm(), **kwargs)

    async def aclose(self, *, pool: bool = True, llm: bool = True) -> None:
        """释放依赖资源（幂等）。

        ``pool`` / ``llm`` 按所有权分别开关：容器池可能是进程级共享的
        （API lifespan 注入），而 LLM 客户端始终是每任务新建的，两者的
        生命周期不能绑成一个开关。
        """
        if llm:
            close_llm = getattr(self.llm, "aclose", None)
            if close_llm is not None:
                try:
                    await close_llm()
                except Exception as exc:  # pragma: no cover - 关闭失败不阻断
                    logger.warning("LLM 客户端关闭失败", extra={"error": str(exc)})
        if pool:
            try:
                await self.pool.aclose()
            except Exception as exc:  # pragma: no cover - 关闭失败不阻断
                logger.warning("容器池关闭失败", extra={"error": str(exc)})


class NexusWorkflow:
    """自愈工作流编排器。"""

    def __init__(
        self,
        deps: WorkflowDeps | None = None,
        *,
        checkpointer: Any | None = None,
        use_langgraph: bool = True,
        worktree_root: str | Path | None = None,
        close_pool: bool = False,
        close_llm: bool = False,
    ) -> None:
        self.deps = deps or WorkflowDeps()
        self.checkpointer = checkpointer
        self.worktree_root = worktree_root
        # 资源所有权显式拆分：容器池可能是进程级共享的（API lifespan 注入，
        # 由应用统一关闭），LLM 客户端则始终是每任务新建，必须随任务释放。
        self.close_pool = close_pool
        self.close_llm = close_llm
        self._graph: Any | None = None
        if use_langgraph and LANGGRAPH_AVAILABLE:
            self._graph = self._build_graph(self.deps)

    # -- 图构建 ----------------------------------------------------------- #

    async def _build_deps(self, repo: Path) -> WorkflowDeps:
        """在线程池中构建索引依赖。

        ``WorkflowDeps.from_repo`` 会全仓扫描 + AST 解析 + 建图 + 向量化，是
        同步 CPU/IO 密集操作；直接在事件循环里执行会让同进程的其他任务、
        SSE 心跳、健康检查整段停摆（实测 400 文件仓库约 1s，期间事件循环
        调度次数为 0）。容器池探测（``docker.from_env().ping()``）同属阻塞
        网络调用，一并放线程。
        """
        return await asyncio.to_thread(
            WorkflowDeps.from_repo, repo, pool=self.deps.pool, lsp=self.deps.lsp
        )

    def _build_graph(self, deps: WorkflowDeps) -> Any:
        assert StateGraph is not None

        async def _localizer(state: AgentState) -> dict[str, Any]:
            return await localizer_node(
                state,
                retriever=deps.retriever,  # type: ignore[arg-type]
                graph=deps.graph,
                llm=deps.llm,
                n_hop=global_settings.agent.n_hop,
            )

        async def _architect(state: AgentState) -> dict[str, Any]:
            return await architect_node(state, llm=deps.llm)

        async def _coder(state: AgentState) -> dict[str, Any]:
            return await coder_node(state, llm=deps.llm, graph=deps.graph)

        async def _tester(state: AgentState) -> dict[str, Any]:
            return await tester_node(state, pool=deps.pool, lsp=deps.lsp)

        async def _critic(state: AgentState) -> dict[str, Any]:
            return await critic_node(state, llm=deps.llm)

        builder = StateGraph(AgentState)
        builder.add_node("localizer", _localizer)
        builder.add_node("architect", _architect)
        builder.add_node("coder", _coder)
        builder.add_node("tester", _tester)
        builder.add_node("critic", _critic)

        builder.set_entry_point("localizer")
        builder.add_edge("localizer", "architect")
        builder.add_edge("architect", "coder")
        builder.add_edge("coder", "tester")
        builder.add_edge("tester", "critic")

        def _route(state: AgentState) -> str:
            status = state.get("status", "CODING")
            if status in {"SUCCESS", "HUMAN_ESCALATION"}:
                return END
            if status == "PLANNING":
                return "architect"
            return "coder"

        builder.add_conditional_edges(
            "critic",
            _route,
            {"coder": "coder", "architect": "architect", END: END},
        )

        compile_kwargs: dict[str, Any] = {}
        if self.checkpointer is not None:
            compile_kwargs["checkpointer"] = self.checkpointer
        compiled = builder.compile(**compile_kwargs)
        logger.info("LangGraph 工作流已编译")
        return compiled

    # -- 执行 ------------------------------------------------------------- #

    def _config(self, task_id: str) -> dict[str, Any]:
        config: dict[str, Any] = {
            "recursion_limit": global_settings.agent.graph_recursion_limit
        }
        if self.checkpointer is not None:
            config["configurable"] = {"thread_id": task_id}
        return config

    def _sync_container_mount(self) -> None:
        """docker 后端下把容器挂载根对齐到实际 worktree 根目录。

        ``NexusWorkflow(worktree_root=...)`` 允许覆盖默认 worktree 根目录；
        若 pool 的容器挂载根仍指向 settings 默认值，会因容器内路径不一致
        导致测试跑在错误目录（_relative_cwd 退化 /workspace）。

        注意：已预热的容器 bind 的是**创建时**的旧路径，无法事后迁移；
        此时直接报错，而不是留下一个只在日志里的 warning 后跑错目录。
        """
        pool = self.deps.pool
        if pool.backend != "docker" or not self.worktree_root:
            return
        root = Path(self.worktree_root)
        target = (root if root.is_absolute() else (Path.cwd() / root)).resolve()
        if Path(pool.container_mount_root).resolve() == target:
            return
        if pool.has_active_containers():
            raise ValueError(
                "容器池已预热，无法更改容器挂载根（已创建容器 bind 的是旧路径）；"
                "请让 NexusWorkflow(worktree_root=...) 与容器池创建时一致，"
                "或在预热前完成设置"
            )
        pool.container_mount_root = target

    async def run(
        self,
        issue_description: str,
        repo_path: str | Path,
        *,
        task_id: str | None = None,
        max_iterations: int | None = None,
    ) -> AgentState:
        """执行完整自愈闭环，返回终态。"""
        repo = Path(repo_path).resolve()
        if self.deps.retriever is None:
            self.deps = await self._build_deps(repo)
            if LANGGRAPH_AVAILABLE and self._graph is not None:
                self._graph = self._build_graph(self.deps)

        state = initial_state(
            issue_description, str(repo), task_id=task_id, max_iterations=max_iterations
        )
        set_task_id(state["task_id"])

        await ensure_git_repo(repo)
        self._sync_container_mount()
        snapshot = WorktreeSnapshot(repo, worktree_root=self.worktree_root, name=f"nexus-{state['task_id'][-8:]}")
        await snapshot.create()
        state["worktree_path"] = str(snapshot.path)
        state["worktree_id"] = snapshot.path.name
        state["events"] = [event("provision", f"Worktree 已创建：{snapshot.path.name}")]
        try:
            if self._graph is not None:
                return await self._graph.ainvoke(state, config=self._config(state["task_id"]))
            return await self._run_fallback(state)
        finally:
            await snapshot.cleanup()
            if self.close_pool or self.close_llm:
                await self.deps.aclose(pool=self.close_pool, llm=self.close_llm)

    async def astream(
        self,
        issue_description: str,
        repo_path: str | Path,
        *,
        task_id: str | None = None,
        max_iterations: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """流式执行，逐节点产出更新（供 SSE 使用）。"""
        repo = Path(repo_path).resolve()
        if self.deps.retriever is None:
            self.deps = await self._build_deps(repo)
            if LANGGRAPH_AVAILABLE and self._graph is not None:
                self._graph = self._build_graph(self.deps)

        state = initial_state(
            issue_description, str(repo), task_id=task_id, max_iterations=max_iterations
        )
        set_task_id(state["task_id"])
        await ensure_git_repo(repo)
        self._sync_container_mount()
        snapshot = WorktreeSnapshot(repo, worktree_root=self.worktree_root, name=f"nexus-{state['task_id'][-8:]}")
        await snapshot.create()
        state["worktree_path"] = str(snapshot.path)
        state["worktree_id"] = snapshot.path.name
        yield {"provision": {"worktree_path": str(snapshot.path), "worktree_id": snapshot.path.name}}
        try:
            if self._graph is not None:
                async for chunk in self._graph.astream(
                    state, config=self._config(state["task_id"]), stream_mode="updates"
                ):
                    yield chunk
            else:
                async for chunk in self._run_fallback_stream(state):
                    yield chunk
        finally:
            await snapshot.cleanup()
            if self.close_pool or self.close_llm:
                await self.deps.aclose(pool=self.close_pool, llm=self.close_llm)

    # -- Time-Travel 状态回溯 ---------------------------------------------- #

    async def state_history(self, thread_id: str) -> list[dict[str, Any]]:
        """返回指定线程的 Checkpoint 历史（Time-Travel 状态回溯）。

        仅在启用 Checkpointer 且有 LangGraph 时可用；按时间倒序
        （最新在前）返回每个快照的路由状态与迭代号。
        """
        if self._graph is None or self.checkpointer is None:
            return []
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        snapshots: list[dict[str, Any]] = []
        try:
            async for snapshot in self._graph.aget_state_history(config):
                values = getattr(snapshot, "values", None) or {}
                checkpoint_id = ""
                if isinstance(snapshot.config, Mapping):
                    checkpoint_id = str(
                        snapshot.config.get("configurable", {}).get("checkpoint_id", "")
                    )
                snapshots.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "thread_id": thread_id,
                        "next": list(getattr(snapshot, "next", ()) or ()),
                        "status": values.get("status", ""),
                        "current_iteration": values.get("current_iteration", 0),
                    }
                )
        except Exception as exc:  # pragma: no cover - 依赖 Checkpointer 实现
            logger.warning("读取 Checkpoint 历史失败", extra={"error": str(exc)})
        return snapshots

    # -- 无 LangGraph 兜底编排 -------------------------------------------- #

    async def _run_fallback(self, state: AgentState) -> AgentState:
        current = state
        while True:
            _node, update = await self._step(current)
            current = cast(AgentState, _merge(current, update))
            if current.get("status") in {"SUCCESS", "HUMAN_ESCALATION"}:
                return current

    async def _run_fallback_stream(
        self, state: AgentState
    ) -> AsyncIterator[dict[str, Any]]:
        current = state
        while True:
            node, update = await self._step(current)
            current = cast(AgentState, _merge(current, update))
            yield {node: update}
            if current.get("status") in {"SUCCESS", "HUMAN_ESCALATION"}:
                return

    async def _step(self, state: AgentState) -> tuple[str, dict[str, Any]]:
        status = state.get("status", "LOCATING")
        if status in {"LOCATING", "PLANNING"}:
            if status == "PLANNING" and state.get("repair_plan"):
                return "architect", await architect_node(state, llm=self.deps.llm)
            if status == "LOCATING":
                return "localizer", await localizer_node(
                    state,
                    retriever=self.deps.retriever,  # type: ignore[arg-type]
                    graph=self.deps.graph,
                    llm=self.deps.llm,
                )
            return "architect", await architect_node(state, llm=self.deps.llm)
        if status == "CODING":
            return "coder", await coder_node(state, llm=self.deps.llm, graph=self.deps.graph)
        if status == "TESTING":
            return "tester", await tester_node(state, pool=self.deps.pool, lsp=self.deps.lsp)
        return "critic", await critic_node(state, llm=self.deps.llm)


def _merge(state: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """合并状态更新（对 patch_history / events 采用追加语义）。"""
    merged = dict(state)
    for key, value in update.items():
        if key in ADDITIVE_FIELDS and isinstance(value, list):
            merged[key] = [*merged.get(key, []), *value]
        else:
            merged[key] = value
    return merged


def build_checkpointer() -> Any | None:
    """构建进程内内存 Checkpointer（LangGraph 不可用时返回 None）。

    Postgres 持久化请使用 :func:`open_checkpointer`（异步上下文管理器）。
    """
    try:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    except Exception:  # pragma: no cover - 依赖缺失
        return None


@asynccontextmanager
async def open_checkpointer() -> AsyncIterator[Any | None]:
    """按配置打开持久化 Checkpointer：Postgres 优先，不可用时降级内存。

    在应用生命周期内 ``async with`` 持有；退出时自动关闭连接池。
    断点续跑与状态回溯（Time-Travel）依赖该实例。

    注意：``yield`` 必须在 ``try/except`` 之外——否则调用方（如 lifespan
    主体）抛出的异常会经 ``athrow`` 进入本生成器并被 except 吞掉，
    导致 asynccontextmanager 抛 ``RuntimeError: generator didn't stop after
    athrow()``，原始异常被掩盖。
    """
    if global_settings.postgres.enabled:
        cm: Any = None
        saver: Any = None
        try:  # pragma: no cover - 需要 Postgres
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            cm = AsyncPostgresSaver.from_conn_string(global_settings.postgres.dsn)
            saver = await cm.__aenter__()
            await saver.setup()
        except Exception as exc:
            logger.warning("Postgres Checkpointer 不可用，降级内存", extra={"error": str(exc)})
            if cm is not None:
                try:
                    await cm.__aexit__(None, None, None)
                except Exception:
                    pass
            saver = None
        if saver is not None:
            logger.info("Postgres Checkpointer 已启用")
            try:
                yield saver
            finally:
                await cm.__aexit__(None, None, None)
            return
    yield build_checkpointer()
