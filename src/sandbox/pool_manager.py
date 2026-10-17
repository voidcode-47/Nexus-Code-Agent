"""Docker 容器预热连接池（解决容器冷启动慢痛点）。

* **预热池**：后台维护固定数量的常驻休眠容器，环境分配从 5~10s 降至 **50ms 以内**；
* **双层隔离**：容器提供进程/文件系统隔离（cgroups 限制 CPU/Memory），
  Git Worktree 提供代码状态隔离（见 :mod:`src.sandbox.git_snapshot`）；
* **优雅降级**：Docker 不可用或 ``backend=local`` 时，自动回退为本地子进程沙箱，
  行为接口保持一致，便于离线开发与 CI。

用法::

    pool = PoolManager()
    await pool.warm_up()
    async with pool.session(worktree_dir) as session:
        result = await session.exec("pytest -q tests/")
    await pool.aclose()
"""

from __future__ import annotations

import asyncio
import re
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import get_logger
from src.config.settings import SandboxSettings
from src.config.settings import settings as global_settings
from src.sandbox.executor import CommandResult, run_command

logger = get_logger(__name__)

try:  # docker SDK 为可选依赖
    import docker
except Exception:  # pragma: no cover
    docker = None


@dataclass
class SandboxSession:
    """一次沙箱会话（异步上下文管理器）。"""

    workdir: Path
    pool: PoolManager
    _container: Any = field(default=None, init=False, repr=False)
    _acquired: bool = field(default=False, init=False)

    async def __aenter__(self) -> SandboxSession:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.release()

    async def acquire(self) -> SandboxSession:
        if self.pool.backend == "docker":
            self._container = await self.pool._acquire_container()
        self._acquired = True
        return self

    async def release(self) -> None:
        if not self._acquired:
            return
        if self.pool.backend == "docker" and self._container is not None:
            await self.pool._release_container(self._container)
            self._container = None
        self._acquired = False

    @property
    def backend(self) -> str:
        return self.pool.backend

    async def exec(
        self,
        command: str | Sequence[str],
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        memory_limit_mb: int | None = None,
    ) -> CommandResult:
        """在沙箱中执行命令。"""
        timeout = timeout or self.pool.settings.command_timeout_s
        if self.pool.backend == "docker" and self._container is not None:
            return await self._exec_docker(command, timeout=timeout, env=env)
        return await run_command(
            command,
            cwd=self.workdir,
            timeout=timeout,
            env=env,
            max_output_bytes=self.pool.settings.max_output_bytes,
            memory_limit_mb=memory_limit_mb or self.pool.settings.memory_limit_mb,
            oom_grace_mb=self.pool.settings.oom_grace_mb,
        )

    def _python_executable(self) -> str:
        """按后端选择 pytest 解释器。

        local：当前解释器（避免 venv 未激活时跑错环境）；
        docker：镜像内的 ``python``——宿主路径（如 ``C:\\...python.exe``）
        在容器中并不存在，曾导致 docker 后端 exec 直接失败。
        """
        return "python" if self.pool.backend == "docker" else sys.executable

    async def run_pytest(
        self,
        targets: Sequence[str] = ("tests",),
        *,
        extra_args: Sequence[str] = ("-q",),
        timeout: float | None = None,
    ) -> CommandResult:
        """执行 pytest 测试套件（解释器按沙箱后端选择）。"""
        cmd = [self._python_executable(), "-m", "pytest", *extra_args, *targets]
        return await self.exec(
            cmd,
            timeout=timeout or self.pool.settings.test_timeout_s,
        )

    async def _exec_docker(
        self,
        command: str | Sequence[str],
        *,
        timeout: float,
        env: dict[str, str] | None,
    ) -> CommandResult:  # pragma: no cover - 需要 Docker
        import time

        container = self._container
        rel = self._relative_cwd()
        start = time.monotonic()

        def _run() -> tuple[int, bytes, bytes]:
            result = container.exec_run(
                command,
                workdir=rel,
                demux=True,
                environment=env or {},
            )
            out, err = result.output if isinstance(result.output, tuple) else (result.output, b"")
            return result.exit_code, out or b"", err or b""

        try:
            exit_code, out, err = await asyncio.wait_for(asyncio.to_thread(_run), timeout=timeout)
        except asyncio.TimeoutError:
            # 终止容器内仍在运行的被测进程，避免僵尸进程与滞留线程
            await self._kill_container_work(command)
            return CommandResult(
                command=_cmd_str(command),
                returncode=-1,
                stdout="",
                stderr="",
                duration_s=time.monotonic() - start,
                timed_out=True,
            )
        max_bytes = self.pool.settings.max_output_bytes
        return CommandResult(
            command=_cmd_str(command),
            returncode=exit_code,
            stdout=out[:max_bytes].decode("utf-8", "replace"),
            stderr=err[:max_bytes].decode("utf-8", "replace"),
            duration_s=time.monotonic() - start,
            truncated=len(out) > max_bytes or len(err) > max_bytes,
        )

    #: 无 procps（pkill）时的兜底：扫描 /proc 按 cmdline 匹配并 kill
    _KILL_FALLBACK_SH = (
        'pat="$1"; for p in /proc/[0-9]*; do '
        'pid="${p#/proc/}"; [ "$pid" = "$$" ] && continue; '
        'grep -aqF "$pat" "$p/cmdline" 2>/dev/null && kill -TERM "$pid" 2>/dev/null; '
        "done; exit 0"
    )

    async def _kill_container_work(self, command: str | Sequence[str]) -> None:  # pragma: no cover - 需要 Docker
        """超时后终止容器内对应的被测进程（防止僵尸进程 / exec 线程滞留）。

        slim 基础镜像默认不带 procps（pkill）；exit_code=127 时用 /proc
        扫描兜底，避免「以为杀了、实际是空操作」。
        """
        if self._container is None:
            return
        pattern = re.escape(_cmd_str(command))
        try:
            result = await asyncio.to_thread(
                self._container.exec_run, ["pkill", "-TERM", "-f", pattern]
            )
            exit_code = (
                result[0]
                if isinstance(result, (tuple, list))
                else getattr(result, "exit_code", 0)
            )
        except Exception:
            logger.warning("容器内超时进程终止失败（可忽略）")
            return
        if exit_code in (0, 1):  # 0=已杀；1=无匹配进程，均属正常
            return
        if exit_code == 127:
            logger.warning("容器内缺少 pkill（建议镜像安装 procps），改用 /proc 扫描兜底")
            try:
                await asyncio.to_thread(
                    self._container.exec_run,
                    ["sh", "-c", self._KILL_FALLBACK_SH, "kill-scan", pattern],
                )
            except Exception:
                logger.warning("容器内兜底终止失败（可忽略）")
        else:
            logger.warning("容器内进程终止返回非零", extra={"exit_code": exit_code})

    def _relative_cwd(self) -> str:
        root = self.pool.container_mount_root
        try:
            rel = self.workdir.resolve().relative_to(root.resolve())
            return f"/workspace/{rel.as_posix()}"
        except ValueError:
            logger.warning(
                "worktree 目录不在容器挂载根下，退化为 /workspace（请检查 worktree_root 配置）",
                extra={"workdir": str(self.workdir), "mount_root": str(root)},
            )
            return "/workspace"


def _cmd_str(command: str | Sequence[str]) -> str:
    return command if isinstance(command, str) else " ".join(command)


@dataclass
class PoolManager:
    """沙箱容器池管理器。"""

    settings: SandboxSettings = field(default_factory=lambda: global_settings.sandbox)
    container_mount_root: Path = field(default_factory=lambda: Path.cwd())
    _queue: asyncio.Queue | None = field(default=None, init=False, repr=False)
    _client: Any = field(default=None, init=False, repr=False)
    _all_containers: list[Any] = field(default_factory=list, init=False, repr=False)
    _backend: str = field(default="local", init=False)
    _warmup_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _acquire_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._backend = self._resolve_backend()

    def _resolve_backend(self) -> str:
        if self.settings.backend != "docker":
            return "local"
        if docker is None:
            return self._fallback_backend("docker SDK 不可用")
        try:
            client = docker.from_env()
            client.ping()
            self._client = client
            if self.settings.worktree_root:
                root = Path(self.settings.worktree_root)
                if not root.is_absolute():
                    root = Path.cwd() / root
                self.container_mount_root = root
            logger.info("docker sandbox 已启用", extra={"image": self.settings.image})
            return "docker"
        except Exception as exc:
            return self._fallback_backend(f"docker daemon 不可用：{exc}")

    def _fallback_backend(self, reason: str) -> str:
        """决定「降级为本地沙箱」还是「拒绝启动」。

        容器是生产环境唯一的执行隔离边界：``validate_production_config`` 只校验
        **配置值**为 docker，若这里再静默降级为 local，被测仓库（可能来自外部
        Issue / PR）就会绕过保护以宿主权限执行——安全基线被绕过却不报错。
        因此 prod 下 fail-fast；仅当运维显式设置
        ``NEXUS_SANDBOX__ALLOW_LOCAL_IN_PROD=true`` 认领风险时才降级。
        """
        if global_settings.is_prod and not self.settings.allow_local_in_prod:
            raise RuntimeError(
                f"{reason}；NEXUS_ENV=prod 下沙箱后端要求 docker，拒绝静默降级为宿主执行。"
                "请修复 Docker 环境（镜像 make sandbox-image、daemon 可连通），"
                "或显式设置 NEXUS_SANDBOX__ALLOW_LOCAL_IN_PROD=true 承担风险"
            )
        logger.warning("%s，回退本地沙箱", reason)
        return "local"

    @property
    def backend(self) -> str:
        return self._backend

    def has_active_containers(self) -> bool:
        """是否已有创建好的容器。

        容器 bind 的挂载根在创建时固定，事后改 ``container_mount_root``
        不会影响已存在的容器——调用方必须据此拒绝"预热后改挂载根"。
        """
        return bool(self._all_containers)

    # -- 生命周期 --------------------------------------------------------- #

    async def warm_up(self) -> None:
        """预热容器池（幂等；并发调用由内部锁串行化，避免队列被覆盖产生孤儿容器）。

        ``pool_size <= 0`` 时只初始化队列、不预建容器，后续按需创建。
        """
        if self._backend != "docker":
            return
        async with self._warmup_lock:
            if self._queue is not None:
                return
            self._queue = asyncio.Queue()
            if self.settings.pool_size <= 0:
                return
            self.container_mount_root.mkdir(parents=True, exist_ok=True)
            await asyncio.gather(
                *(self._spawn_container() for _ in range(self.settings.pool_size))
            )
        logger.info("容器池预热完成", extra={"size": self.settings.pool_size})

    async def aclose(self) -> None:
        """停止并移除全部容器。"""
        if self._backend != "docker":
            return

        def _cleanup() -> None:
            for container in self._all_containers:
                try:
                    container.stop(timeout=3)
                except Exception:
                    continue
                try:
                    container.remove(force=True)
                except Exception:
                    continue

        await asyncio.to_thread(_cleanup)
        self._all_containers.clear()
        self._queue = asyncio.Queue()

    # -- 会话 ------------------------------------------------------------- #

    def session(self, workdir: str | Path) -> SandboxSession:
        """创建沙箱会话（需 ``async with`` 使用）。"""
        path = Path(workdir)
        if not path.is_absolute():
            path = Path.cwd() / path
        return SandboxSession(workdir=path, pool=self)

    # -- Docker 容器池内部 ------------------------------------------------- #

    async def _spawn_container(self) -> None:
        assert self._queue is not None
        container = await asyncio.to_thread(self._create_container)
        self._all_containers.append(container)
        await self._queue.put(container)

    def _create_container(self) -> Any:  # pragma: no cover - 需要 Docker
        assert self._client is not None
        return self._client.containers.run(
            self.settings.image,
            command="sleep infinity",
            detach=True,
            auto_remove=False,
            working_dir="/workspace",
            volumes={str(self.container_mount_root.resolve()): {"bind": "/workspace", "mode": "rw"}},
            mem_limit=f"{self.settings.memory_limit_mb}m",
            nano_cpus=int(self.settings.cpu_limit * 1e9),
            network_disabled=False,
            environment={"PYTHONUNBUFFERED": "1"},
        )

    async def _acquire_container(self) -> Any:
        """取一个空闲容器；总数达 ``pool_size`` 上限时等待其他会话释放。

        旧实现在队列空时无上限地新建容器，长跑进程的容器数会被历史并发
        峰值钉住（峰值过后遗留大量闲置容器无法回收）。
        """
        if self._queue is None:
            await self.warm_up()
        assert self._queue is not None
        queue = self._queue
        async with self._acquire_lock:
            if not queue.empty():
                return queue.get_nowait()
            if len(self._all_containers) < max(1, self.settings.pool_size):
                container = await asyncio.to_thread(self._create_container)
                self._all_containers.append(container)
                return container
        # 已达上限：等待其他会话释放（session.release 会放回队列）。
        # 等待必须有界：容器被长时间占用（卡死的被测进程等）时，无超时的
        # queue.get() 会让任务永久挂起，且日志里看不出卡在哪里。
        timeout = self.settings.acquire_timeout_s
        try:
            return await asyncio.wait_for(
                queue.get(), timeout=timeout if timeout > 0 else None
            )
        # 必须捕获 asyncio.TimeoutError：Python 3.10 下 wait_for 抛的是
        # asyncio.TimeoutError（concurrent.futures 系），与内置 TimeoutError
        # 并非同一类型，写成内置类型会漏捕获并让任务悬空等待。
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"等待空闲沙箱容器超过 {timeout}s（pool_size={self.settings.pool_size}）："
                "可能有会话长时间占用容器，请检查被测进程是否卡死，或调大池容量 / 超时"
            ) from exc

    async def _release_container(self, container: Any) -> None:
        if self._queue is None:
            return
        try:
            # reload() 是同步 HTTP 调用，必须放到线程里避免阻塞事件循环
            await asyncio.to_thread(container.reload)
            if container.status != "running":
                await asyncio.to_thread(container.restart)
        except Exception:  # pragma: no cover
            pass
        await self._queue.put(container)

    # -- 工具 ------------------------------------------------------------- #

    @staticmethod
    def docker_available() -> bool:
        if docker is None:
            return False
        try:
            docker.from_env().ping()
            return True
        except Exception:
            return False

    @staticmethod
    def local_available() -> bool:
        return shutil.which("python") is not None or Path(sys.executable).exists()
