"""FastAPI 入口、生命周期管理与 CORS/鉴权中间件。

启动::

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

端点一览：

* ``GET  /health``                       健康检查
* ``POST /api/v1/tasks``                 提交自愈任务
* ``GET  /api/v1/tasks/{id}``            查询任务状态
* ``GET  /api/v1/tasks/{id}/events``     SSE 实时思考流
* ``GET  /api/v1/tasks/{id}/checkpoints`` Checkpoint 状态回溯（Time-Travel）
* ``POST /api/v1/tasks/{id}/intervention`` Time-Travel 人工介入
* ``POST /webhooks/github``              GitHub Webhook 接收器
"""

from __future__ import annotations

import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import __version__
from src.agent.workflow import open_checkpointer
from src.api.routers import task_router, webhook_router
from src.api.schemas import HealthResponse
from src.config import get_logger, setup_logging
from src.config.settings import settings
from src.sandbox.pool_manager import PoolManager

logger = get_logger(__name__)

#: 免鉴权路径：探活/文档/根路径，以及自带 HMAC 验签的 Webhook（不能要求 Bearer token，
#: 否则开启 ``NEXUS_API__AUTH_TOKEN`` 后 GitHub 回调将永久 401）。
_PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json", "/", "/webhooks/github"}


def validate_production_config() -> None:
    """生产环境安全基线：缺少鉴权 / 路径白名单时 fail-fast 拒绝启动。

    只打 warning 挡不住误配置：一旦按 README 指引改为对外监听却忘了配
    token / 白名单，就会回到「未鉴权任意目录读写 + 执行目标仓库 pytest」
    的状态（且 repo_path 会被 git init 产生副作用）。
    """
    if not settings.is_prod:
        return
    problems: list[str] = []
    if not settings.api.auth_token:
        problems.append("NEXUS_API__AUTH_TOKEN 未配置：接口将处于无鉴权状态")
    if not settings.api.allowed_repo_root_list:
        problems.append(
            "NEXUS_API__ALLOWED_REPO_ROOTS 未配置：repo_path 可指向任意目录"
            "（会被 git init / 执行测试，产生副作用）"
        )
    if settings.sandbox.backend != "docker" and not settings.sandbox.allow_local_in_prod:
        problems.append(
            "沙箱后端为 local：被测代码将以宿主权限执行（由 Issue 触发的仓库"
            "可能含恶意代码）；生产环境请设置 NEXUS_SANDBOX__BACKEND=docker，"
            "或显式设置 NEXUS_SANDBOX__ALLOW_LOCAL_IN_PROD=true 承担风险"
        )
    if problems:
        raise RuntimeError("生产环境配置校验失败：\n- " + "\n- ".join(problems))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期：初始化日志、校验配置、装配 Checkpointer 与容器池、优雅关闭。"""
    setup_logging(settings.log_level, json_output=settings.log_json, force=True)
    logger.info(
        "Nexus-Code-Agent API 启动",
        extra={
            "env": settings.env,
            "llm": settings.llm.provider,
            "sandbox": settings.sandbox.backend,
            "qdrant": settings.retrieval.use_qdrant,
        },
    )
    validate_production_config()
    # 容器池归应用生命周期管理：启动预热（docker 后端），关闭时统一回收，
    # 避免每个任务自建池导致容器只增不减。
    pool = PoolManager()
    if pool.backend == "docker":
        await pool.warm_up()
    task_router.set_shared_pool(pool)
    try:
        async with open_checkpointer() as checkpointer:
            task_router.set_checkpointer(checkpointer)
            logger.info(
                "Checkpointer 装配完成",
                extra={"kind": type(checkpointer).__name__ if checkpointer else "none"},
            )
            yield
    finally:
        task_router.set_shared_pool(None)
        await pool.aclose()
    logger.info("Nexus-Code-Agent API 关闭")


def create_app() -> FastAPI:
    """构建 FastAPI 应用（便于测试注入）。"""
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="企业级自动化代码自愈与演进智能体平台 API 网关",
        lifespan=lifespan,
    )

    origins = settings.api.cors_origin_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # 通配来源 + 凭证并存会被浏览器直接拒绝，仅在显式来源时启用凭证
        allow_credentials=origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        token = settings.api.auth_token
        path = request.url.path
        # CORS 预检（OPTIONS）由浏览器自动发起、不携带 Authorization；鉴权中间件
        # 位于 CORSMiddleware 外层，若在此 401，浏览器侧的所有跨域调用（含
        # /docs 调试与前端 SSE）都会在预检阶段失败。预检不含业务数据，直接放行。
        if request.method == "OPTIONS":
            return await call_next(request)
        if token and path not in _PUBLIC_PATHS and not path.startswith(("/docs", "/redoc", "/openapi")):
            header = request.headers.get("authorization", "")
            provided = header[7:] if header.lower().startswith("bearer ") else request.headers.get("x-api-token", "")
            # 常量时间比较，避免时序侧信道
            if not hmac.compare_digest(provided, token):
                return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        return await call_next(request)

    app.include_router(task_router.router)
    app.include_router(webhook_router.router)

    @app.get("/", summary="平台信息")
    async def root() -> dict[str, str]:
        return {
            "app": settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health", response_model=HealthResponse, summary="健康检查")
    async def health() -> HealthResponse:
        return HealthResponse(app=settings.app_name, env=settings.env, version=__version__)

    return app


app = create_app()
