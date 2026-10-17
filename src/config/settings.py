"""Pydantic-Settings 多环境配置管理 (Dev/Prod)。

所有配置项均可通过 ``NEXUS_`` 前缀的环境变量或 ``.env`` 文件覆盖，
嵌套配置使用双下划线分隔，例如::

    NEXUS_API__PORT=9000
    NEXUS_SANDBOX__BACKEND=docker
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Env = Literal["dev", "prod", "test"]


class ApiSettings(BaseModel):
    """异步服务网关配置。"""

    host: str = "127.0.0.1"  # 默认仅监听本机；容器/局域网部署需显式改为 0.0.0.0
    port: int = 8000
    cors_origins: str = "*"
    auth_token: str = ""
    allowed_repo_roots: str = ""  # 逗号分隔；非空时 repo_path 必须落在白名单目录内
    request_timeout_s: int = 900

    @property
    def cors_origin_list(self) -> list[str]:
        if not self.cors_origins or self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_repo_root_list(self) -> list[str]:
        if not self.allowed_repo_roots:
            return []
        return [r.strip() for r in self.allowed_repo_roots.split(",") if r.strip()]


class RetrievalSettings(BaseModel):
    """代码图谱混合检索配置。"""

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    collection: str = "nexus_code"
    embedding_dim: int = 256
    top_k: int = 12
    rrf_k: int = 60
    use_qdrant: bool = False
    bm25_weight: float = 1.0
    dense_weight: float = 1.0


class LLMSettings(BaseModel):
    """LLM 提供方配置。"""

    provider: Literal["heuristic", "openai", "deepseek"] = "heuristic"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 4096
    request_timeout_s: int = 120


class SandboxSettings(BaseModel):
    """沙箱执行环境配置。"""

    backend: Literal["local", "docker"] = "local"
    allow_local_in_prod: bool = False  # prod 下显式确认才允许 local 沙箱（无隔离）
    image: str = "nexus-sandbox:latest"
    pool_size: int = 2
    cpu_limit: float = 2.0
    memory_limit_mb: int = 2048
    test_timeout_s: int = 300
    command_timeout_s: int = 120
    acquire_timeout_s: int = 600  # 等待空闲容器上限（秒）；0 = 不限制（本地调试用）
    oom_grace_mb: int = 512
    worktree_root: str = ".nexus/worktrees"
    max_output_bytes: int = 200_000

    @field_validator("pool_size")
    @classmethod
    def _positive_pool(cls, v: int) -> int:
        if v < 0:
            raise ValueError("pool_size 必须 >= 0")
        return v


class PostgresSettings(BaseModel):
    """Postgres 状态持久化配置。"""

    dsn: str = "postgresql://nexus:nexus@localhost:5432/nexus"
    enabled: bool = False


class RedisSettings(BaseModel):
    """Redis 缓存配置。"""

    url: str = "redis://localhost:6379/0"
    enabled: bool = False


class AgentSettings(BaseModel):
    """LangGraph 编排与防劣化参数。"""

    max_iterations: int = 6
    same_error_streak: int = 2
    stall_streak: int = 2
    token_budget: int = 120_000
    context_window: int = 32_000
    graph_recursion_limit: int = 50
    n_hop: int = 2


class GithubSettings(BaseModel):
    """GitHub 集成配置。"""

    token: str = ""
    webhook_secret: str = ""
    allow_unverified_webhooks: bool = False  # secret 未配置时是否放行 Webhook（默认 fail-closed）
    auto_pr: bool = True
    api_base: str = "https://api.github.com"
    repo_root: str = ""  # 本地仓库根目录：<repo_root>/<repo_name> 映射 GitHub 仓库


class Settings(BaseSettings):
    """全局配置根对象。"""

    model_config = SettingsConfigDict(
        env_prefix="NEXUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Nexus-Code-Agent"
    env: Env = "dev"
    debug: bool = True
    log_level: str = "INFO"
    log_json: bool = True

    api: ApiSettings = Field(default_factory=ApiSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    github: GithubSettings = Field(default_factory=GithubSettings)

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"

    @property
    def is_test(self) -> bool:
        return self.env == "test"

    def resolve_path(self, value: str) -> Path:
        """将相对路径解析为相对当前工作目录 (cwd) 的绝对路径。"""
        p = Path(value)
        return p if p.is_absolute() else (Path.cwd() / p)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程级单例配置。"""
    return Settings()


#: 便捷单例（导入即完成一次解析）。
settings: Settings = get_settings()
