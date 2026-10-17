"""请求与响应 Pydantic 模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskSubmitRequest(BaseModel):
    """提交自愈任务。"""

    repo_path: str = Field(..., description="本地仓库绝对路径")
    issue_description: str = Field(..., min_length=1, description="Issue 描述 / 报错日志 / 自然语言指令")
    task_id: str | None = Field(default=None, description="可选，自定义任务 ID")
    max_iterations: int | None = Field(default=None, ge=1, le=20)


class TaskSubmitResponse(BaseModel):
    task_id: str
    status: str


class PatchRecordSchema(BaseModel):
    iteration: int = 0
    lsp_passed: bool = False
    test_passed: bool = False
    error_fingerprint: str = ""
    error_summary: str = ""
    strategy: str = ""


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    repo_path: str
    issue_description: str
    current_iteration: int = 0
    max_iterations: int = 0
    suspected_files: list[str] = Field(default_factory=list)
    repair_plan: str = ""
    worktree_path: str = ""
    token_usage: int = 0
    termination_reason: str = ""
    final_diff: str = ""
    patch_history: list[PatchRecordSchema] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)


class InterventionRequest(BaseModel):
    """Time-Travel 人工介入请求。"""

    human_feedback: str = Field(default="", description="人工补充的修复建议 / 约束")
    resume: bool = True


class InterventionResponse(BaseModel):
    task_id: str
    status: str
    message: str


class WebhookResponse(BaseModel):
    accepted: bool
    task_id: str | None = None
    detail: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    app: str
    env: str
    version: str
