"""GitHub Issue / PR 自动触发 Webhook 接收器。

实现「提 Issue 即自动提 PR 修复」的无人值守闭环：

* 校验 ``X-Hub-Signature-256``（HMAC-SHA256）确保来源可信；
* 监听 ``issues.opened`` / ``issues.labeled`` 事件 → 自动创建自愈任务；
* 监听 ``issue_comment.created`` / ``pull_request_review_comment.created``
  （``/nexus fix`` 指令）与 ``pull_request_review.submitted``（changes_requested）→ 触发修复；
* 修复成功后自动推送分支、创建 PR，并回评 Issue / PR（需配置 ``github.token``）。
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Header, HTTPException, Request

from src.api.routers.task_router import _execute, store
from src.api.schemas import WebhookResponse
from src.config import get_logger
from src.config.settings import settings
from src.mcp_tools.pr_publisher import publish_patch

logger = get_logger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

TRIGGER_COMMANDS = ("/nexus fix", "/nexus heal", "@nexus fix")

#: 后台任务的强引用集合。
#:
#: ``asyncio.create_task`` 的返回值若不持有引用，事件循环只保留弱引用，
#: 任务可能在执行中途被 GC 回收（官方文档明确要求保存引用）——表现为
#: 「自动建 PR 偶发不执行」，且没有任何报错。
_background_tasks: set[asyncio.Task] = set()


def _spawn_background(coro: Any) -> asyncio.Task:
    """启动后台任务并持有强引用，完成后自动移出集合。"""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def verify_signature(
    secret: str,
    body: bytes,
    signature: str | None,
    *,
    allow_unverified: bool = False,
) -> bool:
    """校验 GitHub Webhook 签名（默认 fail-closed）。

    未配置 secret 时默认**拒绝**未签名请求；确需在无 secret 的联调环境
    放行时，显式开启 ``NEXUS_GITHUB__ALLOW_UNVERIFIED_WEBHOOKS=true``。
    """
    if not secret:
        return allow_unverified
    if not signature:
        return False
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


@router.post("/github", response_model=WebhookResponse, summary="GitHub Webhook 接收器")
async def github_webhook(
    request: Request,
    x_github_event: str = Header(default=""),
    x_hub_signature_256: str = Header(default=""),
) -> WebhookResponse:
    """接收并处理 GitHub 事件。"""
    body = await request.body()
    if not verify_signature(
        settings.github.webhook_secret,
        body,
        x_hub_signature_256,
        allow_unverified=settings.github.allow_unverified_webhooks,
    ):
        raise HTTPException(status_code=401, detail="invalid webhook signature")

    try:
        payload: dict[str, Any] = json.loads(body.decode("utf-8") or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON payload") from exc

    action = payload.get("action", "")
    repo_full = str(payload.get("repository", {}).get("full_name", ""))
    local_repo = _resolve_local_repo(repo_full)

    if x_github_event == "issues" and action in {"opened", "labeled"}:
        issue = payload.get("issue", {})
        title = issue.get("title", "")
        body_text = issue.get("body") or ""
        description = f"{title}\n\n{body_text}".strip()
        return await _dispatch(
            repo_full, local_repo, description, source="issue", issue_number=_issue_number(payload)
        )

    if x_github_event == "issue_comment" and action == "created":
        comment = str(payload.get("comment", {}).get("body", ""))
        if any(cmd in comment.lower() for cmd in TRIGGER_COMMANDS):
            issue = payload.get("issue", {})
            description = f"{issue.get('title', '')}\n\n{issue.get('body') or ''}".strip()
            return await _dispatch(
                repo_full,
                local_repo,
                description,
                source="command",
                issue_number=_issue_number(payload),
            )

    if x_github_event == "pull_request_review_comment" and action == "created":
        comment = str(payload.get("comment", {}).get("body", ""))
        if any(cmd in comment.lower() for cmd in TRIGGER_COMMANDS):
            pr = payload.get("pull_request", {})
            description = (
                f"{pr.get('title', '')}\n\n{pr.get('body') or ''}\n\n[Review 评论] {comment}"
            ).strip()
            return await _dispatch(
                repo_full,
                local_repo,
                description,
                source="review_comment",
                issue_number=_issue_number(payload),
            )

    if x_github_event == "pull_request_review" and action == "submitted":
        review = payload.get("review", {})
        review_body = str(review.get("body") or "")
        if review.get("state") == "changes_requested" or any(
            cmd in review_body.lower() for cmd in TRIGGER_COMMANDS
        ):
            pr = payload.get("pull_request", {})
            description = (
                f"{pr.get('title', '')}\n\n{pr.get('body') or ''}\n\n[Review] {review_body}"
            ).strip()
            return await _dispatch(
                repo_full,
                local_repo,
                description,
                source="review",
                issue_number=_issue_number(payload),
            )

    return WebhookResponse(accepted=False, detail=f"事件 {x_github_event}.{action} 未匹配触发条件")


def _issue_number(payload: dict[str, Any]) -> int | None:
    """从事件负载中提取 Issue / PR 编号（用于回评）。"""
    for key in ("issue", "pull_request"):
        number = payload.get(key, {}).get("number")
        if isinstance(number, int):
            return number
    return None


async def _dispatch(
    repo_full: str,
    local_repo: str | None,
    description: str,
    *,
    source: str,
    issue_number: int | None = None,
) -> WebhookResponse:
    if not local_repo:
        detail = f"未找到 {repo_full} 的本地仓库映射（配置 NEXUS_GITHUB__REPO_ROOT）"
        logger.warning("webhook 无法定位本地仓库", extra={"repo": repo_full})
        return WebhookResponse(accepted=False, detail=detail)

    record = await store.create(local_repo, description or "GitHub Issue 自动修复")
    record.handle = asyncio.create_task(_execute(record))
    logger.info(
        "webhook 触发自愈任务",
        extra={"source": source, "repo": repo_full, "task_id": record.task_id},
    )
    if settings.github.auto_pr:
        _spawn_background(_publish_after(repo_full, local_repo, record, issue_number))
    return WebhookResponse(accepted=True, task_id=record.task_id, detail=f"已排队自愈任务（{source}）")


async def _publish_after(
    repo_full: str,
    local_repo: str,
    record: Any,
    issue_number: int | None,
) -> None:
    """任务成功后自动提交 PR 并回评（后台任务，失败绝不影响主流程）。

    注意：本函数由 ``asyncio.create_task`` 启动，异常无人 await 接管——
    必须整体兜底并结构化打日志，否则只剩事件循环的一条
    "Task exception was never retrieved"，自动建 PR 的失败原因对运维
    完全不可见。典型失败场景：final_diff 相对任务开始时的 HEAD，若运行
    期间源仓库 HEAD 前移，git apply 会失败。
    """
    try:
        if record.handle is not None:
            try:
                await record.handle
            except Exception:  # pragma: no cover - _execute 已捕获异常
                return
        if record.status != "SUCCESS":
            return

        patch = str(record.state.get("final_diff", ""))
        headline = (record.issue_description or "自动修复").splitlines()[0][:72]
        result = await publish_patch(
            local_repo,
            patch,
            title=f"fix: {headline}",
            body=f"由 Nexus-Code-Agent 自动修复生成。任务 ID：`{record.task_id}`",
            repo_full=repo_full,
        )
        logger.info(
            "补丁发布完成",
            extra={"task_id": record.task_id, "detail": result.describe()},
        )
        if settings.github.token and issue_number is not None:
            await _comment_back(repo_full, issue_number, f"Nexus-Code-Agent 修复完成。{result.describe()}")
    except Exception as exc:
        logger.exception(
            "自动发布 PR / 回评失败（任务本身结果不受影响）",
            extra={"task_id": getattr(record, "task_id", "?"), "error": str(exc)},
        )


def _resolve_local_repo(repo_full: str) -> str | None:
    """将 GitHub 仓库名映射到本地目录（严格限制在配置的 repo_root 内）。

    未配置 ``repo_root`` 时不再回退 CWD 猜测，防止外部请求自选本地目录；
    仓库名做白名单字符校验并做路径包含检查，杜绝 ``..`` 穿越。
    """
    if not repo_full:
        return None
    root = settings.github.repo_root
    if not root:
        logger.warning(
            "未配置 NEXUS_GITHUB__REPO_ROOT，拒绝从 CWD 猜测本地仓库路径",
            extra={"repo": repo_full},
        )
        return None
    name = repo_full.split("/")[-1]
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        logger.warning("非法 GitHub 仓库名", extra={"repo": repo_full})
        return None
    root_path = Path(root).expanduser().resolve()
    candidate = (root_path / name).resolve()
    if not candidate.is_relative_to(root_path):
        return None
    if candidate.exists():
        return str(candidate)
    return None


async def _comment_back(repo_full: str, issue_number: int, body: str) -> None:  # pragma: no cover - 需要网络
    """（可选）在 Issue / PR 下回评（官方 API 需 Issue 编号）。"""
    url = f"{settings.github.api_base}/repos/{repo_full}/issues/{issue_number}/comments"
    headers = {
        "Authorization": f"Bearer {settings.github.token}",
        "Accept": "application/vnd.github+json",
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(url, headers=headers, json={"body": body})
    except httpx.HTTPError as exc:
        logger.warning("回评 GitHub 失败", extra={"error": str(exc)})
