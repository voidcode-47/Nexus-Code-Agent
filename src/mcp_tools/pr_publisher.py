"""Git Patch 发布器：把自愈补丁发布为远端分支与 Pull Request。

补齐文档中「提 Issue 即自动提 PR 修复」的无人值守闭环最后一公里：

1. 在临时 Git Worktree 中创建修复分支并应用补丁（源仓库工作区零污染）；
2. 提交并推送到远端（``origin``）；
3. 通过 GitHub REST API 创建 Pull Request，返回 PR 链接。

**优雅降级**：未配置远端 / 网络不可用 / 缺少 Token 时逐步降级为
「仅导出补丁」→「仅推送分支」→「创建 PR」，并以 :class:`PublishResult` 说明原因，
绝不让发布失败影响主修复流程。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx

from src.config import get_logger
from src.config.settings import settings as global_settings
from src.sandbox.git_snapshot import WorktreeSnapshot, run_git

logger = get_logger(__name__)


@dataclass(slots=True)
class PublishResult:
    """补丁发布结果（逐级降级时各字段含义见 ``reason``）。"""

    branch: str = ""
    base: str = ""
    commit: str = ""
    pushed: bool = False
    published: bool = False  # 是否成功创建 PR
    pr_url: str = ""
    reason: str = ""

    def describe(self) -> str:
        if self.published:
            return f"已创建 PR：{self.pr_url}（分支 {self.branch}）"
        if self.pushed:
            return f"已推送分支 {self.branch}；{self.reason}"
        return self.reason or "未发布"


def _default_branch() -> str:
    return f"nexus/fix-{uuid.uuid4().hex[:8]}"


async def _detect_base(repo: Path, remote: str) -> str:
    """探测远端默认分支（``origin/HEAD`` → main → master 逐级回退）。"""
    head = await run_git(repo, "symbolic-ref", "--short", f"refs/remotes/{remote}/HEAD", check=False)
    if head.ok and head.stdout.strip():
        ref = head.stdout.strip()
        return ref.split("/", 1)[1] if "/" in ref else ref
    for candidate in ("main", "master"):
        probe = await run_git(
            repo, "rev-parse", "--verify", f"refs/remotes/{remote}/{candidate}", check=False
        )
        if probe.ok:
            return candidate
    return "main"


async def publish_patch(
    repo_path: str | Path,
    patch_text: str,
    *,
    title: str,
    body: str = "",
    base: str = "",
    branch: str = "",
    remote: str = "origin",
    repo_full: str = "",
    token: str = "",
    api_base: str = "",
    worktree_root: str | Path | None = None,
    commit_message: str = "",
) -> PublishResult:
    """把标准 Git Patch 发布为远端分支与 Pull Request。

    Args:
        repo_path: 本地仓库路径。
        patch_text: 标准 Unified Diff（如 ``final_diff``）。
        title: PR / 提交标题。
        repo_full: ``owner/repo``；为空则跳过 PR 创建（仅推送分支）。
        token: GitHub Token；留空时读取 ``NEXUS_GITHUB__TOKEN``。
        api_base: GitHub API 地址；留空读取配置（便于测试指向本地假服务）。
    """
    repo = Path(repo_path).resolve()
    if not patch_text.strip():
        return PublishResult(reason="补丁为空，跳过发布")

    branch = branch or _default_branch()
    remote_url = await run_git(repo, "remote", "get-url", remote, check=False)
    if not remote_url.ok:
        return PublishResult(
            branch=branch, reason=f"仓库未配置远端 {remote}，仅保留补丁（跳过推送与 PR）"
        )

    base = base or await _detect_base(repo, remote)
    token = token or global_settings.github.token
    api_base = (api_base or global_settings.github.api_base).rstrip("/")

    snapshot = WorktreeSnapshot(
        repo,
        worktree_root=worktree_root,
        name=f"pr-{uuid.uuid4().hex[:8]}",
        branch=branch,
    )
    await snapshot.create()
    commit = ""
    pushed = False
    try:
        await snapshot.apply_patch(patch_text)
        commit = await snapshot.commit(commit_message or title)
        push = await run_git(
            snapshot.path,
            "push",
            "--force",
            remote,
            f"HEAD:refs/heads/{branch}",
            check=False,
        )
        if not push.ok:
            tail = (push.stderr or push.stdout).strip().splitlines()
            detail = tail[-1] if tail else "未知错误"
            logger.warning("分支推送失败", extra={"branch": branch, "detail": detail})
            return PublishResult(
                branch=branch, base=base, commit=commit, reason=f"分支推送失败：{detail[:300]}"
            )
        pushed = True
    finally:
        await snapshot.cleanup()

    if pushed:
        # 推送成功后删除本地临时分支引用，保持源仓库干净
        await run_git(repo, "branch", "-D", branch, check=False)

    logger.info("修复分支已推送", extra={"branch": branch, "base": base})

    if not (repo_full and token):
        return PublishResult(
            branch=branch,
            base=base,
            commit=commit,
            pushed=True,
            reason="缺少 repo/token（NEXUS_GITHUB__REPO_ROOT 与 NEXUS_GITHUB__TOKEN），跳过 PR 创建",
        )

    pr_url, error = await create_pull_request(
        repo_full,
        head=branch,
        base=base,
        title=title,
        body=body,
        token=token,
        api_base=api_base,
    )
    if error:
        return PublishResult(
            branch=branch, base=base, commit=commit, pushed=True, reason=error
        )
    return PublishResult(
        branch=branch,
        base=base,
        commit=commit,
        pushed=True,
        published=True,
        pr_url=pr_url,
        reason="PR 创建成功",
    )


async def create_pull_request(
    repo_full: str,
    *,
    head: str,
    base: str,
    title: str,
    body: str = "",
    token: str,
    api_base: str = "https://api.github.com",
) -> tuple[str, str]:
    """调用 GitHub REST API 创建 Pull Request。

    Returns:
        ``(pr_url, error)``：成功时 error 为空；失败时 pr_url 为空。
    """
    url = f"{api_base.rstrip('/')}/repos/{repo_full}/pulls"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    payload = {"title": title, "head": head, "base": base, "body": body or ""}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(url, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        logger.warning("PR 创建请求失败", extra={"error": str(exc)})
        return "", f"PR 创建请求失败：{exc}"
    if resp.status_code >= 300:
        return "", f"PR 创建失败 HTTP {resp.status_code}：{resp.text[:200]}"
    return str(resp.json().get("html_url", "")), ""
