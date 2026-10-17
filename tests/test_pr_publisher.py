"""PR 发布器测试：离线降级、真实推送到本地 bare 远端、PR 创建（本地假 API）。"""

from __future__ import annotations

import difflib
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from src.mcp_tools.pr_publisher import create_pull_request, publish_patch


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )


@pytest.fixture
def patch_text(tmp_git_repo: Path) -> str:
    """从真实文件内容生成标准 Unified Diff。"""
    source = (tmp_git_repo / "calculator.py").read_text(encoding="utf-8")
    patched = source.replace("return a - b", "return a + b", 1)
    return "".join(
        difflib.unified_diff(
            source.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile="a/calculator.py",
            tofile="b/calculator.py",
        )
    )


@pytest.fixture
def repo_with_remote(tmp_git_repo: Path, tmp_path: Path) -> tuple[Path, Path]:
    """准备一个带 bare 远端 origin 的仓库。"""
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    _git(tmp_git_repo, "remote", "add", "origin", str(bare))
    _git(tmp_git_repo, "push", "-u", "origin", "HEAD")
    return tmp_git_repo, bare


class _FakeGitHubHandler(BaseHTTPRequestHandler):
    status_code = 201
    response: dict = {}
    requests: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        type(self).requests.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization", ""),
                "json": json.loads(body or b"{}"),
            }
        )
        payload = json.dumps(type(self).response).encode("utf-8")
        self.send_response(type(self).status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:  # 静默测试输出
        return


@pytest.fixture
def fake_github():
    _FakeGitHubHandler.requests = []
    _FakeGitHubHandler.status_code = 201
    _FakeGitHubHandler.response = {"html_url": "https://github.com/acme/demo/pull/1"}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeGitHubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


# --------------------------------------------------------------------------- #
# 降级路径
# --------------------------------------------------------------------------- #


async def test_publish_empty_patch_is_noop(tmp_git_repo: Path):
    result = await publish_patch(tmp_git_repo, "   ", title="empty")
    assert not result.published and not result.pushed
    assert "空" in result.reason


async def test_publish_without_remote_degrades(tmp_git_repo: Path, patch_text: str):
    result = await publish_patch(tmp_git_repo, patch_text, title="fix add")
    assert not result.pushed and not result.published
    assert "远端" in result.reason
    # 源仓库工作区未被污染
    assert _git(tmp_git_repo, "status", "--porcelain").stdout.strip() == ""


async def test_publish_requires_token_for_pr(repo_with_remote, patch_text: str):
    repo, bare = repo_with_remote
    result = await publish_patch(repo, patch_text, title="fix add")
    assert result.pushed and not result.published
    assert "token" in result.reason.lower() or "跳过 PR" in result.reason

    # 分支已真实到达远端，且包含修复
    verify = _git(bare, "rev-parse", "--verify", f"refs/heads/{result.branch}")
    assert verify.returncode == 0
    show = _git(bare, "show", f"{result.branch}:calculator.py").stdout
    assert "return a + b" in show
    # 本地临时分支引用已清理，工作区保持干净
    assert _git(repo, "rev-parse", "--verify", result.branch, check=False).returncode != 0
    assert _git(repo, "status", "--porcelain").stdout.strip() == ""


# --------------------------------------------------------------------------- #
# 完整路径：推送 + PR
# --------------------------------------------------------------------------- #


async def test_publish_creates_pr(repo_with_remote, patch_text: str, fake_github: str):
    repo, bare = repo_with_remote
    result = await publish_patch(
        repo,
        patch_text,
        title="fix(add): 修正加法实现",
        body="自动修复：add() 误写为减法",
        repo_full="acme/demo",
        token="test-token",
        api_base=fake_github,
    )
    assert result.published, result.reason
    assert result.pr_url == "https://github.com/acme/demo/pull/1"
    assert result.pushed and result.branch and result.commit

    # 假 GitHub 收到了合规请求
    requests = _FakeGitHubHandler.requests
    assert len(requests) == 1
    sent = requests[0]
    assert sent["path"] == "/repos/acme/demo/pulls"
    assert sent["authorization"] == "Bearer test-token"
    assert sent["json"]["head"] == result.branch
    assert sent["json"]["base"] == result.base
    assert sent["json"]["title"] == "fix(add): 修正加法实现"

    # 分支内容正确
    show = _git(bare, "show", f"{result.branch}:calculator.py").stdout
    assert "return a + b" in show


async def test_publish_pr_error_degrades(repo_with_remote, patch_text: str, fake_github: str):
    _FakeGitHubHandler.status_code = 422
    _FakeGitHubHandler.response = {"message": "Validation Failed"}
    repo, _bare = repo_with_remote
    result = await publish_patch(
        repo,
        patch_text,
        title="fix add",
        repo_full="acme/demo",
        token="test-token",
        api_base=fake_github,
    )
    assert result.pushed and not result.published
    assert "422" in result.reason


# --------------------------------------------------------------------------- #
# GitHub API 封装
# --------------------------------------------------------------------------- #


async def test_create_pull_request_http_error(fake_github: str):
    _FakeGitHubHandler.status_code = 403
    _FakeGitHubHandler.response = {"message": "rate limited"}
    pr_url, error = await create_pull_request(
        "acme/demo",
        head="nexus/fix-1",
        base="main",
        title="t",
        token="tok",
        api_base=fake_github,
    )
    assert pr_url == ""
    assert "403" in error
