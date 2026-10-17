"""API 网关测试：健康检查、任务生命周期、鉴权与 Webhook。"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import subprocess
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.config.settings import settings


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def _sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _post_webhook(
    client: TestClient,
    payload: dict,
    event: str,
    *,
    secret: str = "topsecret",
):
    """以合法 HMAC 签名发送 Webhook（模拟 GitHub 的行为）。"""
    body = json.dumps(payload).encode()
    return client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-GitHub-Event": event,
            "X-Hub-Signature-256": _sign(secret, body),
            "Content-Type": "application/json",
        },
    )


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["app"]


def test_root_and_openapi(client: TestClient):
    assert client.get("/").status_code == 200
    spec = client.get("/openapi.json").json()
    assert "/api/v1/tasks" in spec["paths"]
    assert "/webhooks/github" in spec["paths"]


def test_unknown_task_returns_404(client: TestClient):
    assert client.get("/api/v1/tasks/nope").status_code == 404
    assert client.post("/api/v1/tasks/nope/intervention", json={"human_feedback": "x"}).status_code == 404


def test_list_tasks_empty(client: TestClient):
    resp = client.get("/api/v1/tasks")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_auth_required_when_token_set(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings.api, "auth_token", "s3cret")
    assert client.get("/api/v1/tasks").status_code == 401
    ok = client.get("/api/v1/tasks", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    # 健康检查始终公开
    assert client.get("/health").status_code == 200


def test_cors_preflight_not_blocked_by_auth(client: TestClient, monkeypatch):
    """开启鉴权后 CORS 预检仍必须放行，否则浏览器侧所有跨域调用都失败。

    回归背景：鉴权中间件注册在 CORSMiddleware 外层，而预检 OPTIONS 请求由
    浏览器自动发起、不携带 Authorization；旧实现在此处直接 401，正文错误地
    变成「未鉴权」而非 CORS 响应头，`/docs` 调试与前端 SSE 均不可用。
    """
    monkeypatch.setattr(settings.api, "auth_token", "s3cret")
    resp = client.options(
        "/api/v1/tasks",
        headers={
            "Origin": "https://app.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,authorization",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "*"
    # 真正的业务请求仍然必须鉴权
    blocked = client.post(
        "/api/v1/tasks", json={"repo_path": "/tmp", "issue_description": "x"}
    )
    assert blocked.status_code == 401


def test_submit_rejects_nonexistent_repo(client: TestClient, tmp_path: Path):
    resp = client.post(
        "/api/v1/tasks",
        json={"repo_path": str(tmp_path / "missing"), "issue_description": "x"},
    )
    assert resp.status_code == 400


def test_submit_enforces_allowed_repo_roots(
    client: TestClient, monkeypatch, tmp_git_repo: Path, tmp_path: Path
):
    """配置 NEXUS_API__ALLOWED_REPO_ROOTS 后，白名单外的 repo_path 必须被拒绝。"""
    other = tmp_path / "other-root"
    other.mkdir()
    monkeypatch.setattr(settings.api, "allowed_repo_roots", str(other))
    resp = client.post(
        "/api/v1/tasks",
        json={"repo_path": str(tmp_git_repo), "issue_description": "x"},
    )
    assert resp.status_code == 403


def test_webhook_rejects_bad_signature(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings.github, "webhook_secret", "topsecret")
    resp = client.post("/webhooks/github", json={"action": "opened"})
    assert resp.status_code == 401


def test_webhook_fail_closed_without_secret(client: TestClient, monkeypatch):
    """未配置 secret 时默认拒绝未签名请求（fail-closed）。"""
    monkeypatch.setattr(settings.github, "webhook_secret", "")
    resp = client.post(
        "/webhooks/github", json={"action": "opened"}, headers={"X-GitHub-Event": "issues"}
    )
    assert resp.status_code == 401


def test_webhook_allows_unverified_when_explicitly_enabled(client: TestClient, monkeypatch):
    """显式开启 allow_unverified_webhooks 后才放行无签名请求（仅限联调）。"""
    monkeypatch.setattr(settings.github, "webhook_secret", "")
    monkeypatch.setattr(settings.github, "allow_unverified_webhooks", True)
    resp = client.post(
        "/webhooks/github",
        json={"action": "closed", "repository": {"full_name": "a/b"}},
        headers={"X-GitHub-Event": "pull_request"},
    )
    assert resp.status_code == 200
    assert resp.json()["accepted"] is False


def test_webhook_remains_reachable_with_auth_token(client: TestClient, monkeypatch):
    """开启 API 鉴权后 Webhook 仍可用（走自身签名校验，不被 Bearer 中间件 401）。"""
    monkeypatch.setattr(settings.api, "auth_token", "s3cret")
    monkeypatch.setattr(settings.github, "webhook_secret", "topsecret")
    resp = _post_webhook(
        client, {"action": "closed", "repository": {"full_name": "a/b"}}, "pull_request"
    )
    assert resp.status_code == 200


def test_webhook_ignores_unrelated_event(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings.github, "webhook_secret", "topsecret")
    resp = _post_webhook(
        client, {"action": "closed", "repository": {"full_name": "a/b"}}, "pull_request"
    )
    assert resp.status_code == 200
    assert resp.json()["accepted"] is False


def test_webhook_dispatches_issue(client: TestClient, monkeypatch, tmp_git_repo: Path):
    monkeypatch.setattr(settings.github, "webhook_secret", "topsecret")
    monkeypatch.setattr(settings.github, "repo_root", str(tmp_git_repo.parent))
    monkeypatch.setattr(settings.github, "auto_pr", False)

    async def _noop_execute(record) -> None:  # 避免测试中真正启动工作流
        record.status = "PENDING"

    monkeypatch.setattr("src.api.routers.webhook_router._execute", _noop_execute)

    payload = {
        "action": "opened",
        "repository": {"full_name": f"acme/{tmp_git_repo.name}"},
        "issue": {"title": "add() 返回错误", "body": "add(1,2) 应该等于 3"},
    }
    resp = _post_webhook(client, payload, "issues")
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True
    assert body["task_id"]


def test_webhook_unknown_local_repo_rejected(client: TestClient, monkeypatch):
    """repo_root 里找不到对应仓库时不得回退 CWD 猜测。"""
    monkeypatch.setattr(settings.github, "webhook_secret", "topsecret")
    monkeypatch.setattr(settings.github, "repo_root", "")
    payload = {
        "action": "opened",
        "repository": {"full_name": "acme/not-mapped-anywhere"},
        "issue": {"title": "t", "body": "b"},
    }
    resp = _post_webhook(client, payload, "issues")
    assert resp.status_code == 200
    assert resp.json()["accepted"] is False


def test_webhook_dispatches_pr_review_comment(client: TestClient, monkeypatch, tmp_git_repo: Path):
    """PR Review 评论中的 /nexus fix 指令应触发修复任务。"""
    monkeypatch.setattr(settings.github, "webhook_secret", "topsecret")
    monkeypatch.setattr(settings.github, "repo_root", str(tmp_git_repo.parent))
    monkeypatch.setattr(settings.github, "auto_pr", False)

    async def _noop_execute(record) -> None:
        record.status = "PENDING"

    monkeypatch.setattr("src.api.routers.webhook_router._execute", _noop_execute)

    payload = {
        "action": "created",
        "repository": {"full_name": f"acme/{tmp_git_repo.name}"},
        "pull_request": {"number": 7, "title": "feat: calc", "body": "adds calculator"},
        "comment": {"body": "/nexus fix add() 仍然是错的"},
    }
    resp = _post_webhook(client, payload, "pull_request_review_comment")
    assert resp.status_code == 200
    assert resp.json()["accepted"] is True

    # 无触发指令的 Review 评论不派发
    payload["comment"] = {"body": "LGTM"}
    resp2 = _post_webhook(client, payload, "pull_request_review_comment")
    assert resp2.json()["accepted"] is False


def test_webhook_dispatches_changes_requested_review(
    client: TestClient, monkeypatch, tmp_git_repo: Path
):
    """Review 判为 changes_requested 时应自动触发修复（README 承诺的闭环）。"""
    monkeypatch.setattr(settings.github, "webhook_secret", "topsecret")
    monkeypatch.setattr(settings.github, "repo_root", str(tmp_git_repo.parent))
    monkeypatch.setattr(settings.github, "auto_pr", False)

    async def _noop_execute(record) -> None:
        record.status = "PENDING"

    monkeypatch.setattr("src.api.routers.webhook_router._execute", _noop_execute)

    payload = {
        "action": "submitted",
        "repository": {"full_name": f"acme/{tmp_git_repo.name}"},
        "review": {"state": "changes_requested", "body": "add() 的返回值不对"},
        "pull_request": {"number": 7, "title": "feat: calc", "body": "adds calculator"},
    }
    resp = _post_webhook(client, payload, "pull_request_review")
    assert resp.status_code == 200
    assert resp.json()["accepted"] is True

    # 普通评论（approved 且无指令）不触发
    payload["review"] = {"state": "approved", "body": "LGTM"}
    resp2 = _post_webhook(client, payload, "pull_request_review")
    assert resp2.json()["accepted"] is False


async def test_task_lifecycle_end_to_end(tmp_git_repo: Path):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/tasks",
            json={
                "repo_path": str(tmp_git_repo),
                "issue_description": (
                    "test_add 失败：AssertionError: assert -1 == 3\n"
                    '  File "tests/test_calculator.py", line 4, in test_add\n'
                    "    assert add(1, 2) == 3\n"
                ),
            },
        )
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]

        status = "PENDING"
        deadline = time.monotonic() + 90
        payload: dict = {}
        while time.monotonic() < deadline:
            r = await client.get(f"/api/v1/tasks/{task_id}")
            payload = r.json()
            status = payload["status"]
            if status in {"SUCCESS", "HUMAN_ESCALATION", "ERROR"}:
                break
            await asyncio.sleep(0.5)

        assert status == "SUCCESS", f"unexpected status={status} payload={payload}"
        assert "return a + b" in payload["final_diff"]
        assert payload["patch_history"]
        listed = (await client.get("/api/v1/tasks")).json()
        assert any(t["task_id"] == task_id for t in listed)

        # Time-Travel：Checkpoint 历史可按线程回溯
        ckpts = (await client.get(f"/api/v1/tasks/{task_id}/checkpoints")).json()
        assert ckpts, "应有 Checkpoint 历史"
        assert {"SUCCESS"} & {c["status"] for c in ckpts}
        assert all(c["checkpoint_id"] and c["thread_id"] == task_id for c in ckpts)

        # 不触发续跑的人工反馈：仅记录
        intervention = (
            await client.post(
                f"/api/v1/tasks/{task_id}/intervention",
                json={"human_feedback": "请补充边界用例", "resume": False},
            )
        ).json()
        assert "未触发续跑" in intervention["message"]


# --------------------------------------------------------------------------- #
# 生产环境安全基线（fail-fast）
# --------------------------------------------------------------------------- #


def test_prod_config_requires_auth_token_and_repo_roots(monkeypatch):
    """env=prod 时缺少 token / 白名单 / 沙箱隔离必须启动失败，而不是只打 warning。"""
    from src.api.main import validate_production_config

    monkeypatch.setattr(settings, "env", "prod")
    monkeypatch.setattr(settings.api, "auth_token", "")
    monkeypatch.setattr(settings.api, "allowed_repo_roots", "")
    with pytest.raises(RuntimeError, match="AUTH_TOKEN"):
        validate_production_config()

    monkeypatch.setattr(settings.api, "auth_token", "s3cret")
    with pytest.raises(RuntimeError, match="ALLOWED_REPO_ROOTS"):
        validate_production_config()

    monkeypatch.setattr(settings.api, "allowed_repo_roots", str(Path.cwd()))
    with pytest.raises(RuntimeError, match="沙箱"):
        validate_production_config()  # local 沙箱默认不满足生产隔离要求

    monkeypatch.setattr(settings.sandbox, "backend", "docker")
    validate_production_config()  # 配齐（docker 沙箱）后通过

    # local 后端 + 显式确认开关同样放行（风险自负）
    monkeypatch.setattr(settings.sandbox, "backend", "local")
    monkeypatch.setattr(settings.sandbox, "allow_local_in_prod", True)
    validate_production_config()


def test_dev_config_skips_production_validation(monkeypatch):
    from src.api.main import validate_production_config

    monkeypatch.setattr(settings, "env", "dev")
    monkeypatch.setattr(settings.api, "auth_token", "")
    monkeypatch.setattr(settings.api, "allowed_repo_roots", "")
    validate_production_config()  # 开发环境不设门槛


def test_prod_startup_ok_with_security_config(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "env", "prod")
    monkeypatch.setattr(settings.api, "auth_token", "s3cret")
    monkeypatch.setattr(settings.api, "allowed_repo_roots", str(tmp_path))
    # 显式确认 local 沙箱（避免测试环境真的去拉起 docker 容器）
    monkeypatch.setattr(settings.sandbox, "allow_local_in_prod", True)
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200


def test_prod_startup_fails_fast_via_lifespan(monkeypatch):
    """校验挂在 lifespan 上：生产配置缺失时 TestClient 启动即抛错。"""
    monkeypatch.setattr(settings, "env", "prod")
    monkeypatch.setattr(settings.api, "auth_token", "")
    monkeypatch.setattr(settings.api, "allowed_repo_roots", "")
    with pytest.raises(RuntimeError, match="AUTH_TOKEN"):
        with TestClient(app):
            pass


# --------------------------------------------------------------------------- #
# 资源生命周期与任务存储上限
# --------------------------------------------------------------------------- #


async def test_execute_shared_pool_closes_llm_only(monkeypatch, tmp_git_repo):
    """共享池注入时：容器池不关（归应用生命周期），LLM 客户端必须随任务关闭。"""
    from src.api.routers import task_router as tr

    captured: dict[str, object] = {}

    class _FakeWorkflow:
        def __init__(self, deps, **kwargs):
            captured.update(kwargs)
            self.deps = deps

        async def astream(self, *args, **kwargs):
            if False:  # pragma: no cover - 空异步生成器
                yield {}

    monkeypatch.setattr(tr, "NexusWorkflow", _FakeWorkflow)
    monkeypatch.setattr(tr, "_resolve_checkpointer", lambda: None)
    monkeypatch.setattr(tr, "_shared_pool", object())

    record = tr.TaskRecord(task_id="t-flags", repo_path=str(tmp_git_repo), issue_description="x")
    await tr._execute(record)
    assert captured["close_pool"] is False
    assert captured["close_llm"] is True

    # 无共享池（任务私有池）时两者都要关
    captured.clear()
    monkeypatch.setattr(tr, "_shared_pool", None)
    record2 = tr.TaskRecord(task_id="t-flags-2", repo_path=str(tmp_git_repo), issue_description="x")
    await tr._execute(record2)
    assert captured["close_pool"] is True
    assert captured["close_llm"] is True


async def test_task_level_timeout_marks_error(monkeypatch, tmp_git_repo):
    """卡死的工作流必须被 NEXUS_API__REQUEST_TIMEOUT_S 任务级超时中止为 ERROR。"""
    from src.api.routers import task_router as tr

    class _HangingWorkflow:
        def __init__(self, deps, **kwargs):
            pass

        async def astream(self, *args, **kwargs):
            await asyncio.sleep(3600)
            yield {}  # pragma: no cover - 不会走到

    monkeypatch.setattr(tr, "NexusWorkflow", _HangingWorkflow)
    monkeypatch.setattr(tr, "_resolve_checkpointer", lambda: None)
    monkeypatch.setattr(settings.api, "request_timeout_s", 0.1)

    record = tr.TaskRecord(
        task_id="t-timeout", repo_path=str(tmp_git_repo), issue_description="x"
    )
    await tr._execute(record)

    assert record.status == "ERROR"
    assert "REQUEST_TIMEOUT_S=0.1" in record.state["termination_reason"]


async def test_task_store_enforces_capacity():
    """插入后稳态容量恰为 max_tasks；淘汰最旧的终态任务。"""
    from src.api.routers.task_router import TaskStore

    store = TaskStore(max_tasks=3, max_events=10)
    created: list[str] = []
    for i in range(3):
        rec = await store.create(f"repo{i}", f"issue{i}")
        rec.status = "SUCCESS"
        created.append(rec.task_id)
    newest = await store.create("repo-new", "issue-new")
    assert store.size == 3
    assert store.get(created[0]) is None  # 最旧的终态任务被淘汰
    assert store.get(newest.task_id) is newest
    assert created[1] in {r.task_id for r in store.list()}


async def test_task_store_keeps_running_tasks():
    """全部任务运行中时不淘汰（软上限），不允许杀掉在跑任务。"""
    from src.api.routers.task_router import TaskStore

    store = TaskStore(max_tasks=2, max_events=10)
    first = await store.create("a", "x")
    await store.create("b", "y")
    await store.create("c", "z")
    assert store.size == 3
    assert store.get(first.task_id) is not None


async def test_task_store_rejects_duplicate_task_id():
    """重复 task_id 必须拒绝（覆盖会丢失原任务的事件与终态）。"""
    from src.api.routers.task_router import TaskStore

    store = TaskStore(max_tasks=3, max_events=10)
    await store.create("a", "x", task_id="dup")
    with pytest.raises(ValueError, match="已存在"):
        await store.create("b", "y", task_id="dup")


def test_submit_task_duplicate_id_returns_409(monkeypatch, tmp_git_repo: Path):
    from src.api.routers import task_router as tr

    async def _noop_execute(record) -> None:  # pragma: no cover - 不真正执行
        record.status = "PENDING"

    monkeypatch.setattr(tr, "_execute", _noop_execute)
    with TestClient(app) as client:
        payload = {
            "repo_path": str(tmp_git_repo),
            "issue_description": "x",
            "task_id": "dup-1",
        }
        assert client.post("/api/v1/tasks", json=payload).status_code == 200
        resp = client.post("/api/v1/tasks", json=payload)
        assert resp.status_code == 409
        assert "dup-1" in resp.json()["detail"]


async def test_publish_after_swallows_and_logs_errors(monkeypatch):
    """后台 PR 发布失败必须整体兜底（否则只剩 unretrieved task exception）。"""
    from types import SimpleNamespace

    from src.api.routers import webhook_router as wr

    async def _boom(*args, **kwargs):
        raise RuntimeError("git apply failed: patch does not apply")

    monkeypatch.setattr(wr, "publish_patch", _boom)
    record = SimpleNamespace(
        status="SUCCESS",
        handle=None,
        state={"final_diff": "diff --git a/calculator.py b/calculator.py"},
        issue_description="fix add()",
        task_id="t-pub",
    )
    await wr._publish_after("acme/repo", ".", record, None)  # 不应抛异常


async def test_sse_stream_delivers_done_event(tmp_git_repo: Path):
    """SSE 必须以携带 final_diff 的 done 事件收尾。

    回归背景：``record.status`` 在 critic 节点返回时先置为终态，而 done 事件
    要等 Worktree 清理（``git worktree remove/prune``）之后才发布；旧实现以
    「终态 status + 队列空」作为收流条件，会在这段窗口内提前收流，客户端
    永远拿不到 done（实测稳定复现）。
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=120) as c:
        resp = await c.post(
            "/api/v1/tasks",
            json={
                "repo_path": str(tmp_git_repo),
                "issue_description": (
                    "test_add 失败：AssertionError: assert -1 == 3\n"
                    '  File "tests/test_calculator.py", line 4, in test_add\n'
                    "    assert add(1, 2) == 3\n"
                ),
            },
        )
        task_id = resp.json()["task_id"]

        events: list[dict] = []
        async with c.stream("GET", f"/api/v1/tasks/{task_id}/events") as stream:
            assert stream.status_code == 200
            async for line in stream.aiter_lines():
                if line.startswith("data:"):
                    payload = json.loads(line[5:])
                    events.append(payload)
                    if payload.get("type") == "done":
                        break

    assert events[-1].get("type") == "done", (
        f"未收到 done 事件，事件序列={[e.get('type') for e in events]}"
    )
    assert "return a + b" in events[-1]["final_diff"]
    assert events[-1]["status"] == "SUCCESS"


async def test_background_publish_task_keeps_strong_reference():
    """后台 PR 发布任务必须被强引用持有，否则可能执行中途被 GC 回收。"""
    from src.api.routers import webhook_router as wr

    started = asyncio.Event()
    gate = asyncio.Event()

    async def work() -> None:
        started.set()
        await gate.wait()

    task = wr._spawn_background(work())
    await started.wait()
    assert task in wr._background_tasks  # 执行期间必须持有引用

    gate.set()
    await task
    await asyncio.sleep(0.05)  # 等 done callback 回收引用
    assert task not in wr._background_tasks


async def test_parallel_tasks_are_isolated(tmp_path: Path):
    """并发任务必须彼此隔离：各自独立 Worktree / 沙箱，源仓库零污染。

    对应 README 的「高并发安全沙箱」主张：多个任务同时在跑时，不能出现
    补丁串仓库、源仓库被改动或 worktree 残留。
    """
    from conftest import CALCULATOR_BUGGY, TEST_CALCULATOR, init_git_repo

    repos: list[Path] = []
    for name in ("parallel-a", "parallel-b"):
        repo = tmp_path / name
        repo.mkdir()
        (repo / "calculator.py").write_text(CALCULATOR_BUGGY, encoding="utf-8")
        (repo / "conftest.py").write_text('"""test path bootstrap"""\n', encoding="utf-8")
        (repo / "tests").mkdir()
        (repo / "tests" / "test_calculator.py").write_text(TEST_CALCULATOR, encoding="utf-8")
        repos.append(init_git_repo(repo))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=180) as c:
        task_ids: list[str] = []
        for repo in repos:
            submitted = await c.post(
                "/api/v1/tasks",
                json={
                    "repo_path": str(repo),
                    "issue_description": (
                        "test_add 失败：AssertionError: assert -1 == 3\n"
                        '  File "tests/test_calculator.py", line 4, in test_add\n'
                        "    assert add(1, 2) == 3\n"
                    ),
                },
            )
            assert submitted.status_code == 200
            task_ids.append(submitted.json()["task_id"])

        states: list[dict] = []
        deadline = time.monotonic() + 150
        while time.monotonic() < deadline:
            states = [(await c.get(f"/api/v1/tasks/{tid}")).json() for tid in task_ids]
            if all(s["status"] in {"SUCCESS", "HUMAN_ESCALATION", "ERROR"} for s in states):
                break
            await asyncio.sleep(0.5)

    assert [s["status"] for s in states] == ["SUCCESS", "SUCCESS"], states
    for state in states:
        assert "return a + b" in state["final_diff"]

    # 修复只发生在各自的临时 Worktree 中，源仓库工作区必须保持干净
    for repo in repos:
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert dirty.stdout.strip() == "", f"源仓库被污染：{repo} -> {dirty.stdout}"
