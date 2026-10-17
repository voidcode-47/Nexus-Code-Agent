"""共享测试夹具：临时 Git 仓库、示例代码与配置隔离。"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

CALCULATOR_BUGGY = '''"""示例计算器（加法被误写为减法）。"""

from __future__ import annotations


def add(a: int, b: int) -> int:
    """返回两数之和。"""
    return a - b


def subtract(a: int, b: int) -> int:
    """返回两数之差。"""
    return a - b


def multiply(a: int, b: int) -> int:
    """返回两数之积。"""
    return a * b
'''

TEST_CALCULATOR = """from calculator import add, multiply, subtract


def test_add():
    assert add(1, 2) == 3


def test_subtract():
    assert subtract(5, 3) == 2


def test_multiply():
    assert multiply(3, 4) == 12
"""


def _run_git(repo: Path, *args: str) -> None:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_AUTHOR_DATE": "2020-01-01T00:00:00",
           "GIT_COMMITTER_DATE": "2020-01-01T00:00:00"}
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env=env,
    )


def init_git_repo(repo: Path) -> Path:
    """初始化一个带一次提交的 Git 仓库。"""
    _run_git(repo, "init")
    commit_all(repo, "baseline")
    return repo


def commit_all(repo: Path, message: str = "update") -> None:
    """提交工作区全部改动。"""
    _run_git(repo, "add", "-A")
    _run_git(
        repo,
        "-c", "user.email=test@local",
        "-c", "user.name=test",
        "commit", "-m", message,
    )


@pytest.fixture
def git_commit():
    """返回一个提交当前工作区改动的辅助函数。"""
    return commit_all


@pytest.fixture
def tmp_git_repo(tmp_path: Path) -> Path:
    """包含一个加法缺陷的临时仓库。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calculator.py").write_text(CALCULATOR_BUGGY, encoding="utf-8")
    (repo / "conftest.py").write_text('"""test path bootstrap"""\n', encoding="utf-8")
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_calculator.py").write_text(TEST_CALCULATOR, encoding="utf-8")
    return init_git_repo(repo)


@pytest.fixture
def worktree_root(tmp_path: Path) -> Path:
    root = tmp_path / "worktrees"
    root.mkdir()
    return root


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch, tmp_path):
    """确保测试不读取外部 .env/环境，并强制离线模式。

    注意：``src.config.settings.settings`` 是导入期由 lru_cache 固化的单例，
    只做 monkeypatch.setenv 对它无效——必须直接改写单例字段，否则开发者本机
    ``.env`` 里配置了 ``NEXUS_SANDBOX__BACKEND=docker`` /
    ``NEXUS_RETRIEVAL__USE_QDRANT=true`` 时，测试会意外打到真实 Docker / Qdrant。
    """
    monkeypatch.setenv("NEXUS_ENV", "test")
    monkeypatch.setenv("NEXUS_LLM__PROVIDER", "heuristic")
    monkeypatch.setenv("NEXUS_LLM__API_KEY", "")
    monkeypatch.setenv("NEXUS_SANDBOX__BACKEND", "local")
    monkeypatch.setenv("NEXUS_RETRIEVAL__USE_QDRANT", "false")
    monkeypatch.setenv("NEXUS_POSTGRES__ENABLED", "false")
    monkeypatch.setenv("NEXUS_LOG_JSON", "false")

    from src.config.settings import settings

    monkeypatch.setattr(settings, "env", "test")
    monkeypatch.setattr(settings, "log_json", False)
    monkeypatch.setattr(settings.llm, "provider", "heuristic")
    monkeypatch.setattr(settings.llm, "api_key", "")
    monkeypatch.setattr(settings.sandbox, "backend", "local")
    monkeypatch.setattr(settings.sandbox, "allow_local_in_prod", False)
    monkeypatch.setattr(settings.sandbox, "worktree_root", str(tmp_path / "default-worktrees"))
    monkeypatch.setattr(settings.retrieval, "use_qdrant", False)
    monkeypatch.setattr(settings.postgres, "enabled", False)
    monkeypatch.setattr(settings.github, "token", "")
    monkeypatch.setattr(settings.github, "webhook_secret", "")
    monkeypatch.setattr(settings.github, "allow_unverified_webhooks", False)
    monkeypatch.setattr(settings.github, "auto_pr", False)
    monkeypatch.setattr(settings.github, "repo_root", "")
    monkeypatch.setattr(settings.api, "auth_token", "")
    monkeypatch.setattr(settings.api, "allowed_repo_roots", "")
    yield
