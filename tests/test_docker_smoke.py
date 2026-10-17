"""docker 沙箱后端冒烟测试（真实容器内执行 pytest）。

回归护栏：``run_pytest`` 曾无条件使用宿主 ``sys.executable``，docker 后端把
宿主路径（如 ``C:\\...python.exe``）交给容器 exec 会直接失败——在
「API 跑在宿主 + 沙箱用 docker」这种最典型的组合下 docker 沙箱完全不可用，
且没有任何测试能拦住。本测试真实执行容器内 pytest，端到端覆盖该路径。

本地无 Docker daemon / 无镜像时自动跳过；CI 先构建镜像再运行本文件::

    docker build -f Dockerfile.sandbox -t nexus-sandbox:latest .
    python -m pytest tests/test_docker_smoke.py
"""

from __future__ import annotations

import pytest

from src.config.settings import settings
from src.sandbox.pool_manager import PoolManager


def _docker_image_ready() -> bool:
    if not PoolManager.docker_available():
        return False
    import docker

    try:
        docker.from_env().images.get(settings.sandbox.image)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _docker_image_ready(),
    reason=f"docker 不可用或缺少镜像 {settings.sandbox.image}（先 make sandbox-image）",
)


async def test_docker_backend_runs_pytest_in_container(tmp_path, monkeypatch):
    """docker 后端 run_pytest 必须能在容器内执行（解释器为镜像内 python）。"""
    monkeypatch.setattr(settings.sandbox, "backend", "docker")
    monkeypatch.setattr(settings.sandbox, "worktree_root", str(tmp_path))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "test_ok.py").write_text(
        "def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8"
    )

    pool = PoolManager()
    if pool.backend != "docker":  # pragma: no cover - daemon 中途不可用时优雅降级
        pytest.skip("docker 后端不可用（已降级 local）")
    await pool.warm_up()
    try:
        async with pool.session(repo) as session:
            result = await session.run_pytest(["test_ok.py"])
    finally:
        await pool.aclose()

    assert result.ok, result.stderr or result.stdout
    assert "1 passed" in result.stdout
