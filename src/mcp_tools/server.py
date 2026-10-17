"""FastMCP 异步服务端：把平台核心能力暴露为标准 MCP 工具。

工具矩阵：

* ``apply_patch``    —— 模糊匹配补丁应用（Search-Replace / Unified Diff）；
* ``read_file``      —— 读取仓库文件（限定在仓库根目录内，防目录穿越）；
* ``list_files``     —— 列出仓库源码文件（AST 可解析文件）；
* ``static_check``   —— Pyright / 语法静态检查；
* ``run_tests``      —— 在本地子进程沙箱中执行 pytest；
* ``search_code``    —— 基于符号表 / BM25 的代码检索；
* ``create_pull_request`` —— 把补丁发布为远端分支并创建 Pull Request（无人值守闭环）。

既可作为库调用（:func:`build_server`），也可通过 ``python -m src.mcp_tools.server``
以 stdio transport 启动，供 Claude Desktop / 任意 MCP Host 连接。
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from src.config import get_logger, setup_logging
from src.config.settings import settings
from src.mcp_tools.diff_patcher import apply_patch_to_file

logger = get_logger(__name__)


def _load_fastmcp() -> Any:
    """运行时解析 FastMCP 类：独立 ``fastmcp`` 包优先，官方 SDK 内置实现兜底。

    改为运行时解析的原因：``mcp`` 2.x 已移除 ``mcp.server.fastmcp``
    （FastMCP 更名 MCPServer），静态 import 在该版本下必然失败；且静态 import
    同名符号会让 mypy 报 ``no-redef`` / ``attr-defined``，CI 的 quality 关卡
    直接变红。两者都不可用时返回 None，由 :func:`build_server` 给出明确报错。
    """
    for module_name in ("fastmcp", "mcp.server.fastmcp"):
        try:
            module = importlib.import_module(module_name)
        except Exception:  # pragma: no cover - 依赖缺失 / 版本不兼容
            continue
        candidate = getattr(module, "FastMCP", None)
        if candidate is not None:
            logger.info("MCP 服务端使用 FastMCP 实现", extra={"module": module_name})
            return candidate
    return None


FastMCP: Any = _load_fastmcp()


def _safe_path(root: Path, candidate: str) -> Path:
    target = (root / candidate).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise ValueError(f"path escapes repository root: {candidate}")
    return target


def build_server(root: str | Path = ".") -> Any:
    """构建 FastMCP 服务端并注册工具。"""
    if FastMCP is None:  # pragma: no cover
        raise RuntimeError("fastmcp 未安装，无法启动 MCP 服务端")
    repo_root = Path(root).resolve()
    mcp = FastMCP(f"{settings.app_name}-tools")

    @mcp.tool()
    def apply_patch(path: str, patch: str) -> dict[str, Any]:
        """把 Search-Replace / Unified Diff 补丁应用到仓库内文件。"""
        target = _safe_path(repo_root, path)
        outcome = apply_patch_to_file(target, patch)
        return {
            "applied": outcome.applied,
            "method": outcome.method,
            "detail": outcome.describe(),
        }

    @mcp.tool()
    def read_file(path: str) -> str:
        """读取仓库内文件内容。"""
        return _safe_path(repo_root, path).read_text(encoding="utf-8")

    @mcp.tool()
    def list_files(suffix: str = ".py") -> list[str]:
        """列出仓库内指定后缀的源码文件（相对路径）。"""
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        skip = {
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            ".nexus",
            "node_modules",
            "build",
            "dist",
        }
        found: list[str] = []
        for p in repo_root.rglob(f"*{suffix}"):
            if not p.is_file():
                continue
            rel = p.relative_to(repo_root)
            # 只按相对路径过滤：绝对路径含祖先目录，检出路径带 build/dist
            # 等名字时会把整个仓库判空
            if any(part in skip for part in rel.parts):
                continue
            found.append(rel.as_posix())
        return sorted(found)

    @mcp.tool()
    async def static_check(paths: list[str]) -> dict[str, Any]:
        """对指定文件执行静态检查。"""
        from src.mcp_tools.lsp_bridge import LSPBridge

        report = await LSPBridge().check(
            [_safe_path(repo_root, p) for p in paths], root=repo_root
        )
        return {
            "passed": report.passed,
            "engine": report.engine,
            "summary": report.summary(),
            "diagnostics": [d.render() for d in report.diagnostics],
        }

    @mcp.tool()
    async def run_tests(targets: list[str] | None = None) -> dict[str, Any]:
        """在本地子进程沙箱中运行 pytest。"""
        from src.sandbox.pool_manager import PoolManager

        pool = PoolManager()
        try:
            async with pool.session(repo_root) as session:
                result = await session.run_pytest(targets or ["tests"])
        finally:
            # 每次调用不回收池会导致 docker 后端容器只增不减
            await pool.aclose()
        return {
            "ok": result.ok,
            "returncode": result.returncode,
            "summary": result.summary(),
        }

    @mcp.tool()
    def search_code(query: str, top_k: int = 8) -> list[dict[str, Any]]:
        """基于符号表 + BM25 的代码检索。"""
        from src.indexer.call_graph import CallGraph
        from src.indexer.hybrid_retriever import HybridRetriever
        from src.indexer.symbol_table import build_symbol_table

        table = build_symbol_table(repo_root)
        graph = CallGraph.build(table)
        # repo_root 作为稳定指纹后缀：增删文件不换集合名，不产生孤儿集合
        retriever = HybridRetriever(table=table, call_graph=graph, repo_root=str(repo_root))
        retriever.index_repo(table, graph)
        return [
            {
                "symbol": h.symbol,
                "path": h.path,
                "lines": [h.start_line, h.end_line],
                "score": round(h.score, 4),
            }
            for h in retriever.search(query, top_k=top_k)
        ]

    @mcp.tool()
    async def create_pull_request(
        patch: str, title: str, body: str = "", base: str = ""
    ) -> dict[str, Any]:
        """把修复补丁发布为远端分支并创建 Pull Request（离线时逐级降级）。"""
        from src.mcp_tools.pr_publisher import publish_patch

        result = await publish_patch(repo_root, patch, title=title, body=body, base=base)
        return {
            "published": result.published,
            "pushed": result.pushed,
            "branch": result.branch,
            "base": result.base,
            "pr_url": result.pr_url,
            "detail": result.describe(),
        }

    return mcp


def main() -> None:  # pragma: no cover - 入口
    setup_logging(settings.log_level, json_output=False)
    server = build_server(Path.cwd())
    logger.info("启动 MCP 服务端 (stdio)")
    server.run()


if __name__ == "__main__":  # pragma: no cover
    main()
