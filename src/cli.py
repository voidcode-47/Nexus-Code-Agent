"""Nexus-Code-Agent 命令行入口。

用法::

    python -m src.cli heal   --repo examples/sample_repo --issue "..."
    python -m src.cli index  --repo examples/sample_repo
    python -m src.cli serve  --port 8000
    python -m src.cli evaluate --repo examples/sample_repo --limit 2
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from src.config import setup_logging
from src.config.settings import settings

app = typer.Typer(add_completion=False, help="Nexus-Code-Agent CLI")
console = Console()


@app.command()
def heal(
    repo: Path = typer.Option(..., "--repo", "-r", help="本地仓库路径"),
    issue: str = typer.Option(..., "--issue", "-i", help="Issue 描述 / 报错日志 / 自然语言指令"),
    max_iterations: int = typer.Option(0, "--max-iterations", "-m", help="最大迭代次数（0=使用默认）"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="输出调试日志"),
) -> None:
    """执行完整的「定位 → 规划 → 改码 → 校验」自愈闭环。"""
    setup_logging("DEBUG" if verbose else settings.log_level, json_output=False, force=True)
    from src.agent.workflow import NexusWorkflow, WorkflowDeps

    repo_path = repo.resolve()
    console.print(Panel.fit(f"[bold]Repo[/]\n{repo_path}\n\n[bold]Issue[/]\n{issue}", title="Nexus heal"))

    deps = WorkflowDeps.from_repo(repo_path)
    # 单次运行的依赖（LLM 连接池 / 容器池）随流程结束释放
    workflow = NexusWorkflow(deps, close_pool=True, close_llm=True)
    state = asyncio.run(
        workflow.run(issue, repo_path, max_iterations=max_iterations or None)
    )

    console.print(f"\n[bold cyan]状态[/]: {state.get('status')}")
    console.print(f"[bold cyan]说明[/]: {state.get('termination_reason')}")
    console.print(f"[bold cyan]迭代[/]: {state.get('current_iteration')} 次")
    console.print(f"[bold cyan]Token[/]: {state.get('token_usage')}")
    console.print(f"[bold cyan]可疑文件[/]: {state.get('suspected_files')}")
    plan = state.get("repair_plan", "")
    if plan:
        console.print(Panel(plan, title="修复计划", border_style="blue"))
    diff = state.get("final_diff", "")
    if diff:
        console.print(Panel(Syntax(diff, "diff", theme="monokai"), title="最终补丁", border_style="green"))
    else:
        test_output = state.get("test_output", "")
        console.print(Panel(test_output[:2000], title="最后一轮测试输出", border_style="red"))

    raise typer.Exit(code=0 if state.get("status") == "SUCCESS" else 1)


@app.command()
def index(
    repo: Path = typer.Option(..., "--repo", "-r", help="本地仓库路径"),
    query: str = typer.Option("", "--query", "-q", help="可选：检索查询以验证索引质量"),
) -> None:
    """构建代码图谱与混合索引，并打印核心度排行。"""
    setup_logging(settings.log_level, json_output=False, force=True)
    from src.indexer.call_graph import CallGraph
    from src.indexer.hybrid_retriever import HybridRetriever
    from src.indexer.symbol_table import build_symbol_table

    repo_path = repo.resolve()
    table = build_symbol_table(repo_path)
    graph = CallGraph.build(table)
    # repo_root 作为稳定指纹后缀：增删文件不换集合名，不产生孤儿集合
    retriever = HybridRetriever(table=table, call_graph=graph, repo_root=str(repo_path))
    count = retriever.index_repo(table, graph)

    console.print(f"[bold]符号数[/]: {len(table)}   [bold]切片数[/]: {count}   调用图节点: {graph.graph.number_of_nodes()}")
    console.print("\n[bold]PageRank 核心度 Top 10[/]")
    for name, score in graph.top_core(10):
        console.print(f"  {score:.5f}  {name}")

    if query:
        console.print(f"\n[bold]检索[/]: {query}")
        for hit in retriever.search(query, top_k=8):
            console.print(
                f"  {hit.score:.4f}  {hit.symbol}  ({hit.path}:{hit.start_line})  "
                f"dense#{hit.dense_rank} sparse#{hit.sparse_rank}"
            )


@app.command()
def evaluate(
    repo: Path | None = typer.Option(None, "--repo", "-r", help="本地仓库路径（离线评测）"),
    limit: int = typer.Option(2, "--limit", "-n", help="评测题目数量"),
    offline: Path | None = typer.Option(None, "--offline", help="本地 SWE-bench JSONL 路径"),
    workers: int = typer.Option(0, "--workers", "-w", help=">1 时启用多进程评测"),
    attempts: int = typer.Option(1, "--attempts", "-k", help="每道题最大尝试轮数（用于 Pass@k）"),
    report: Path = typer.Option(Path("eval_results/report.md"), "--report", help="Markdown 报表输出路径"),
) -> None:
    """在 SWE-bench Lite 子集（或离线 JSONL / 合成样例）上跑分并生成报表。"""
    setup_logging(settings.log_level, json_output=False, force=True)
    from src.evaluation import metrics_report, swe_bench_loader
    from src.evaluation.runner import EvaluationRunner

    instances = swe_bench_loader.load_swe_bench_lite(
        limit=limit, offline_path=offline
    )
    if repo is not None:
        for inst in instances:
            inst.local_repo = str(repo.resolve())

    runner = EvaluationRunner(concurrency=2, attempts=attempts)
    results = (
        runner.run_multiprocess(instances, workers=workers)
        if workers > 1
        else runner.run_sync(instances)
    )
    metrics = metrics_report.compute_metrics(results)
    console.print(metrics_report.render_console(metrics))
    report_path = metrics_report.save_report(
        metrics_report.render_markdown(metrics, results), report
    )
    console.print(f"\n[green]报表已生成[/]: {report_path}")


@app.command()
def serve(
    host: str = typer.Option(settings.api.host, help="监听地址"),
    port: int = typer.Option(settings.api.port, help="监听端口"),
    reload: bool = typer.Option(False, "--reload", help="开发热重载"),
) -> None:
    """启动异步服务网关。"""
    import uvicorn

    uvicorn.run("src.api.main:app", host=host, port=port, reload=reload)


@app.command()
def mcp(
    repo: Path = typer.Option(Path("."), "--repo", "-r", help="仓库根目录"),
) -> None:
    """以 stdio transport 启动 MCP 工具服务端。"""
    from src.mcp_tools.server import build_server

    build_server(repo.resolve()).run()


if __name__ == "__main__":
    app()
