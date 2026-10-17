"""Nexus-Code-Agent 受控缺陷基准驱动：平台评测栈端到端跑分。

流程：bugs.jsonl -> 逐题准备独立仓库副本(注入缺陷并提交) -> 捕获失败测试输出作为
problem_statement -> SweBenchInstance(SWE-bench 兼容 schema) -> EvaluationRunner 并发求解
-> EvalResult -> compute_metrics / render_markdown。

用法（在 Nexus-Code-Agent 项目根目录下）：
  python benchmarks/run_cerberus.py --bugs benchmarks/artifacts/cerberus-bugs.jsonl \
      --work <instances_dir> --pristine <cerberus 干净克隆> --attempts 2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path


def rm_tree(path: Path) -> None:
    """Windows 下 git 对象文件带只读属性，需先解除再删。"""
    def _onexc(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    shutil.rmtree(path, onexc=_onexc)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 项目根，保证 import src

from src.evaluation.metrics_report import compute_metrics, render_console, render_markdown  # noqa: E402
from src.evaluation.runner import EvaluationRunner  # noqa: E402
from src.evaluation.swe_bench_loader import SweBenchInstance  # noqa: E402

PRISTINE: Path | None = None  # 运行时注入：干净克隆路径


def run_git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def prep_instance(bug: dict, work: Path) -> tuple[Path, str]:
    """复制干净克隆 -> 注入单行缺陷 -> 提交 -> 捕获失败测试输出。返回 (repo_dir, statement)。"""
    inst_dir = work / bug["instance_id"]
    if inst_dir.exists():
        rm_tree(inst_dir)
    shutil.copytree(PRISTINE, inst_dir)
    target = inst_dir / bug["bug"]["file"]
    lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
    idx = bug["bug"]["line"] - 1
    assert bug["bug"]["find"] in lines[idx], f"{bug['instance_id']} 行内容不匹配"
    lines[idx] = lines[idx].replace(bug["bug"]["find"], bug["bug"]["replace"], 1)
    target.write_text("".join(lines), encoding="utf-8")
    run_git(inst_dir, "add", "-A")
    run_git(inst_dir, "commit", "-m", f"inject {bug['instance_id']}")

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short", "-rf", "-p", "no:cacheprovider",
         "cerberus/tests"],
        cwd=inst_dir, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120,
    )
    out = proc.stdout or ""
    failed_lines = [ln.strip() for ln in out.splitlines() if ln.startswith("FAILED")]
    assert_lines = [ln.strip() for ln in out.splitlines() if ln.lstrip().startswith("E ")]
    assert failed_lines, f"{bug['instance_id']} 注入后测试未失败？"
    statement = (
        "运行 pytest 后以下测试失败：\n" + "\n".join(f"- {x}" for x in failed_lines[:6]) +
        "\n\n关键断言输出：\n" + "\n".join(assert_lines[:8]) +
        "\n\n请定位缺陷并以最小改动修复，保持公开 API 与现有行为兼容。"
        "修复后上述失败测试必须通过，且不得引入其他回归。"
    )
    return inst_dir, statement


def main() -> None:
    global PRISTINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugs", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--pristine", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default="results.jsonl")
    args = ap.parse_args()

    PRISTINE = Path(args.pristine).resolve()
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    bugs = [json.loads(x) for x in Path(args.bugs).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        bugs = bugs[: args.limit]

    instances: list[SweBenchInstance] = []
    inst_meta: dict[str, dict] = {}
    for bug in bugs:
        inst_dir, statement = prep_instance(bug, work)
        instances.append(SweBenchInstance(
            instance_id=bug["instance_id"], repo=bug["repo"], base_commit="",
            problem_statement=statement, FAIL_TO_PASS=bug["FAIL_TO_PASS"],
            version=bug["version"], local_repo=str(inst_dir),
        ))
        inst_meta[bug["instance_id"]] = bug
    print(f"prepared {len(instances)} instances")

    runner = EvaluationRunner(concurrency=args.concurrency, attempts=args.attempts, timeout_s=args.timeout)
    t0 = time.monotonic()
    results = runner.run_sync(instances)
    wall = time.monotonic() - t0

    out_path = Path(args.out).resolve()
    with out_path.open("w", encoding="utf-8") as fh:
        for r in results:
            full = r.to_dict()
            full["predicted_patch"] = r.predicted_patch  # 报告单独保留补丁
            full["f2p"] = inst_meta[r.instance_id]["FAIL_TO_PASS"]
            full["bug_file"] = inst_meta[r.instance_id]["bug"]["file"]
            fh.write(json.dumps(full, ensure_ascii=False) + "\n")

    metrics = compute_metrics(results)
    print(render_console(metrics))
    print(f"\nwall={wall:.1f}s total_tokens={metrics.total_tokens} resolved={metrics.resolved}/{metrics.total}")
    md = render_markdown(metrics, results, title="Nexus-Code-Agent 受控缺陷基准")
    Path(out_path).with_suffix(".md").write_text(md, encoding="utf-8")
    print(f"results -> {out_path}")


if __name__ == "__main__":
    main()
