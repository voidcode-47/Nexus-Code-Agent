"""评测平台测试。"""

from __future__ import annotations

import json

import pytest

from src.agent.workflow import NexusWorkflow, WorkflowDeps
from src.evaluation.metrics_report import (
    compute_metrics,
    pass_at_k,
    render_console,
    render_markdown,
    save_report,
)
from src.evaluation.runner import EvalResult, EvaluationRunner
from src.evaluation.swe_bench_loader import (
    SweBenchInstance,
    dump_jsonl,
    load_jsonl,
    load_swe_bench_lite,
)


def _results():
    return [
        EvalResult(instance_id="a", resolved=True, status="SUCCESS", iterations=1, tokens=1000, duration_s=12.0),
        EvalResult(instance_id="b", resolved=False, status="HUMAN_ESCALATION", iterations=3, tokens=5000, duration_s=200.0, lsp_blocked=True),
        EvalResult(instance_id="c", resolved=True, status="SUCCESS", iterations=2, tokens=2000, duration_s=45.0),
    ]


def test_compute_metrics():
    metrics = compute_metrics(_results())
    assert metrics.total == 3
    assert metrics.resolved == 2
    assert abs(metrics.pass_at_1 - 2 / 3) < 1e-6
    assert metrics.total_tokens == 8000
    assert metrics.lsp_block_rate > 0
    assert set(metrics.dimensions) == {"修复率", "效率", "成本控制", "稳定性", "静态拦截收益"}


def test_compute_metrics_pass_at_k_curve():
    """多轮尝试结果应生成 Pass@k 曲线。"""
    results = [
        EvalResult(
            instance_id="a",
            resolved=True,
            status="SUCCESS",
            attempt_outcomes=[False, True],
            attempts_made=2,
        ),
        EvalResult(
            instance_id="b",
            resolved=False,
            status="HUMAN_ESCALATION",
            attempt_outcomes=[False, False],
            attempts_made=2,
        ),
    ]
    metrics = compute_metrics(results)
    assert metrics.pass_k == 2
    assert metrics.pass_at_k_curve["Pass@1"] == 0.0
    assert metrics.pass_at_k_curve["Pass@2"] == 0.5
    assert metrics.pass_at_k == 0.5
    assert metrics.avg_attempts == 2.0


def test_metrics_to_dict_roundtrip():
    data = compute_metrics(_results()).to_dict()
    assert isinstance(data["pass@1"], float)
    assert "pass@k" in data
    json.dumps(data)  # 必须可 JSON 序列化


def test_render_markdown_contains_sections():
    metrics = compute_metrics(_results())
    md = render_markdown(metrics, _results(), title="T")
    assert "# T" in md
    assert "Pass@1" in md
    assert "多维雷达图" in md
    assert "radar-beta" in md  # Mermaid 雷达图
    assert "耗时分布" in md
    assert "逐题明细" in md


def test_render_markdown_pass_at_k():
    results = [
        EvalResult(instance_id="a", resolved=True, status="SUCCESS",
                   attempt_outcomes=[False, True], attempts_made=2),
    ]
    md = render_markdown(compute_metrics(results), results, title="T")
    assert "Pass@2" in md
    assert "Pass@k 曲线" in md


def test_render_console_and_save(tmp_path):
    metrics = compute_metrics(_results())
    assert "Pass@1" in render_console(metrics)
    target = save_report("hello", tmp_path / "out" / "r.md")
    assert target.read_text(encoding="utf-8") == "hello"


def test_pass_at_k():
    attempts = {"a": [False, True], "b": [False, False]}
    assert pass_at_k(attempts, 1) == 0.0
    assert pass_at_k(attempts, 2) == 0.5


# --------------------------------------------------------------------------- #
# loader
# --------------------------------------------------------------------------- #


def test_synthetic_loader_limit():
    instances = load_swe_bench_lite(limit=1, allow_online=False, allow_synthetic=True)
    assert len(instances) == 1
    assert instances[0].instance_id


def test_jsonl_roundtrip(tmp_path):
    inst = SweBenchInstance(
        instance_id="x1",
        repo="r",
        base_commit="abc",
        problem_statement="fix",
        FAIL_TO_PASS=["tests/test_a.py::test_b"],
    )
    path = tmp_path / "inst.jsonl"
    dump_jsonl([inst], path)
    loaded = load_jsonl(path)
    assert loaded[0].instance_id == "x1"
    assert loaded[0].fail_to_pass_tests == ["tests/test_a.py::test_b"]


def test_loader_from_offline_path(tmp_path):
    inst = SweBenchInstance(instance_id="off1", repo="r", base_commit="c", problem_statement="p")
    path = tmp_path / "d.jsonl"
    dump_jsonl([inst], path)
    got = load_swe_bench_lite(offline_path=path, allow_online=False)
    assert got[0].instance_id == "off1"


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #


async def test_runner_solves_offline(tmp_git_repo, worktree_root):
    instance = SweBenchInstance(
        instance_id="nexus__local-1",
        repo="local/sample",
        base_commit="0" * 40,
        problem_statement=(
            "test_add 失败：AssertionError: assert -1 == 3\n"
            '  File "tests/test_calculator.py", line 4, in test_add\n'
            "    assert add(1, 2) == 3\n"
        ),
        FAIL_TO_PASS=["tests/test_calculator.py::test_add"],
        local_repo=str(tmp_git_repo),
    )

    def factory(repo: str) -> NexusWorkflow:
        deps = WorkflowDeps.from_repo(repo)
        return NexusWorkflow(deps, worktree_root=worktree_root)

    runner = EvaluationRunner(concurrency=1, workflow_factory=factory)
    results = await runner.run([instance])
    assert len(results) == 1
    assert results[0].resolved
    assert results[0].predicted_patch is not None
    assert results[0].duration_s > 0


async def test_runner_attempts_early_stop(tmp_git_repo, worktree_root):
    """attempts=2 时应记录尝试结果，且成功后早停。"""
    instance = SweBenchInstance(
        instance_id="nexus__local-2",
        repo="local/sample",
        base_commit="0" * 40,
        problem_statement=(
            "test_add 失败：AssertionError: assert -1 == 3\n"
            '  File "tests/test_calculator.py", line 4, in test_add\n'
            "    assert add(1, 2) == 3\n"
        ),
        FAIL_TO_PASS=["tests/test_calculator.py::test_add"],
        local_repo=str(tmp_git_repo),
    )

    def factory(repo: str) -> NexusWorkflow:
        deps = WorkflowDeps.from_repo(repo)
        return NexusWorkflow(deps, worktree_root=worktree_root)

    runner = EvaluationRunner(concurrency=1, attempts=2, workflow_factory=factory)
    results = await runner.run([instance])
    result = results[0]
    assert result.resolved
    assert result.attempt_outcomes == [True]  # 首轮成功 → 早停
    assert result.attempts_made == 1


async def test_runner_skips_without_repo():
    instance = SweBenchInstance(
        instance_id="no-repo", repo="x/y", base_commit="c", problem_statement="p"
    )
    runner = EvaluationRunner()
    results = await runner.run([instance])
    assert results[0].status == "SKIPPED"


def test_run_multiprocess_rejects_unpicklable_factory(tmp_git_repo):
    """闭包工厂无法跨进程传递时必须显式报错，而非静默退化到默认工作流。"""
    instance = SweBenchInstance(
        instance_id="nexus__local-mp",
        repo="local/sample",
        base_commit="0" * 40,
        problem_statement="x",
        local_repo=str(tmp_git_repo),
    )

    def closure_factory(_: str) -> NexusWorkflow:  # 局部闭包：不可 pickle
        raise AssertionError("多进程测试不应真正调用该工厂")

    runner = EvaluationRunner(workflow_factory=closure_factory)
    with pytest.raises(ValueError, match="无法序列化"):
        runner.run_multiprocess([instance], workers=2)
