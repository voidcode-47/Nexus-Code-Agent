"""状态、Token 计数器与熔断器测试。"""

from __future__ import annotations

from src.agent.repair_heuristics import generate_candidate_patches, is_test_only_path
from src.agent.state import (
    add_tokens,
    count_tokens,
    error_fingerprint,
    fold_patch_history,
    initial_state,
    record,
    register_failure,
    should_escalate,
    trim_context,
)

SOURCE = '''"""m."""


def add(a, b):
    return a - b
'''


def test_count_tokens_handles_mixed_text():
    assert count_tokens("") == 0
    assert count_tokens("hello world hello world") >= 1
    assert count_tokens("中文内容") >= 4


def test_initial_state_defaults():
    state = initial_state("issue", "/repo")
    assert state["task_id"].startswith("task-")
    assert state["status"] == "LOCATING"
    assert state["patch_history"] == []
    assert state["max_iterations"] >= 1


def test_add_tokens_accumulates():
    state = initial_state("issue", "/repo")
    total = add_tokens(state, "a" * 40)
    assert total >= 10
    state["token_usage"] = total
    assert add_tokens(state, "b" * 40) > total


def test_error_fingerprint_delegates():
    text = 'Traceback (most recent call last):\n  File "x.py", line 1, in f\nValueError: e\n'
    assert error_fingerprint(text) == error_fingerprint(text)
    assert len(error_fingerprint(text)) == 16


def test_record_contains_fingerprint():
    rec = record(1, diff_content="d", test_passed=False, error_text="ValueError: bad")
    assert rec["iteration"] == 1
    assert rec["error_fingerprint"]


def test_fold_patch_history_summarizes():
    history = [
        record(i, test_passed=(i == 5), error_summary=f"err {i}") for i in range(1, 6)
    ]
    recent, summary = fold_patch_history(history, keep_recent=2)
    assert len(recent) == 2
    assert "iter1" in summary


def test_trim_context_truncates():
    long_text = "x" * 100000
    trimmed = trim_context(long_text, budget_tokens=100)
    assert trimmed.endswith("(context trimmed)")


def test_should_escalate_on_iterations():
    state = initial_state("i", "/r")
    state["max_iterations"] = 3
    state["current_iteration"] = 3
    escalate, reason = should_escalate(state)
    assert escalate
    assert "最大重试" in reason


def test_should_escalate_on_repeated_error():
    state = initial_state("i", "/r")
    state["same_error_streak"] = 99
    escalate, _ = should_escalate(state)
    assert escalate


def test_register_failure_streaks():
    state = initial_state("i", "/r")
    update = register_failure(state, "fp1", stalled=False)
    assert update["same_error_streak"] == 1
    state.update(update)
    update2 = register_failure(state, "fp1", stalled=True)
    assert update2["same_error_streak"] == 2
    assert update2["stall_streak"] == 1
    update3 = register_failure(state, "fp2", stalled=False)
    assert update3["same_error_streak"] == 1
    assert update3["stall_streak"] == 0


def test_candidates_prioritize_binop():
    candidates = generate_candidate_patches(SOURCE)
    assert candidates
    assert candidates[0].strategy.startswith("binop")
    assert "+" in candidates[0].replace


def test_candidate_patch_format():
    candidates = generate_candidate_patches(SOURCE)
    patch = candidates[0].to_patch()
    assert "<<<<<<< SEARCH" in patch
    assert ">>>>>>> REPLACE" in patch


def test_is_test_only_path():
    assert is_test_only_path("tests/test_x.py")
    assert is_test_only_path("pkg/test_util.py")
    assert not is_test_only_path("pkg/util.py")
