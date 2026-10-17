"""补丁引擎测试：Search-Replace、Unified Diff、模糊匹配与缩进自适应。"""

from __future__ import annotations

from src.mcp_tools.diff_patcher import (
    SearchReplaceBlock,
    apply_patch,
    apply_patch_to_file,
    apply_search_replace,
    build_unified_diff,
    count_changed_lines,
    is_search_replace,
    is_unified_diff,
    levenshtein,
    parse_search_replace,
    similarity,
)

ORIGINAL = """def add(a, b):
    return a - b


def mul(a, b):
    return a * b
"""


def _block(search: str, replace: str) -> SearchReplaceBlock:
    return SearchReplaceBlock(search=search, replace=replace)


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #


def test_levenshtein_basic():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("same", "same") == 0


def test_similarity_range():
    assert similarity("abc", "abc") == 1.0
    assert 0.0 <= similarity("abc", "xyz") <= 1.0


# --------------------------------------------------------------------------- #
# Search-Replace
# --------------------------------------------------------------------------- #


def test_parse_single_block():
    patch = """<<<<<<< SEARCH
    return a - b
=======
    return a + b
>>>>>>> REPLACE"""
    blocks = parse_search_replace(patch)
    assert len(blocks) == 1
    assert "a - b" in blocks[0].search
    assert "a + b" in blocks[0].replace


def test_parse_multiple_blocks():
    patch = """<<<<<<< SEARCH
one
=======
1
>>>>>>> REPLACE
<<<<<<< SEARCH
two
=======
2
>>>>>>> REPLACE"""
    blocks = parse_search_replace(patch)
    assert len(blocks) == 2


def test_parse_variant_markers():
    patch = """------- SEARCH
old
=======
new
+++++++ REPLACE"""
    blocks = parse_search_replace(patch)
    assert len(blocks) == 1
    assert blocks[0].search == "old"
    assert blocks[0].replace == "new"


def test_exact_apply():
    content, result = apply_search_replace(
        ORIGINAL, _block("    return a - b", "    return a + b")
    )
    assert result.applied
    assert result.strategy == "exact"
    assert "return a + b" in content


def test_trailing_whitespace_match():
    content = "def f():\n    return 1   \n"
    new, result = apply_search_replace(content, _block("    return 1", "    return 2"))
    assert result.applied
    assert "return 2" in new


def test_whitespace_insensitive_with_reindent():
    content = "class A:\n    def f(self):\n        return 1\n"
    # search 块缩进比实际小
    new, result = apply_search_replace(content, _block("return 1", "return 42"))
    assert result.applied
    assert result.strategy in {"whitespace", "fuzzy"}
    assert "\n        return 42" in new


def test_fuzzy_match():
    content = "def f():\n    value = compute(1, 2)\n    return value\n"
    new, result = apply_search_replace(
        content, _block("value = comput(1, 2)", "value = compute(3, 4)")
    )
    assert result.applied
    assert "compute(3, 4)" in new


def test_no_match_reports_failure():
    content, result = apply_search_replace(
        ORIGINAL, _block("completely unrelated text", "other")
    )
    assert not result.applied
    assert content == ORIGINAL
    assert result.error


def test_empty_search_creates_content():
    new, result = apply_search_replace("", _block("", "print('hi')\n"))
    assert result.applied
    assert "print('hi')" in new


def test_multiple_blocks_applied_sequentially():
    patch = """<<<<<<< SEARCH
    return a - b
=======
    return a + b
>>>>>>> REPLACE
<<<<<<< SEARCH
    return a * b
=======
    return a ** b
>>>>>>> REPLACE"""
    outcome = apply_patch(ORIGINAL, patch)
    assert outcome.applied
    assert outcome.applied_count == 2
    assert "a + b" in outcome.new_content
    assert "a ** b" in outcome.new_content


def test_partial_failure_marked_not_applied():
    patch = """<<<<<<< SEARCH
    return a - b
=======
    return a + b
>>>>>>> REPLACE
<<<<<<< SEARCH
nonexistent
=======
whatever
>>>>>>> REPLACE"""
    outcome = apply_patch(ORIGINAL, patch)
    assert not outcome.applied
    assert outcome.applied_count == 1


def test_indentation_preserved_on_exact():
    content = "def f():\n    if x:\n        return 1\n"
    new, result = apply_search_replace(content, _block("        return 1", "        return 2"))
    assert result.applied
    assert "        return 2" in new


# --------------------------------------------------------------------------- #
# Unified Diff
# --------------------------------------------------------------------------- #


def test_is_format_detection():
    assert is_search_replace("<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE")
    assert not is_search_replace("def f():\n    pass")
    assert is_unified_diff("@@ -1,3 +1,3 @@\n a\n-b\n+c")
    assert not is_unified_diff("plain text")


def test_unified_diff_apply():
    new = ORIGINAL.replace("return a - b", "return a + b")
    diff = build_unified_diff(ORIGINAL, new, path="calc.py")
    outcome = apply_patch(ORIGINAL, diff)
    assert outcome.applied
    assert outcome.method == "unified_diff"
    assert "return a + b" in outcome.new_content


def test_unified_diff_with_offset():
    # 目标文件比 diff 描述的多出若干前置行
    prefix = "\n# extra header\n\n"
    content = prefix + ORIGINAL
    new = ORIGINAL.replace("return a - b", "return a + b")
    diff = build_unified_diff(ORIGINAL, new, path="calc.py")
    outcome = apply_patch(content, diff)
    assert outcome.applied
    assert "return a + b" in outcome.new_content


def test_unified_diff_hunk_not_found():
    diff = "@@ -1,1 +1,1 @@\n-nonexistent line\n+replacement\n"
    outcome = apply_patch(ORIGINAL, diff)
    assert not outcome.applied
    assert outcome.results[0].applied is False


def test_unified_diff_preserves_trailing_newline():
    new = ORIGINAL.replace("return a - b", "return a + b")
    diff = build_unified_diff(ORIGINAL, new, path="calc.py")
    outcome = apply_patch(ORIGINAL, diff)
    assert outcome.new_content.endswith("\n")


def test_count_changed_lines():
    added, removed = count_changed_lines("a\nb\n", "a\nc\nd\n")
    assert added >= 1
    assert removed >= 1


# --------------------------------------------------------------------------- #
# 文件级
# --------------------------------------------------------------------------- #


def test_apply_patch_to_file(tmp_path):
    target = tmp_path / "calc.py"
    target.write_text(ORIGINAL, encoding="utf-8")
    patch = """<<<<<<< SEARCH
    return a - b
=======
    return a + b
>>>>>>> REPLACE"""
    outcome = apply_patch_to_file(target, patch)
    assert outcome.applied
    assert "return a + b" in target.read_text(encoding="utf-8")


def test_search_replace_preserves_multiple_trailing_newlines():
    """补丁不应压缩文件末尾的多个空行（旧实现 rstrip 后只补回一个换行）。"""
    content = "def f():\n    return 1\n\n\n"
    new, result = apply_search_replace(content, _block("    return 1", "    return 2"))
    assert result.applied
    assert new == "def f():\n    return 2\n\n\n"


def test_reindent_follows_tab_indented_file():
    """tab 缩进文件的 whitespace/fuzzy 路径必须补 tab，而非改成空格（混合缩进）。"""
    content = "class A:\n\tdef f(self):\n\t\treturn 1\n"
    new, result = apply_search_replace(content, _block("return 1", "return 42"))
    assert result.applied
    assert result.strategy in {"whitespace", "fuzzy"}
    assert "\t\treturn 42" in new


def test_unified_diff_keeps_multiple_trailing_newlines():
    content = "def f():\n    return 1\n\n"
    diff = "@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 2\n"
    outcome = apply_patch(content, diff)
    assert outcome.applied
    assert outcome.new_content == "def f():\n    return 2\n\n"


def test_apply_patch_unknown_format():
    outcome = apply_patch(ORIGINAL, "just some random text")
    assert not outcome.applied
    assert outcome.method == "unknown"


def test_describe_output():
    outcome = apply_patch(
        ORIGINAL, "<<<<<<< SEARCH\n    return a - b\n=======\n    return a + b\n>>>>>>> REPLACE"
    )
    text = outcome.describe()
    assert "method=search_replace" in text
    assert "OK" in text


def test_anchor_prefilter_hits_correct_occurrence():
    """按块首行定位候选起点后仍必须命中正确位置（重复首行场景）。"""
    content = "".join(f"def f{i}():\n    return {i}\n" for i in range(20))
    content += "def g():\n    return 99\n"
    new, result = apply_search_replace(
        content, _block("def f1():\n    return 1", "def f1():\n    return 111")
    )
    assert result.applied
    assert result.strategy == "exact"
    assert new.count("return 111") == 1
    # 命中正确位置：其余函数与 f1 的签名均未被波及
    assert "def f1():\n    return 111\n" in new
    assert "    return 19\n" in new
    assert "    return 0\n" in new
    assert new.count("    return 1\n") == 0  # f1 的原返回值已被替换


def test_pathological_block_cost_is_bounded():
    """病态 SEARCH 块不得退化为分钟级 CPU 占用（可被外部触发的 CPU DoS）。

    回归背景：块首行在全文找不到时会退化为「每个窗口一次 SequenceMatcher」，
    实测 5000 行文件 + 300 行块需 8.95s、单行 60KB 块需 27.09s；而该函数在
    事件循环内被调用，等于把整个服务拖住。修复后两者均在毫秒级返回「未匹配」。
    """
    import time

    content = "".join(f"line_{i} = {i}\n" for i in range(5000))
    blocks = [
        "<<<<<<< SEARCH\n"
        + "\n".join(f"missing_{i} = {i}" for i in range(300))
        + "\n=======\n    return 99\n>>>>>>> REPLACE",
        "<<<<<<< SEARCH\n" + "x" * 60000 + "\n=======\n    return 99\n>>>>>>> REPLACE",
    ]
    for patch in blocks:
        start = time.monotonic()
        outcome = apply_patch(content, patch)
        elapsed = time.monotonic() - start
        assert not outcome.applied
        assert elapsed < 2.0, f"病态补丁耗时 {elapsed:.2f}s（应保持在毫秒级）"
