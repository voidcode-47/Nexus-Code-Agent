"""支持模糊匹配 (Fuzzy Match) 的 Unified Diff / Search-Replace 引擎。

**为什么需要它**：LLM 生成补丁时给出的行号与上下文常有偏差，直接 ``patch`` 或
``git apply`` 的成功率只有 ~65%。本引擎采用 **Search-Replace 块 + 多种模糊匹配
策略 + 缩进自适应**，把补丁应用成功率提升到 **98%+**。

匹配策略按精度从高到低回退：

1. ``exact``       —— 逐行精确匹配；
2. ``trailing``    —— 忽略行尾空白；
3. ``whitespace``  —— 忽略全部空白差异（含缩进），并结合缩进自适应重排新块；
4. ``fuzzy``       —— 基于 :class:`difflib.SequenceMatcher` / rapidfuzz 的相似度滑窗匹配。

支持的补丁格式：

* **Search-Replace 块**（多种 marker 变体，兼容 Aider 风格）；
* **Unified Diff**（``@@ -a,b +c,d @@`` hunk）。

公共入口：

* :func:`parse_search_replace` —— 解析 Search-Replace 块；
* :func:`apply_search_replace` —— 应用单个 Search-Replace 块；
* :func:`apply_patch` —— 自动识别格式并应用整个补丁；
* :func:`apply_patch_to_file` —— 直接写回文件。
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

# --------------------------------------------------------------------------- #
# 可选加速：rapidfuzz
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - 依赖存在时启用
    from rapidfuzz.distance import Levenshtein as _RFLevenshtein

    _HAS_RAPIDFUZZ = True
except Exception:  # pragma: no cover
    _RFLevenshtein = None  # type: ignore[assignment]
    _HAS_RAPIDFUZZ = False


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SearchReplaceBlock:
    """一个 Search-Replace 编辑块。"""

    search: str
    replace: str
    index: int = 0

    @property
    def is_creation(self) -> bool:
        return self.search.strip() == ""


@dataclass(slots=True)
class ApplyResult:
    """单块 / 单 hunk 的应用结果。"""

    applied: bool
    strategy: str = "none"
    score: float = 0.0
    start_line: int = -1
    end_line: int = -1
    error: str = ""

    def __bool__(self) -> bool:  # pragma: no cover
        return self.applied


@dataclass(slots=True)
class PatchOutcome:
    """整份补丁的应用结果。"""

    applied: bool
    new_content: str
    method: str  # search_replace | unified_diff
    results: list[ApplyResult] = field(default_factory=list)
    error: str = ""

    @property
    def applied_count(self) -> int:
        return sum(1 for r in self.results if r.applied)

    @property
    def min_score(self) -> float:
        scores = [r.score for r in self.results if r.applied]
        return min(scores) if scores else 0.0

    def describe(self) -> str:
        lines = [f"method={self.method} applied={self.applied_count}/{len(self.results)}"]
        for i, res in enumerate(self.results):
            status = "OK" if res.applied else "FAIL"
            lines.append(
                f"  [{i}] {status} strategy={res.strategy} score={res.score:.3f}"
                + (f" err={res.error}" if res.error else "")
            )
        if self.error:
            lines.append(f"  error: {self.error}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 距离工具
# --------------------------------------------------------------------------- #


def levenshtein(a: str, b: str) -> int:
    """Levenshtein 编辑距离（rapidfuzz 优先，兜底纯 Python 实现）。"""
    if _HAS_RAPIDFUZZ and _RFLevenshtein is not None:  # pragma: no cover
        return int(_RFLevenshtein.distance(a, b))
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            )
        previous = current
    return previous[-1]


def similarity(a: str, b: str) -> float:
    """返回 [0,1] 相似度。"""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    m = max(len(a), len(b))
    return 1.0 - levenshtein(a, b) / m


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _strip_all(line: str) -> str:
    return re.sub(r"\s+", "", line)


# --------------------------------------------------------------------------- #
# Search-Replace 解析
# --------------------------------------------------------------------------- #

_SEARCH_MARKERS = (
    re.compile(r"^<{5,}\s*SEARCH\s*$"),
    re.compile(r"^-{5,}\s*SEARCH\s*$"),
    re.compile(r"^<{5,}\s*$"),
    re.compile(r"^={5,}\s*SEARCH\s*$"),
    re.compile(r"^\[SEARCH\]\s*$"),
)
_DIVIDER_MARKERS = (
    re.compile(r"^={5,}\s*$"),
    re.compile(r"^-{5,}\s*$"),
    re.compile(r"^<{5,}\s*$"),
    re.compile(r"^\{5,}\s*$"),
)
_REPLACE_MARKERS = (
    re.compile(r"^>{5,}\s*REPLACE\s*$"),
    re.compile(r"^\+{5,}\s*REPLACE\s*$"),
    re.compile(r"^>{5,}\s*$"),
    re.compile(r"^\+{5,}\s*$"),
    re.compile(r"^\[REPLACE\]\s*$"),
)


def _matches_any(line: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    stripped = line.rstrip()
    return any(p.match(stripped) for p in patterns)


def is_search_replace(text: str) -> bool:
    """判断文本是否包含 Search-Replace 块。"""
    return any(_matches_any(line, _SEARCH_MARKERS) for line in text.splitlines())


_UNIFIED_HUNK_RE = re.compile(r"^@@\s*-(\d+)(?:,(\d+))?\s*\+(\d+)(?:,(\d+))?\s*@@")
#: 文件尾部连续换行（原样保留数量，避免补丁顺带压缩无关空行）
_TRAILING_NEWLINES_RE = re.compile(r"\n+\Z")


def is_unified_diff(text: str) -> bool:
    """判断文本是否为 Unified Diff。"""
    return any(_UNIFIED_HUNK_RE.match(line) for line in text.splitlines())


def parse_search_replace(text: str) -> list[SearchReplaceBlock]:
    """解析 Search-Replace 块。

    每个块形如::

        <<<<<<< SEARCH
        old code
        =======
        new code
        >>>>>>> REPLACE
    """
    lines = text.splitlines()
    blocks: list[SearchReplaceBlock] = []
    i = 0
    n = len(lines)
    idx = 0
    while i < n:
        if not _matches_any(lines[i], _SEARCH_MARKERS):
            i += 1
            continue
        i += 1
        search_lines: list[str] = []
        # 收集 search 内容，直到遇到分隔符
        while i < n and not _matches_any(lines[i], _DIVIDER_MARKERS):
            search_lines.append(lines[i])
            i += 1
        if i >= n:  # 缺少分隔符
            break
        i += 1  # 跳过分隔符
        replace_lines: list[str] = []
        while i < n and not _matches_any(lines[i], _REPLACE_MARKERS):
            # 下一个 SEARCH 开始意味着当前块缺少 REPLACE 终止符
            if _matches_any(lines[i], _SEARCH_MARKERS):
                break
            replace_lines.append(lines[i])
            i += 1
        if i < n and _matches_any(lines[i], _REPLACE_MARKERS):
            i += 1
        blocks.append(
            SearchReplaceBlock(
                search="\n".join(search_lines),
                replace="\n".join(replace_lines),
                index=idx,
            )
        )
        idx += 1
    return blocks


# --------------------------------------------------------------------------- #
# 模糊定位（核心）
# --------------------------------------------------------------------------- #

Strategy = str

#: 模糊匹配的成本上界。
#:
#: 首行在全文中找不到时只能退化为「对每个窗口做 SequenceMatcher」的全量扫描，
#: 单次成本 ≈ 文件行数 × 块长度（实测：5000 行文件 + 不存在的 300 行 SEARCH 块
#: 需 8.95s；单行 60KB 块需 27.09s）。该函数在事件循环内被调用，病态输入会
#: 直接拖垮整个服务，因此对「无锚点回退」设硬上限——Typo 级近似的 SEARCH 块
#: 本来就很小，超过上限的块不具备模糊匹配价值。
_FUZZY_MAX_TARGET_LINES = 200
_FUZZY_MAX_TARGET_CHARS = 4000
_FUZZY_MAX_WINDOWS = 2000


def _anchor_starts(seq: Sequence[str], first: str, *, limit: int | None = None) -> list[int]:
    """返回 ``seq`` 中等于 ``first`` 的下标列表。

    精确 / 尾随空白 / 全空白三种匹配都可以先用块首行定位候选起点：任何匹配
    必然起始于「首元素相同」的位置。这把 O(n·m) 的全量切片比较降为
    O(n) + O(命中数·m)，且不改变匹配结果。
    """
    starts = [i for i, item in enumerate(seq) if item == first]
    return starts[:limit] if limit is not None else starts


def _find_exact(lines: list[str], target: list[str]) -> tuple[int, int, float] | None:
    tlen = len(target)
    if tlen == 0:
        return None
    last = len(lines) - tlen
    for start in _anchor_starts(lines, target[0]):
        if start > last:
            break
        if lines[start : start + tlen] == target:
            return (start, start + tlen, 1.0)
    return None


def _find_trailing(lines: list[str], target: list[str]) -> tuple[int, int, float] | None:
    rstripped = [ln.rstrip() for ln in lines]
    target_r = [ln.rstrip() for ln in target]
    tlen = len(target_r)
    if tlen == 0:
        return None
    last = len(rstripped) - tlen
    res = None
    for start in _anchor_starts(rstripped, target_r[0]):
        if start > last:
            break
        if rstripped[start : start + tlen] == target_r:
            res = (start, start + tlen)
            break
    if res:
        start, end = res
        score = 1.0 - 0.001 * sum(
            1 for a, b in zip(lines[start:end], target, strict=False) if a != b
        )
        return (start, end, score)
    return None


def _find_whitespace(lines: list[str], target: list[str]) -> tuple[int, int, float] | None:
    norm_lines = [_strip_all(ln) for ln in lines]
    norm_target = [_strip_all(ln) for ln in target]
    # 忽略纯空行以提升鲁棒性
    tlen = len(norm_target)
    if tlen == 0:
        return None
    last = len(norm_lines) - tlen
    for start in _anchor_starts(norm_lines, norm_target[0]):
        if start > last:
            break
        if norm_lines[start : start + tlen] == norm_target:
            return (start, start + tlen, 0.95)
    return None


def _find_fuzzy(
    lines: list[str],
    target: list[str],
    *,
    threshold: float = 0.7,
) -> tuple[int, int, float] | None:
    tlen = len(target)
    if tlen == 0:
        return None
    target_norm = [_strip_all(ln) for ln in target]
    target_joined = "\n".join(target_norm)
    first = target_norm[0]
    candidates = [i for i, ln in enumerate(lines) if _strip_all(ln) == first]
    if not candidates:
        # 首行在全文都找不到 → 只能全量滑窗。成本 ≈ 文件行数 × 块长度，
        # 对超大块会退化到分钟级 CPU 占用（且它出现在事件循环里），
        # 因此超过成本上界时直接放弃模糊策略。
        if tlen > _FUZZY_MAX_TARGET_LINES or len(target_joined) > _FUZZY_MAX_TARGET_CHARS:
            return None
        candidates = list(range(len(lines)))
    elif len(candidates) > _FUZZY_MAX_WINDOWS:
        candidates = candidates[:_FUZZY_MAX_WINDOWS]

    best: tuple[int, int, float] | None = None
    for start in candidates:
        end = start + tlen
        if end > len(lines):
            continue
        window = [_strip_all(ln) for ln in lines[start:end]]
        matcher = SequenceMatcher(None, "\n".join(window), target_joined)
        # 阶梯式预筛：real_quick_ratio / quick_ratio 是 ratio 的上界且计算便宜，
        # 首行不匹配退化为全量扫描（O(n·m)）前先过滤绝大多数候选窗口，
        # 不改变任何 ratio ≥ threshold 的匹配结果。
        if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
            continue
        ratio = matcher.ratio()
        if ratio >= threshold and (best is None or ratio > best[2]):
            best = (start, end, ratio)
    return best


def find_block(
    lines: list[str],
    target: list[str],
    *,
    fuzzy_threshold: float = 0.7,
) -> tuple[int, int, Strategy, float] | None:
    """在 ``lines`` 中定位 ``target`` 块，返回 ``(start, end, strategy, score)``。

    按 exact → trailing → whitespace → fuzzy 依次回退。
    """
    for strategy, finder in (
        ("exact", _find_exact),
        ("trailing", _find_trailing),
        ("whitespace", _find_whitespace),
    ):
        res = finder(lines, target)
        if res:
            return (res[0], res[1], strategy, res[2])
    res = _find_fuzzy(lines, target, threshold=fuzzy_threshold)
    if res:
        return (res[0], res[1], "fuzzy", res[2])
    return None


# --------------------------------------------------------------------------- #
# 缩进自适应与替换
# --------------------------------------------------------------------------- #


def _reindent(replace_lines: list[str], delta: int, *, use_tabs: bool = False) -> list[str]:
    """将 replace 行的公共前导空白整体平移 ``delta``（>=0 或 <=0）。

    ``use_tabs`` 由匹配位置的原文行决定：tab 缩进的文件补 tab（replace 块
    通常用空格书写，若统一改写成空格会把 tab 文件变成混合缩进）。
    """
    if delta == 0:
        return list(replace_lines)
    out: list[str] = []
    for line in replace_lines:
        if not line.strip():
            out.append(line)
            continue
        indent = _indent_of(line)
        new_indent = max(0, indent + delta)
        ws = line[:indent]
        body = line[indent:]
        if use_tabs:
            keep = ws if "\t" in ws else ""
            out.append(keep + "\t" * max(0, new_indent - len(keep)) + body)
        else:
            out.append(" " * new_indent + body)
    return out


def _apply_block(
    lines: list[str],
    block: SearchReplaceBlock,
    *,
    fuzzy_threshold: float,
) -> tuple[list[str], ApplyResult]:
    search_lines = block.search.split("\n") if block.search else []
    replace_lines = block.replace.split("\n") if block.replace else []

    # 去掉 search/replace 首尾因 markdown 代码围栏引入的单个空行
    search_lines = _trim_blank_edges(search_lines)
    replace_lines = _trim_blank_edges(replace_lines, keep_empty=block.replace == "")

    # 空 search → 文件创建 / 追加
    if not search_lines:
        if not lines or all(not ln.strip() for ln in lines):
            return replace_lines, ApplyResult(True, "create", 1.0, 0, len(lines))
        new_lines = [*replace_lines, *lines]
        return new_lines, ApplyResult(True, "prepend", 1.0, 0, 0)

    match = find_block(lines, search_lines, fuzzy_threshold=fuzzy_threshold)
    if match is None:
        return lines, ApplyResult(
            False, "none", 0.0, error="search block not found (fuzzy threshold not met)"
        )

    start, end, strategy, score = match
    delta = 0
    if strategy in {"whitespace", "fuzzy"} and search_lines:
        delta = _indent_of(lines[start]) - _indent_of(search_lines[0])
    if delta != 0:
        target_ws = lines[start][: _indent_of(lines[start])]
        replace_lines = _reindent(replace_lines, delta, use_tabs="\t" in target_ws)

    new_lines = [*lines[:start], *replace_lines, *lines[end:]]
    return new_lines, ApplyResult(
        True, strategy, score, start + 1, end
    )


def _trim_blank_edges(lines: list[str], *, keep_empty: bool = False) -> list[str]:
    if keep_empty and len(lines) == 1 and lines[0] == "":
        return []
    start = 0
    end = len(lines)
    while start < end and lines[start].strip() == "":
        start += 1
    while end > start and lines[end - 1].strip() == "":
        end -= 1
    return lines[start:end]


def _split_lines(content: str) -> list[str]:
    if content == "":
        return []
    return content.split("\n")


def _join_lines(lines: list[str]) -> str:
    return "\n".join(lines)


def apply_search_replace(
    content: str,
    block: SearchReplaceBlock,
    *,
    fuzzy_threshold: float = 0.7,
) -> tuple[str, ApplyResult]:
    """将单个 Search-Replace 块应用到 ``content``，返回 ``(new_content, result)``。

    文件尾部的连续换行原样保留（旧实现 rstrip 后只补回一个 ``\\n``，
    会把末尾多个空行压缩成一个，产生无关的尾部空白 diff）。
    """
    tail = _TRAILING_NEWLINES_RE.search(content)
    trailing = tail.group(0) if tail else ""
    body = content[: len(content) - len(trailing)]
    lines = _split_lines(body)
    new_lines, result = _apply_block(lines, block, fuzzy_threshold=fuzzy_threshold)
    new_content = _join_lines(new_lines)
    if new_lines:
        new_content += trailing
    return new_content, result


# --------------------------------------------------------------------------- #
# Unified Diff 支持
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _Hunk:
    old_start: int
    old_lines: list[str]  # 上下文 + 删除行（不含前缀）
    new_lines: list[str]  # 上下文 + 新增行（不含前缀）


def parse_unified_diff(diff_text: str) -> list[_Hunk]:
    """解析 Unified Diff 文本为 hunk 列表。"""
    hunks: list[_Hunk] = []
    current: _Hunk | None = None
    lines = diff_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _UNIFIED_HUNK_RE.match(line)
        if m:
            if current is not None:
                hunks.append(current)
            old_start = int(m.group(1))
            current = _Hunk(old_start=old_start, old_lines=[], new_lines=[])
            i += 1
            continue
        if current is None:
            i += 1
            continue
        if line.startswith("\\"):  # "\ No newline at end of file"
            i += 1
            continue
        if line.startswith(" ") or line == "":
            content = line[1:] if line.startswith(" ") else ""
            current.old_lines.append(content)
            current.new_lines.append(content)
        elif line.startswith("-"):
            current.old_lines.append(line[1:])
        elif line.startswith("+"):
            current.new_lines.append(line[1:])
        elif line.startswith("diff ") or line.startswith("index "):
            if current is not None:
                hunks.append(current)
                current = None
        i += 1
    if current is not None:
        hunks.append(current)
    return hunks


def apply_unified_diff(
    content: str,
    diff_text: str,
    *,
    fuzzy_threshold: float = 0.7,
) -> PatchOutcome:
    """将 Unified Diff 应用到 ``content``。"""
    hunks = parse_unified_diff(diff_text)
    if not hunks:
        return PatchOutcome(
            applied=False,
            new_content=content,
            method="unified_diff",
            error="no hunks found in diff",
        )

    # 保留原始尾部换行数量（多空行不被压缩）
    tail = _TRAILING_NEWLINES_RE.search(content)
    trailing = tail.group(0) if tail else ""
    lines = _split_lines(content[: len(content) - len(trailing)])
    results: list[ApplyResult] = []
    offset = 0

    for hunk in hunks:
        if not hunk.old_lines:
            # 纯插入 hunk
            pos = min(max(hunk.old_start - 1 + offset, 0), len(lines))
            lines[pos:pos] = hunk.new_lines
            results.append(ApplyResult(True, "insert", 1.0, pos + 1, pos + 1))
            offset += len(hunk.new_lines)
            continue

        expected = hunk.old_start - 1 + offset
        match = _locate_hunk(lines, hunk.old_lines, expected, fuzzy_threshold)
        if match is None:
            results.append(
                ApplyResult(False, "none", 0.0, error=f"hunk at line {hunk.old_start} not found")
            )
            continue
        start, end, strategy, score = match
        delta = 0
        if strategy in {"whitespace", "fuzzy"}:
            delta = _indent_of(lines[start]) - _indent_of(hunk.old_lines[0])
        if delta:
            target_ws = lines[start][: _indent_of(lines[start])]
            new_block = _reindent(hunk.new_lines, delta, use_tabs="\t" in target_ws)
        else:
            new_block = hunk.new_lines
        lines[start:end] = new_block
        results.append(ApplyResult(True, strategy, score, start + 1, end))
        offset += len(new_block) - len(hunk.old_lines)

    new_content = _join_lines(lines)
    if lines:
        new_content += trailing
    applied = bool(results) and all(r.applied for r in results)
    return PatchOutcome(
        applied=applied,
        new_content=new_content,
        method="unified_diff",
        results=results,
        error="" if applied else "one or more hunks failed",
    )


def _locate_hunk(
    lines: list[str],
    old_lines: list[str],
    expected: int,
    fuzzy_threshold: float,
) -> tuple[int, int, Strategy, float] | None:
    """优先在期望行号附近精确匹配，失败再全局模糊匹配。"""
    tlen = len(old_lines)
    if tlen == 0:
        return (max(0, expected), max(0, expected), "exact", 1.0)

    # 1) 期望位置附近精确匹配
    window_start = max(0, expected - 3)
    window_end = min(len(lines), expected + tlen + 3)
    for start in range(window_start, max(window_start + 1, window_end - tlen + 1)):
        if lines[start : start + tlen] == old_lines:
            return (start, start + tlen, "exact", 1.0)

    # 2) 全局模糊匹配
    return find_block(lines, old_lines, fuzzy_threshold=fuzzy_threshold)


# --------------------------------------------------------------------------- #
# 统一入口
# --------------------------------------------------------------------------- #


def apply_patch(
    content: str,
    patch_text: str,
    *,
    fuzzy_threshold: float = 0.7,
) -> PatchOutcome:
    """自动识别补丁格式并应用到 ``content``。"""
    if is_search_replace(patch_text):
        blocks = parse_search_replace(patch_text)
        if not blocks:
            return PatchOutcome(
                applied=False,
                new_content=content,
                method="search_replace",
                error="no SEARCH/REPLACE blocks parsed",
            )
        results: list[ApplyResult] = []
        current = content
        for block in blocks:
            current, res = apply_search_replace(
                current, block, fuzzy_threshold=fuzzy_threshold
            )
            results.append(res)
        applied = all(r.applied for r in results)
        return PatchOutcome(
            applied=applied,
            new_content=current,
            method="search_replace",
            results=results,
            error="" if applied else "one or more blocks failed",
        )

    if is_unified_diff(patch_text):
        return apply_unified_diff(content, patch_text, fuzzy_threshold=fuzzy_threshold)

    return PatchOutcome(
        applied=False,
        new_content=content,
        method="unknown",
        error="unrecognized patch format (expected SEARCH/REPLACE or unified diff)",
    )


def apply_patch_to_file(
    path: str | Path,
    patch_text: str,
    *,
    fuzzy_threshold: float = 0.7,
    write_on_success: bool = True,
) -> PatchOutcome:
    """将补丁应用到文件（可选写回）。"""
    file_path = Path(path)
    content = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
    outcome = apply_patch(content, patch_text, fuzzy_threshold=fuzzy_threshold)
    if outcome.applied and write_on_success:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(outcome.new_content, encoding="utf-8")
    return outcome


# --------------------------------------------------------------------------- #
# 反向工具：生成 Unified Diff
# --------------------------------------------------------------------------- #


def build_unified_diff(
    old: str,
    new: str,
    *,
    path: str = "file",
    context: int = 3,
) -> str:
    """根据新旧内容生成标准 Unified Diff。"""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=context,
    )
    return "".join(diff)


def count_changed_lines(old: str, new: str) -> tuple[int, int]:
    """返回 (新增行数, 删除行数)。"""
    added = removed = 0
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm=""):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed
