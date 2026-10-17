"""变异挖掘：在 cerberus 源码自动注入单行算子变异，保留「被测试捕获」的缺陷。

输出 SWE-bench 兼容 schema 的 JSONL（instance_id / problem_statement / FAIL_TO_PASS / bug 元数据）。
方法学与 Defects4J / mutation testing 一致：每个保留缺陷都有真实失败测试作为证据。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
TARGET = int(sys.argv[3]) if len(sys.argv) > 3 else 20
MAX_PER_FILE = 5
MAX_CANDIDATES = 90
MAX_FAILED = 30  # 排除近似「全崩」的变异（如破坏核心导入路径）

OPS = [
    ("==", "!="), ("<=", "<"), (">=", ">"),
    (" and ", " or "), (" is ", " is not "), (" in ", " not in "),
    ("True", "False"),
]

PKG = REPO / "cerberus"
SRC_FILES = sorted(p for p in PKG.glob("*.py") if p.name not in {"__init__.py", "platform.py"})


def run_tests() -> tuple[bool, list[str], str]:
    """跑 cerberus/tests，返回 (有失败, FAILED 测试 id 列表, 尾部摘要)。"""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=no", "-rf", "-p", "no:cacheprovider",
         "cerberus/tests"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120,
    )
    out = proc.stdout or ""
    failed = [ln.split()[1] for ln in out.splitlines() if ln.startswith("FAILED")]
    return bool(failed), failed, out.strip().splitlines()[-1] if out.strip() else ""


def candidates() -> list[tuple[Path, int, str, str]]:
    """枚举 (文件, 行号, 原片段, 变异片段)，round-robin 分散到各文件。"""
    per_file: dict[Path, list[tuple[int, str, str]]] = {}
    for path in SRC_FILES:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        found = []
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith('"""'):
                continue
            for old, new in OPS:
                if old in line:
                    found.append((idx, old, new))
                    break  # 每行只取一种变异，保证单点缺陷
        if found:
            per_file[path] = found
    picked: list[tuple[Path, int, str, str]] = []
    cursors = {p: 0 for p in per_file}
    while len(picked) < MAX_CANDIDATES and any(cursors[p] < len(per_file[p]) for p in per_file):
        for path in per_file:
            i = cursors[path]
            if i < len(per_file[path]) and picked.count(path) < MAX_PER_FILE * 4:
                lineno, old, new = per_file[path][i]
                cursors[path] = i + 1
                picked.append((path, lineno, old, new))
                if len(picked) >= MAX_CANDIDATES:
                    break
    return picked


def main() -> None:
    viable: list[dict] = []
    tested = 0
    baseline_ok, _, tail = run_tests()
    assert not baseline_ok, f"基线必须全绿，实际: {tail}"
    print(f"baseline OK: {tail}")

    for path, lineno, old, new in candidates():
        if len(viable) >= TARGET:
            break
        tested += 1
        original = path.read_bytes()
        lines = original.decode("utf-8").splitlines(keepends=True)
        if old not in lines[lineno - 1]:
            continue
        lines[lineno - 1] = lines[lineno - 1].replace(old, new, 1)
        path.write_bytes("".join(lines).encode("utf-8"))
        try:
            failed, ids, tail = run_tests()
            if failed and len(ids) <= MAX_FAILED and "error" not in tail.lower():
                rel = path.relative_to(REPO).as_posix()
                bug_line = lines[lineno - 1].rstrip()
                viable.append({
                    "instance_id": f"cerberus-mut-{len(viable) + 1:02d}",
                    "repo": "pyeve/cerberus",
                    "version": "1.3.x",
                    "bug": {"file": rel, "line": lineno, "find": old, "replace": new,
                            "mutated_line": bug_line},
                    "FAIL_TO_PASS": ids,
                    "n_failed": len(ids),
                })
                print(f"[+] {viable[-1]['instance_id']}  {rel}:{lineno}  {old.strip()!r}->{new.strip()!r}  失败 {len(ids)} 个测试")
        finally:
            path.write_bytes(original)  # 无条件还原

    OUT.write_text("".join(json.dumps(b, ensure_ascii=False) + "\n" for b in viable), encoding="utf-8")
    print(f"\n完成：候选 {tested}，可用缺陷 {len(viable)} -> {OUT}")


if __name__ == "__main__":
    main()
