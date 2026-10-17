"""HuggingFace SWE-bench Lite 数据集加载器。

* 在线模式：从 HuggingFace ``princeton-nlp/SWE-bench_Lite`` 拉取并缓存；
* 离线模式：从本地 ``.jsonl`` 读取（便于内网 / CI / 单元测试）；
* 沙盒兜底：两者都不可用时生成**合成样例**，用于验证评测流水线自身。

字段命名对齐官方 SWE-bench schema，便于无缝切换。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.config import get_logger

logger = get_logger(__name__)

HF_DATASET = "princeton-nlp/SWE-bench_Lite"


@dataclass
class SweBenchInstance:
    """单道 SWE-bench 题目。"""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str = ""  # 官方金标准补丁（仅用于参考 / 对照）
    test_patch: str = ""
    FAIL_TO_PASS: list[str] = field(default_factory=list)
    PASS_TO_PASS: list[str] = field(default_factory=list)
    version: str = ""
    environment_setup_commit: str = ""
    #: 本地仓库路径（离线评测时使用）
    local_repo: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SweBenchInstance:
        valid = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in valid})  # type: ignore[arg-type]

    @property
    def fail_to_pass_tests(self) -> list[str]:
        return _normalize_tests(self.FAIL_TO_PASS)

    @property
    def pass_to_pass_tests(self) -> list[str]:
        return _normalize_tests(self.PASS_TO_PASS)


def _normalize_tests(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return list(parsed) if isinstance(parsed, list) else [raw]
        except json.JSONDecodeError:
            return [raw] if raw else []
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    return []


def load_swe_bench_lite(
    *,
    limit: int | None = None,
    split: str = "test",
    cache_dir: str | Path | None = None,
    offline_path: str | Path | None = None,
    allow_online: bool = True,
    allow_synthetic: bool = True,
) -> list[SweBenchInstance]:
    """加载 SWE-bench Lite 子集。

    优先级：``offline_path`` → HuggingFace（``allow_online``）→ 合成样例。
    """
    if offline_path is not None:
        instances = load_jsonl(offline_path)
        return instances[:limit] if limit else instances

    if allow_online:
        try:  # pragma: no cover - 需要网络
            from datasets import load_dataset

            logger.info("从 HuggingFace 加载 SWE-bench Lite", extra={"split": split})
            dataset = load_dataset(
                HF_DATASET, split=split, cache_dir=str(cache_dir) if cache_dir else None
            )
            instances = [SweBenchInstance.from_dict(_coerce_row(row)) for row in dataset]
            return instances[:limit] if limit else instances
        except Exception as exc:  # pragma: no cover
            logger.warning("HuggingFace 加载失败，尝试离线兜底", extra={"error": str(exc)})

    if allow_synthetic:
        logger.warning("使用合成样例数据（仅用于跑通评测流水线）")
        instances = _synthetic_instances()
        return instances[:limit] if limit else instances
    return []


def _coerce_row(row: dict[str, object]) -> dict[str, object]:
    data = dict(row)
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        data[key] = _normalize_tests(data.get(key))
    return data


def load_jsonl(path: str | Path) -> list[SweBenchInstance]:
    """从 JSONL 读取题目（坏行跳过并告警；外部数据源不可信）。"""
    instances: list[SweBenchInstance] = []
    text = Path(path).read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError("记录不是 JSON 对象")
            instances.append(SweBenchInstance.from_dict(data))
        except Exception as exc:
            logger.warning(
                "跳过非法 JSONL 记录",
                extra={"path": str(path), "line": lineno, "error": str(exc)},
            )
    return instances


def dump_jsonl(instances: Iterable[SweBenchInstance], path: str | Path) -> None:
    """将题目写入 JSONL。"""
    with Path(path).open("w", encoding="utf-8") as fh:
        for inst in instances:
            fh.write(json.dumps(inst.to_dict(), ensure_ascii=False) + "\n")


def _synthetic_instances() -> list[SweBenchInstance]:
    """生成两条合成题目，用于验证评测流水线。"""
    return [
        SweBenchInstance(
            instance_id="nexus__sample-1",
            repo="nexus/sample",
            base_commit="0" * 40,
            problem_statement="calculator.add 返回了错误的求和结果（减法替代了加法）。",
            FAIL_TO_PASS=["tests/test_calculator.py::test_add"],
            PASS_TO_PASS=["tests/test_calculator.py::test_sub"],
            version="1.0",
        ),
        SweBenchInstance(
            instance_id="nexus__sample-2",
            repo="nexus/sample",
            base_commit="0" * 40,
            problem_statement="calculator.multiply 使用了加法而非乘法。",
            FAIL_TO_PASS=["tests/test_calculator.py::test_multiply"],
            PASS_TO_PASS=["tests/test_calculator.py::test_add"],
            version="1.0",
        ),
    ]
