"""对标 SWE-bench 的自动化评测平台。"""

from src.evaluation.metrics_report import compute_metrics, render_markdown, save_report
from src.evaluation.runner import EvalResult, EvaluationRunner
from src.evaluation.swe_bench_loader import SweBenchInstance, load_swe_bench_lite

__all__ = [
    "compute_metrics",
    "render_markdown",
    "save_report",
    "EvalResult",
    "EvaluationRunner",
    "SweBenchInstance",
    "load_swe_bench_lite",
]
