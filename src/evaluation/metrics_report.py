"""Pass@k、Token 成本、修复耗时分布统计与 Markdown 报表生成。

除修复率外，精细核算：

* 单题平均 **Token 消耗** 与成本（按可配置单价折算）；
* **LSP 拦截率**（静态护栏节省的沙箱算力比例）；
* 沙箱 **平均耗时** 与修复耗时分布；
* **Pass@k 曲线**（依赖 runner 的多轮尝试数据）；
* 多维雷达图（Markdown 内嵌 Mermaid ``radar-beta``，附 ASCII 降级视图），
  便于横向对比不同策略版本。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.evaluation.runner import EvalResult

#: 每百万 token 的估算成本（USD），可按评测配置调整
DEFAULT_PRICE_PER_MTOK = 0.5


@dataclass
class Metrics:
    """评测汇总指标。"""

    total: int
    resolved: int
    pass_at_1: float
    status_counts: dict[str, int]
    avg_iterations: float
    avg_tokens: float
    avg_duration_s: float
    total_tokens: int
    estimated_cost_usd: float
    lsp_block_rate: float
    duration_buckets: dict[str, int]
    dimensions: dict[str, float]
    #: Pass@k 曲线（k → 通过率，依据每题多轮尝试结果计算）
    pass_at_k_curve: dict[str, float] = field(default_factory=dict)
    #: 曲线上最大的 k 及对应通过率（即「Pass@k」口径）
    pass_k: int = 1
    pass_at_k: float = 0.0
    avg_attempts: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "resolved": self.resolved,
            "pass@1": round(self.pass_at_1, 4),
            "pass@k": round(self.pass_at_k, 4),
            "pass_k": self.pass_k,
            "pass_at_k_curve": {k: round(v, 4) for k, v in self.pass_at_k_curve.items()},
            "status_counts": self.status_counts,
            "avg_iterations": round(self.avg_iterations, 2),
            "avg_attempts": round(self.avg_attempts, 2),
            "avg_tokens": round(self.avg_tokens, 1),
            "avg_duration_s": round(self.avg_duration_s, 2),
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "lsp_block_rate": round(self.lsp_block_rate, 4),
            "duration_buckets": self.duration_buckets,
            "dimensions": {k: round(v, 3) for k, v in self.dimensions.items()},
        }


def compute_metrics(
    results: Sequence[EvalResult],
    *,
    price_per_mtok: float = DEFAULT_PRICE_PER_MTOK,
) -> Metrics:
    """汇总评测结果。"""
    total = len(results)
    resolved = sum(1 for r in results if r.resolved)
    status_counts = dict(Counter(r.status for r in results))
    total_tokens = sum(r.tokens for r in results)
    total_duration = sum(r.duration_s for r in results)
    lsp_blocked = sum(1 for r in results if r.lsp_blocked)

    avg_iterations = sum(r.iterations for r in results) / total if total else 0.0
    avg_tokens = total_tokens / total if total else 0.0
    avg_duration = total_duration / total if total else 0.0
    lsp_block_rate = lsp_blocked / total if total else 0.0

    buckets = {"<30s": 0, "30-120s": 0, "2-10min": 0, ">10min": 0}
    for r in results:
        d = r.duration_s
        if d < 30:
            buckets["<30s"] += 1
        elif d < 120:
            buckets["30-120s"] += 1
        elif d < 600:
            buckets["2-10min"] += 1
        else:
            buckets[">10min"] += 1

    def _first_succeeded(r: EvalResult) -> bool:
        # Pass@1 = 首次采样成功率；缺逐轮记录（旧数据/外部构造结果）时退化为最终 resolved
        return bool(r.attempt_outcomes[0]) if r.attempt_outcomes else r.resolved

    # resolved 是「配置轮数内任一次成功」（= Pass@max_k），不能直接当 Pass@1，
    # 否则多轮重试会高估首修率
    first_succeeded = sum(1 for r in results if _first_succeeded(r))
    pass_at_1 = first_succeeded / total if total else 0.0
    # Pass@k：依据每题多轮尝试结果计算曲线（无尝试数据时退化为 Pass@1）
    per_instance: dict[str, list[bool]] = {
        r.instance_id: r.attempt_outcomes for r in results if r.attempt_outcomes
    }
    max_k = max((len(v) for v in per_instance.values()), default=1)
    curve = {f"Pass@{k}": pass_at_k(per_instance, k) for k in range(1, max_k + 1)}
    pass_k_value = curve.get(f"Pass@{max_k}", pass_at_1)
    made = [max(1, r.attempts_made or len(r.attempt_outcomes)) for r in results]
    avg_attempts = sum(made) / total if total else 1.0
    # 归一化维度（0~1，越高越好）用于雷达图
    efficiency = max(0.0, 1.0 - min(1.0, avg_duration / 600.0))
    cost_score = max(0.0, 1.0 - min(1.0, avg_tokens / 50_000.0))
    skipped = status_counts.get("SKIPPED", 0)
    executed = total - skipped
    # 稳定性：排除 SKIPPED（被跳过的任务不代表稳定，算进分子会虚高）
    stability = status_counts.get("SUCCESS", 0) / executed if executed > 0 else 0.0
    dimensions = {
        "修复率": pass_at_1,
        "效率": efficiency,
        "成本控制": cost_score,
        "稳定性": stability,
        "静态拦截收益": lsp_block_rate,
    }

    return Metrics(
        total=total,
        resolved=resolved,
        pass_at_1=pass_at_1,
        status_counts=status_counts,
        avg_iterations=avg_iterations,
        avg_tokens=avg_tokens,
        avg_duration_s=avg_duration,
        total_tokens=total_tokens,
        estimated_cost_usd=total_tokens / 1_000_000 * price_per_mtok,
        lsp_block_rate=lsp_block_rate,
        duration_buckets=buckets,
        dimensions=dimensions,
        pass_at_k_curve=curve,
        pass_k=max_k,
        pass_at_k=pass_k_value,
        avg_attempts=avg_attempts,
    )


def pass_at_k(per_instance_attempts: dict[str, list[bool]], k: int) -> float:
    """按情形计算 Pass@k（同一实例多次尝试中前 k 次成功即算通过）。"""
    if not per_instance_attempts:
        return 0.0
    hits = 0
    for attempts in per_instance_attempts.values():
        if any(attempts[:k]):
            hits += 1
    return hits / len(per_instance_attempts)


def render_markdown(
    metrics: Metrics,
    results: Sequence[EvalResult] | None = None,
    *,
    title: str = "Nexus-Code-Agent 评测报告",
) -> str:
    """生成 Markdown 报表（含 Pass@k、雷达图与耗时分布）。"""
    lines: list[str] = [f"# {title}", ""]
    lines.append("## 1. 核心指标")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 题目总数 | {metrics.total} |")
    lines.append(f"| 修复成功 | {metrics.resolved} |")
    lines.append(f"| **Pass@1** | **{metrics.pass_at_1:.1%}** |")
    if metrics.pass_k > 1:
        lines.append(f"| **Pass@{metrics.pass_k}** | **{metrics.pass_at_k:.1%}** |")
        curve = " ｜ ".join(f"{k} {v:.1%}" for k, v in metrics.pass_at_k_curve.items())
        lines.append(f"| Pass@k 曲线 | {curve} |")
    lines.append(f"| 平均尝试轮数 | {metrics.avg_attempts:.2f} |")
    lines.append(f"| 平均迭代次数 | {metrics.avg_iterations:.2f} |")
    lines.append(f"| 单题平均 Token | {metrics.avg_tokens:,.0f} |")
    lines.append(f"| 单题平均耗时 * | {metrics.avg_duration_s:.1f}s |")
    lines.append(f"| 总 Token 消耗 | {metrics.total_tokens:,} |")
    lines.append(f"| 估算成本 (USD) | ${metrics.estimated_cost_usd:.4f} |")
    lines.append(f"| LSP 拦截率 | {metrics.lsp_block_rate:.1%} |")
    lines.append("")
    lines.append("> * 单题平均耗时包含索引、检索、静态校验与沙箱执行的**全流程**耗时。")
    lines.append("")

    lines.append("## 2. 多维雷达图")
    lines.append("")
    lines.append("```mermaid")
    lines.append("radar-beta")
    lines.append(
        '  axis '
        + ", ".join(f'a{i}["{name}"]' for i, name in enumerate(metrics.dimensions, start=1))
    )
    lines.append(
        "  curve nexus[" + title.split()[0] + "]{" + ", ".join(
            f"{max(0.0, min(1.0, v)):.3f}" for v in metrics.dimensions.values()
        ) + "}"
    )
    lines.append("  max 1")
    lines.append("```")
    lines.append("")
    lines.append("终端降级视图（不支持 Mermaid radar-beta 的渲染器请参考）：")
    lines.append("")
    for name, value in metrics.dimensions.items():
        bar = "█" * int(round(value * 20))
        lines.append(f"- {name:<10} {bar:<20} {value:.1%}")
    lines.append("")

    lines.append("## 3. 修复耗时分布")
    lines.append("")
    lines.append("| 区间 | 数量 | 占比 |")
    lines.append("| --- | --- | --- |")
    for bucket, count in metrics.duration_buckets.items():
        share = count / metrics.total if metrics.total else 0.0
        lines.append(f"| {bucket} | {count} | {share:.1%} |")
    lines.append("")

    lines.append("## 4. 状态分布")
    lines.append("")
    lines.append("| 状态 | 数量 |")
    lines.append("| --- | --- |")
    for status, count in sorted(metrics.status_counts.items()):
        lines.append(f"| {status} | {count} |")
    lines.append("")

    if results:
        lines.append("## 5. 逐题明细")
        lines.append("")
        lines.append("| instance_id | resolved | status | iters | tokens | duration | reason |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in results:
            lines.append(
                f"| {r.instance_id} | {'✅' if r.resolved else '❌'} | {r.status} | "
                f"{r.iterations} | {r.tokens:,} | {r.duration_s:.1f}s | "
                f"{r.termination_reason or r.error or '-'} |"
            )
        lines.append("")
    return "\n".join(lines)


def save_report(text: str, path: str | Path) -> Path:
    """保存报表到文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def render_console(metrics: Metrics) -> str:
    """终端友好的简要输出。"""
    headline = f"题数={metrics.total} 修复={metrics.resolved} Pass@1={metrics.pass_at_1:.1%}"
    if metrics.pass_k > 1:
        headline += f" Pass@{metrics.pass_k}={metrics.pass_at_k:.1%}"
    lines = [
        headline,
        f"平均尝试={metrics.avg_attempts:.2f} 平均迭代={metrics.avg_iterations:.2f} "
        f"平均Token={metrics.avg_tokens:,.0f} 平均耗时={metrics.avg_duration_s:.1f}s "
        f"LSP拦截率={metrics.lsp_block_rate:.1%}",
    ]
    return "\n".join(lines)


def aggregate(results: Iterable[EvalResult]) -> dict[str, Any]:
    """便捷：直接返回指标字典。"""
    return compute_metrics(list(results)).to_dict()
