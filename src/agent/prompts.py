"""工业级 Few-Shot 提示词模板管理。

所有 Agent 的系统提示与用户提示集中于此，便于版本化、A/B 对比与评测回归。
每个渲染函数返回 OpenAI Chat Completions 兼容的 ``messages`` 列表。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

SYSTEM_BASE = (
    "你是 Nexus-Code-Agent，一个严谨的企业级代码自愈智能体。"
    "你只能依据提供的仓库上下文作答，禁止臆造不存在的文件、符号或 API。"
    "所有修改必须是最小必要改动，并保持既有代码风格。"
)

LOCALIZER_SYSTEM = (
    f"{SYSTEM_BASE}\n"
    "你的职责是【故障定位】：根据报错堆栈与 Issue 描述，"
    "从候选代码片段中指出最可疑的文件与函数，并解释调用链因果关系。\n"
    "输出严格 JSON：{\"suspected_files\": [...], \"suspected_symbols\": [...], \"reason\": \"...\"}"
)

ARCHITECT_SYSTEM = (
    f"{SYSTEM_BASE}\n"
    "你的职责是【修复规划】：分析跨文件影响面，产出分步修改计划。"
    "只做规划，不要写具体补丁。输出简洁的有序列表，每步说明「改哪个符号、为什么、风险」。"
)

CODER_SYSTEM = (
    f"{SYSTEM_BASE}\n"
    "你的职责是【补丁生成】：根据修复计划输出可精确应用的补丁。\n"
    "严格使用如下 Search-Replace 格式，可以输出多个块；SEARCH 必须是文件中逐字存在的原文"
    "（含缩进），REPLACE 是替换后的内容：\n"
    "<<<<<<< SEARCH\n"
    "<原始代码>\n"
    "=======\n"
    "<修改后的代码>\n"
    ">>>>>>> REPLACE\n"
    "只输出补丁，不要输出任何解释文字或 Markdown 代码围栏。"
)

CRITIC_SYSTEM = (
    f"{SYSTEM_BASE}\n"
    "你的职责是【归因与熔断判别】：分析测试失败 Traceback，判断根因类别"
    "（语法/类型/逻辑/夹具/环境），并给出下一步建议。\n"
    "输出严格 JSON：{\"root_cause\": \"...\", \"category\": \"...\", "
    "\"next_action\": \"revise_patch|replan|escalate\", \"reason\": \"...\"}"
)


def localizer_messages(
    issue: str, traceback: str, candidates: str, *, few_shot: Sequence[dict[str, Any]] = ()
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": LOCALIZER_SYSTEM}]
    messages.extend(few_shot)
    messages.append(
        {
            "role": "user",
            "content": (
                f"## Issue\n{issue}\n\n"
                f"## 报错堆栈\n{traceback or '(无堆栈，仅自然语言描述)'}\n\n"
                f"## 候选代码片段\n{candidates}\n\n"
                "请给出最可疑的文件与函数（JSON）。"
            ),
        }
    )
    return messages


def architect_messages(
    issue: str, localized: str, call_graph: str
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": ARCHITECT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"## Issue\n{issue}\n\n"
                f"## 定位结论\n{localized}\n\n"
                f"## 调用图上下文\n{call_graph}\n\n"
                "请输出分步修复计划。"
            ),
        },
    ]


def coder_messages(
    plan: str,
    target_file: str,
    target_source: str,
    *,
    feedback: str = "",
    attempt: int = 0,
) -> list[dict[str, Any]]:
    feedback_block = f"\n\n## 上一轮失败反馈\n{feedback}\n请针对反馈修正补丁。" if feedback else ""
    return [
        {"role": "system", "content": CODER_SYSTEM},
        {
            "role": "user",
            "content": (
                f"## 修复计划\n{plan}\n\n"
                f"## 目标文件 `{target_file}` 当前内容\n```python\n{target_source}\n```"
                f"{feedback_block}\n\n"
                f"（这是第 {attempt + 1} 次尝试）请输出补丁。"
            ),
        },
    ]


def critic_messages(traceback: str, history_summary: str, iteration: int) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": CRITIC_SYSTEM},
        {
            "role": "user",
            "content": (
                f"## 第 {iteration} 次迭代的测试报错\n{traceback}\n\n"
                f"## 历史修改摘要\n{history_summary or '(无)'}\n\n"
                "请输出归因 JSON。"
            ),
        },
    ]


def render_snippets(hits: Sequence[Any], *, max_chars: int = 6000) -> str:
    """将检索命中渲染为紧凑的代码片段文本。"""
    parts: list[str] = []
    used = 0
    for hit in hits:
        block = (
            f"### {hit.symbol} ({hit.path}:{hit.start_line}-{hit.end_line})\n"
            f"```python\n{hit.snippet()}\n```"
        )
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)
