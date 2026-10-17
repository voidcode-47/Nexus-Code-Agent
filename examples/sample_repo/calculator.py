"""示例仓库：极简计算器（内含一个待修复缺陷，用于演示自愈闭环）。"""

from __future__ import annotations


def add(a: int, b: int) -> int:
    """返回两数之和。"""
    return a - b


def subtract(a: int, b: int) -> int:
    """返回两数之差。"""
    return a - b


def multiply(a: int, b: int) -> int:
    """返回两数之积。"""
    return a * b


def safe_divide(a: float, b: float) -> float:
    """安全除法：除数为 0 时返回 0.0。"""
    if b == 0:
        return 0.0
    return a / b
