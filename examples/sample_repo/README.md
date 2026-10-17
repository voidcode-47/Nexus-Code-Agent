# 示例仓库

一个极简计算器模块，`calculator.add` 中「减法替代加法」制造了一个可被测试发现的缺陷。

用于演示 Nexus-Code-Agent 的端到端自愈闭环：

```bash
python -m src.cli heal \
  --repo examples/sample_repo \
  --issue "test_add 失败：AssertionError，add(1, 2) 期望 3 实际得到 -1"
```

预期：Agent 检索定位到 `calculator.add` → 规划 → 生成补丁（`return a - b` → `return a + b`）
→ LSP 静态检查通过 → 沙箱 pytest 通过 → 导出标准 Git Patch。
