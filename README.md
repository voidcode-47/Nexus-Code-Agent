# Nexus-Code-Agent

[![CI](https://github.com/voidcode-47/Nexus-Code-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/voidcode-47/Nexus-Code-Agent/actions/workflows/ci.yml)
![Coverage](https://img.shields.io/badge/coverage-86%25-green)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)

> 企业级**自动化代码自愈与演进智能体平台**（Autonomous Code Healing & Evolution Platform）

Nexus-Code-Agent 接收 GitHub Issue、报警日志或自然语言指令，自动完成大型代码库的
**跨文件故障定位 → 修复方案规划 → 代码补丁生成 → 沙箱安全验证 → PR 自动提交** 全链路闭环。

## 核心能力

| 能力 | 实现 |
| --- | --- |
| 代码图谱级感知 (Code Graph RAG) | `src/indexer/` — AST 结构化切片 + 调用图 PageRank + Qdrant/BM25 RRF 混合检索 |
| 高并发安全沙箱 (High-Concurrency Sandbox) | `src/sandbox/` — Docker 预热容器池 + Git Worktree 零拷贝快照 |
| 双层校验工具链 (MCP + LSP) | `src/mcp_tools/` — 模糊匹配补丁引擎 + Pyright 静态拦截 + FastMCP 服务端 |
| 可控多智能体编排 (Stateful Multi-Agent) | `src/agent/` — LangGraph 持久化状态机 + Critic 归因熔断 + HITL |
| 自动化评测平台 | `src/evaluation/` — SWE-bench Lite 并发跑分 + Pass@k / Token 成本报表 |
| 异步服务网关 | `src/api/` — FastAPI + SSE 思考流 + GitHub Webhook 闭环 |
| PR 自动提交闭环 | `src/mcp_tools/pr_publisher.py` — 补丁 → 修复分支 → 推送远端 → GitHub API 创建 PR（逐级优雅降级） |

## 目录结构

```
nexus-code-agent/
├── docker-compose.yml
├── pyproject.toml
├── Makefile
├── src/
│   ├── config/          # Pydantic-Settings 多环境配置 + 结构化日志
│   ├── indexer/         # 代码图谱与混合检索引擎
│   ├── sandbox/         # 容器池与 Git Worktree 快照
│   ├── mcp_tools/       # MCP 标准工具矩阵
│   ├── agent/           # LangGraph 多智能体状态机
│   ├── evaluation/      # SWE-bench 评测平台
│   └── api/             # 异步服务网关
├── examples/
│   └── sample_repo/     # 内置示例仓库（含一个待修复缺陷，用于演示闭环）
└── tests/               # 单元与集成测试
```

### 相对设计文档的工程扩展

在严格实现文档目录树的基础上，补充了以下必要文件（均在文档树之外，不改变模块边界）：

| 文件 | 职责 |
| --- | --- |
| `src/agent/llm.py` | LLM 客户端抽象：OpenAI/DeepSeek 兼容实现 + 离线启发式实现（可测试、无 Key 可跑） |
| `src/agent/repair_heuristics.py` | 离线 AST 变异搜索修复引擎（无 LLM 时为 Coder 生成候选补丁） |
| `src/agent/nodes/tester.py` | LSP 静态校验 + 沙箱动态校验的「双层校验」节点 |
| `src/mcp_tools/pr_publisher.py` | PR 自动提交：Worktree 内建分支 → 推送远端 → GitHub API 创建 PR |
| `src/cli.py` | `heal / index / evaluate / serve / mcp` 命令行入口 |
| `Dockerfile`、`Dockerfile.sandbox` | API 服务镜像 + 沙箱底座镜像（`make sandbox-image`） |
| `requirements*.txt` | 非 Poetry 环境的可复现安装（`requirements-eval.txt` 为评测可选依赖） |

### 已知设计偏差（如实说明）

| 偏差 | 现状与理由 |
| --- | --- |
| Python 解析基于 stdlib `ast` 而非 tree-sitter | 遵循文档《避坑指南》第 1 条「第一阶段 100% 聚焦 Python」；tree-sitter 以策略模式实现为 TS/JS/Go 扩展位（`TreeSitterParser`，可提取签名与紧邻注释），grammar 缺失时静默跳过 |
| LSP 桥接为一次性 Pyright CLI 调用（附 `ast` 语法兜底），未保持常驻 LSP 进程 | 一次性调用在离线/CI 下更稳定；常驻进程 + JSON-RPC 为后续优化项 |
| Docker 沙箱执行使用 `exec_run` 线程封装（非流式管道），OOM 监控依赖容器 cgroups 限额 | 本地子进程后端提供完整的流式读取 + RSS 监控；容器后端由 Docker 自身限制兜底 |
| 补丁生成以「单个最可疑业务文件」为目标，逐候选变异尝试 | 先定位再生成的多候选策略保证离线可解释性与成功率；跨文件联合补丁为路线图项 |
| PR 自动推送仅在 Webhook 流程触发（CLI / 任务 API 只产出补丁） | 防止对非目标仓库误推远端；详见下文「PR 自动提交」 |
| PageRank 核心度在不含 scipy 的环境使用内置幂迭代 | `networkx.pagerank` 自 3.x 起内部固定走 scipy 实现，而 scipy 并非本项目依赖；缺失时自动降级为等价的纯 Python 幂迭代（`src/indexer/call_graph.py`），保证干净环境可直接建图 |

## 快速开始

```bash
# 1. 安装依赖（推荐 Poetry）
poetry install
# 或使用 pip
pip install -r requirements.txt

# 2. 准备环境变量（可选；缺少 .env 时使用默认离线配置）
cp .env.example .env

# 3. 启动基础设施（Qdrant / Redis / Postgres；全部可选）
docker compose up -d
make sandbox-image      # 可选：构建 Docker 沙箱底座镜像

# 4. 运行测试
make test

# 5. 启动 API 网关
make run
# → http://localhost:8000/docs
```

### 常用命令

```bash
python -m src.cli heal    --repo <repo> --issue "..."          # 端到端自愈
python -m src.cli index   --repo <repo> --query "add"          # 索引 + 检索验证
python -m src.cli evaluate --repo <repo> --limit 50 --attempts 3   # Pass@k 跑分
python -m src.cli serve                                        # API 网关
python -m src.cli mcp     --repo <repo>                        # MCP stdio 服务端
make package                                                   # 构建 wheel 包
```

### 无外部依赖运行（离线模式）

平台所有重型依赖（Qdrant / Docker / Postgres / LLM）均通过配置开关解耦，
在离线模式下自动降级为本地实现（内存检索、子进程沙箱、内存 Checkpointer、
启发式 LLM），可直接跑通端到端闭环与单元测试：

```bash
make demo        # 对一个内置示例仓库跑通「定位 → 规划 → 改码 → 校验」闭环
make test        # 运行全部单元测试
```

> **Docker 后端冒烟**：`make sandbox-image` 构建镜像后运行
> `python -m pytest tests/test_docker_smoke.py`，可真实验证「宿主跑 API + 沙箱用 docker」
> 组合下容器内 pytest 执行链路（CI 的 `docker-smoke` job 自动执行）。

> **提示**：`examples/sample_repo` 默认以普通目录形式提供。首次对该仓库执行
> `make demo` / `make index` 时，工作流会检测到其并非 Git 仓库，并**自动初始化
> 一个基线提交**（仅此一次，绝不改动已有仓库历史），随后在隔离的 Git Worktree
> 中完成修复与验证，源仓库始终保持干净。你也可以提前自行 `git init`，
> 平台会直接复用现有仓库。

### PR 自动提交（Webhook 无人值守闭环）

**触发路径**：GitHub Issue / PR Review 经 `POST /webhooks/github` 触发的任务，
在修复成功且 `NEXUS_GITHUB__AUTO_PR=true` 时，由 `src/mcp_tools/pr_publisher.py`
在临时 Worktree 中创建 `nexus/fix-*` 分支、提交补丁、推送 `origin`，并通过
GitHub API 创建 Pull Request；MCP 工具 `publish_patch` 共享同一能力。
CLI `heal` 与 `POST /api/v1/tasks` **不自动推送**（仅产出 `final_diff`）。
逐步降级策略保证任何一步失败都不阻塞主流程：

| 条件 | 行为 |
| --- | --- |
| 无远端 `origin` | 仅导出补丁（`final_diff`），跳过推送与 PR |
| 推送失败（网络/权限） | 返回失败原因，保留补丁 |
| 缺少 `NEXUS_GITHUB__TOKEN` | 推送分支成功，跳过 PR 创建 |
| 配置齐全 | 推送分支 + 创建 PR，Webhook 自动回评 PR 链接 |

相关配置（详见 `.env.example`）：`NEXUS_GITHUB__TOKEN`、`NEXUS_GITHUB__AUTO_PR`、
`NEXUS_GITHUB__REPO_ROOT`。GitHub Issue 创建 / PR Review 评论（`/nexus fix` 指令）
会经 `POST /webhooks/github` 触发完整闭环。

### 安全提示（部署到共享网络前必读）

| 配置 | 说明 |
| --- | --- |
| `NEXUS_API__HOST` | 默认 `127.0.0.1`（仅本机）；容器 / 局域网部署需显式改为 `0.0.0.0` 并配置鉴权 |
| `NEXUS_API__AUTH_TOKEN` | `NEXUS_ENV=prod` 时未配置直接**拒绝启动**（fail-fast）；dev 留空则接口无鉴权 |
| `NEXUS_API__ALLOWED_REPO_ROOTS` | `NEXUS_ENV=prod` 时未配置直接**拒绝启动**；它限制 `POST /api/v1/tasks` 可接受的 `repo_path`（否则可对任意目录 git init / 执行测试） |
| `NEXUS_GITHUB__REPO_ROOT` | Webhook 定位本地仓库的唯一依据；未配置时拒绝派发（不再回退 CWD 猜测） |
| `NEXUS_GITHUB__WEBHOOK_SECRET` | 未配置时 Webhook fail-closed；仅联调可临时开启 `NEXUS_GITHUB__ALLOW_UNVERIFIED_WEBHOOKS=true`（不安全） |
| `NEXUS_SANDBOX__BACKEND` | `NEXUS_ENV=prod` 时默认要求 `docker`（被测代码在容器内执行）；`local` 意味着被测仓库的 pytest 以**宿主权限**直接运行，需显式设置 `NEXUS_SANDBOX__ALLOW_LOCAL_IN_PROD=true` 才放行 |
| `docker compose` 端口发布 | `api` 服务默认只发布到宿主 `127.0.0.1:8000`（`NEXUS_API_BIND` 可覆盖）；需局域网 / 公网访问时显式设 `NEXUS_API_BIND=0.0.0.0`，并同步配置 `NEXUS_API__AUTH_TOKEN` 与 `NEXUS_API__ALLOWED_REPO_ROOTS` |
| Docker 构建上下文 | 仓库根目录的 `.dockerignore` 会排除 `.env`、`.git`、`.venv` 等；**请勿删除**，否则 `COPY . .` 会把密钥固化进镜像层 |

### Time-Travel 状态回溯

配置 `NEXUS_POSTGRES__ENABLED=true` 时使用 `AsyncPostgresSaver` 持久化状态；
离线时自动降级为进程内 Checkpointer。每个任务按「任务 ID + 尝试轮数」隔离
Checkpoint 线程：

* `GET /api/v1/tasks/{id}/checkpoints` —— 回溯每个节点快照（最新在前）；
* `POST /api/v1/tasks/{id}/intervention` —— 注入人工反馈并携带反馈开启新一轮
  （新线程，历史不受污染）。

### Pass@k 评测

`evaluate --attempts k` 时每道题至多尝试 k 轮（成功早停、失败重试），
报表自动生成 **Pass@k 曲线**、平均尝试轮数，并内嵌 Mermaid `radar-beta`
雷达图（附终端 ASCII 降级视图）。

#### 受控缺陷基准实测（DeepSeek · cerberus）

为给出可复现的端到端能力数据，在真实开源仓库
[pyeve/cerberus](https://github.com/pyeve/cerberus) `1.3.x`（246 个真实单元测试）上，
用**变异测试**方法构造缺陷集：自动注入 90 个单行算子变异，仅保留「能被仓库自带测试
真实捕获」的 15 个（每个缺陷都有明确的失败测试，等价 SWE-bench 的 FAIL_TO_PASS），
再由智能体在隔离 worktree 中独立定位修复，沙箱跑**全量 246 个测试**判定
（比官方只比对 FAIL_TO_PASS 更严格）。脚本、缺陷集与逐题结果见
[`benchmarks/`](benchmarks/)。

| 指标 | 数值 |
| --- | --- |
| **Pass@1**（首次采样修复率） | **60.0%（9/15）** |
| **Pass@2**（两次采样内修复率） | **73.3%（11/15）** |
| 平均工作流迭代 | 4.67 轮 |
| 单题平均 Token | 5,940 |
| 单题平均全流程耗时 | 47.7s |
| 总 Token / API 成本 | 89,098 / 约 ¥0.3 |
| 成功补丁触碰测试文件 | 0 / 11（无「改测试作弊」） |

模型 `deepseek-chat`，2026-10-17 跑分；完整报表（含雷达图、耗时分布、逐题补丁）见
[cerberus-report.md](benchmarks/artifacts/cerberus-report.md)。
其中 2 个缺陷首轮失败、次轮成功，体现了多轮重试的增益。

> **方法学边界（如实说明）**：上述为**受控变异基准**（mutation-based，与 Defects4J
> 同类思路），用于在固定预算内验证「定位 → 修复 → 验证」闭环；它**不是** SWE-bench
> 官方 harness（官方需按 `base_commit` 重建 per-instance 环境并套用 `test_patch`）。
> 接入官方 SWE-bench Verified harness 已列为 roadmap。

## 文档

本 README 是仓库内唯一的架构与使用说明（仓库未随附独立设计文档）：
模块划分见上文「目录结构」，状态机见 `src/agent/workflow.py`，
各模块的设计取舍与已知偏差写在其 docstring 与本文对应小节中。
