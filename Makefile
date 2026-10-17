.PHONY: help install install-dev install-eval lint format typecheck test test-cov run demo index clean package sandbox-image docker-up docker-down

PY ?= python
PKG = src
DEMO_REPO ?= examples/sample_repo

help: ## 显示帮助
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## 安装运行依赖
	$(PY) -m pip install -r requirements.txt

install-dev: ## 安装开发依赖
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

install-eval: ## 安装评测平台可选依赖（HuggingFace SWE-bench）
	$(PY) -m pip install -r requirements-eval.txt

lint: ## Ruff 静态检查
	$(PY) -m ruff check $(PKG) tests

format: ## Ruff 自动格式化
	$(PY) -m ruff format $(PKG) tests
	$(PY) -m ruff check --fix $(PKG) tests

typecheck: ## mypy 类型检查
	$(PY) -m mypy $(PKG)

test: ## 运行单元测试
	$(PY) -m pytest

test-cov: ## 运行测试并生成覆盖率报告
	$(PY) -m pytest --cov=$(PKG) --cov-report=term-missing --cov-report=html

run: ## 启动 API 网关（默认仅监听 127.0.0.1，可用 NEXUS_API__HOST 覆盖）
	$(PY) -m src.cli serve

demo: ## 对示例仓库跑通端到端闭环
	$(PY) -m src.cli heal --repo $(DEMO_REPO) --issue "test_add 返回了错误的求和结果"

index: ## 索引示例仓库并输出检索结果
	$(PY) -m src.cli index --repo $(DEMO_REPO)

clean: ## 清理构建与缓存产物
	@rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist .nexus
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true

package: ## 构建 wheel 分发包到 dist/
	$(PY) -m pip wheel . --no-deps -w dist

sandbox-image: ## 构建沙箱基础镜像 nexus-sandbox:latest
	docker build -f Dockerfile.sandbox -t nexus-sandbox:latest .

docker-up: ## 启动基础设施
	docker compose up -d

docker-down: ## 停止基础设施
	docker compose down
