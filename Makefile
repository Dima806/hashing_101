.PHONY: help setup sync lint format check typecheck test test-cov \
        notebooks run lab clean reset ci dev

.DEFAULT_GOAL := help

# Lint whatever exists: tests/ and app/ arrive in later build phases, and ruff
# errors out on a path that is not there yet.
LINT_PATHS := $(wildcard src tests app)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## First-time setup
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync --all-extras
	uv run python -m ipykernel install --user --name hashing-101
	@printf "\n✅ Ready. Run 'make test'.\n"

sync: ## Sync deps
	uv sync --all-extras

lint: format check typecheck ## All linters
format: ## ruff format
	uv run ruff format $(LINT_PATHS)
check: ## ruff check
	uv run ruff check --fix $(LINT_PATHS)
typecheck: ## ty check
	uv run ty check src/

test: ## pytest
	uv run pytest
test-cov: ## pytest with coverage
	uv run pytest --cov=src --cov-report=term-missing

notebooks: ## Execute all notebooks in place
	@for nb in notebooks/0*.ipynb; do \
		echo "▶ Executing $$nb ..."; \
		uv run jupyter nbconvert --to notebook --execute --inplace \
			--ExecutePreprocessor.timeout=240 "$$nb" || exit 1; \
	done
	@printf "\n✅ All notebooks executed.\n"

run: ## Streamlit app
	uv run streamlit run app/streamlit_app.py --server.port 8501
lab: ## JupyterLab
	uv run jupyter lab --no-browser --port 8888

clean: ## Clean
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache htmlcov .coverage .pytest_cache .ruff_cache
	@printf "🧹 Cleaned.\n"
reset: clean ## Full reset
	rm -rf .venv

ci: sync lint test ## CI
dev: lint test ## Fast loop
