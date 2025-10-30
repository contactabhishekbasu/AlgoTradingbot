.PHONY: help install dev-install clean test lint format docker-up docker-down docker-restart docker-logs docker-clean setup health-check

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy

# Colors for terminal output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)AlgoTradingbot - AI-Powered Trading System$(NC)"
	@echo ""
	@echo "$(GREEN)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# Installation commands
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Production dependencies installed$(NC)"

dev-install: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	pre-commit install
	@echo "$(GREEN)✓ Development environment ready$(NC)"

setup: ## Initial project setup
	@echo "$(BLUE)Setting up AlgoTradingbot project...$(NC)"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(YELLOW)⚠ Created .env file from .env.example$(NC)"; \
		echo "$(YELLOW)⚠ Please update .env with your configuration$(NC)"; \
	fi
	@mkdir -p logs data/raw data/processed models
	@echo "$(GREEN)✓ Project setup complete$(NC)"

# Docker commands
docker-up: ## Start all Docker services
	@echo "$(BLUE)Starting Docker services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Docker services started$(NC)"
	@echo "$(YELLOW)Run 'make docker-logs' to view logs$(NC)"

docker-up-dev: ## Start Docker services with development tools
	@echo "$(BLUE)Starting Docker services (dev mode)...$(NC)"
	docker-compose --profile dev up -d
	@echo "$(GREEN)✓ Docker services started with dev tools$(NC)"
	@echo "$(YELLOW)PgAdmin: http://localhost:8080$(NC)"
	@echo "$(YELLOW)Redis Commander: http://localhost:8081$(NC)"

docker-down: ## Stop all Docker services
	@echo "$(BLUE)Stopping Docker services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Docker services stopped$(NC)"

docker-restart: ## Restart all Docker services
	@echo "$(BLUE)Restarting Docker services...$(NC)"
	docker-compose restart
	@echo "$(GREEN)✓ Docker services restarted$(NC)"

docker-logs: ## View Docker logs
	docker-compose logs -f

docker-clean: ## Remove all Docker containers, volumes, and images
	@echo "$(RED)⚠ This will remove all Docker containers, volumes, and images!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v --rmi all; \
		echo "$(GREEN)✓ Docker cleaned$(NC)"; \
	fi

docker-status: ## Show status of Docker services
	@docker-compose ps

# Database commands
db-init: ## Initialize database schema
	@echo "$(BLUE)Initializing database...$(NC)"
	$(PYTHON) scripts/init_database.py
	@echo "$(GREEN)✓ Database initialized$(NC)"

db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	alembic upgrade head
	@echo "$(GREEN)✓ Migrations applied$(NC)"

db-shell: ## Connect to PostgreSQL shell
	docker-compose exec postgres psql -U trading_user -d trading

redis-cli: ## Connect to Redis CLI
	docker-compose exec redis redis-cli

# Testing commands
test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTEST) tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Tests completed$(NC)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTEST) tests/unit/ -v
	@echo "$(GREEN)✓ Unit tests completed$(NC)"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTEST) tests/integration/ -v
	@echo "$(GREEN)✓ Integration tests completed$(NC)"

test-e2e: ## Run end-to-end tests only
	@echo "$(BLUE)Running E2E tests...$(NC)"
	$(PYTEST) tests/e2e/ -v
	@echo "$(GREEN)✓ E2E tests completed$(NC)"

test-watch: ## Run tests in watch mode
	$(PYTEST) tests/ -v --looponfail

# Code quality commands
lint: ## Run all linters
	@echo "$(BLUE)Running linters...$(NC)"
	$(FLAKE8) src/ tests/
	$(MYPY) src/
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	$(BLACK) src/ tests/
	$(ISORT) src/ tests/
	@echo "$(GREEN)✓ Code formatted$(NC)"

format-check: ## Check code formatting without making changes
	@echo "$(BLUE)Checking code format...$(NC)"
	$(BLACK) --check src/ tests/
	$(ISORT) --check-only src/ tests/
	@echo "$(GREEN)✓ Format check complete$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	$(MYPY) src/
	@echo "$(GREEN)✓ Type checking complete$(NC)"

# Utility commands
clean: ## Clean up generated files
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	rm -rf htmlcov/ .coverage build/ dist/
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

health-check: ## Run health checks on all services
	@echo "$(BLUE)Running health checks...$(NC)"
	@$(PYTHON) scripts/health_check.py

backtest: ## Run a backtest (requires arguments)
	@echo "$(BLUE)Running backtest...$(NC)"
	$(PYTHON) scripts/run_backtest.py $(ARGS)

train-model: ## Train ML models
	@echo "$(BLUE)Training models...$(NC)"
	$(PYTHON) scripts/train_models.py

load-data: ## Load historical market data
	@echo "$(BLUE)Loading market data...$(NC)"
	$(PYTHON) scripts/load_historical_data.py

# Development helpers
notebook: ## Start Jupyter notebook server
	@echo "$(BLUE)Starting Jupyter notebook...$(NC)"
	jupyter notebook --notebook-dir=notebooks

shell: ## Start Python shell with project context
	@$(PYTHON) -i -c "import sys; sys.path.insert(0, 'src'); print('AlgoTradingbot shell - src/ is in path')"

# CI/CD commands
ci: format-check lint test ## Run all CI checks

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Documentation commands
docs: ## Build documentation (future)
	@echo "$(YELLOW)Documentation build not yet implemented$(NC)"

docs-serve: ## Serve documentation locally (future)
	@echo "$(YELLOW)Documentation server not yet implemented$(NC)"
