# AlgoTradingbot Setup Guide

Complete setup guide for the Claude-Powered AI Trading System

**Version:** 0.1.0
**Date:** October 30, 2025
**Status:** Phase 0 - Preparation Complete

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Detailed Setup](#detailed-setup)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)
6. [Next Steps](#next-steps)

---

## Prerequisites

### Hardware Requirements

- **MacBook M4** (Pro or Max recommended)
- **RAM:** 16GB minimum (32GB recommended)
- **Storage:** 50GB free disk space
- **Internet:** Stable broadband connection (10 Mbps minimum)

### Software Requirements

#### Required Software

1. **macOS Sonoma 14.0+**
2. **Python 3.11+**
   ```bash
   python3 --version  # Should show 3.11.0 or higher
   ```

3. **Docker Desktop for Mac** (Apple Silicon version)
   - Download: https://www.docker.com/products/docker-desktop
   - Version: 4.25+

4. **Git**
   ```bash
   git --version
   ```

5. **Claude Desktop App**
   - Download: https://claude.ai/desktop
   - Required for natural language interface

#### Optional Software (Recommended)

- **Homebrew** (package manager for macOS)
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```

- **Visual Studio Code** or **PyCharm**
- **iTerm2** (better terminal)

---

## Quick Start

For experienced developers who want to get started immediately:

```bash
# 1. Clone repository
git clone https://github.com/contactabhishekbasu/AlgoTradingbot.git
cd AlgoTradingbot

# 2. Create environment file
cp .env.example .env
# Edit .env with your configuration (especially POSTGRES_PASSWORD)

# 3. Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .

# 4. Start Docker services
docker-compose up -d

# 5. Initialize database
python scripts/init_database.py

# 6. Verify installation
make health-check

# 7. Run tests
make test
```

---

## Detailed Setup

### Step 1: Install Prerequisites

#### 1.1 Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### 1.2 Install Python 3.11+

```bash
# Using Homebrew
brew install python@3.11

# Verify installation
python3 --version
```

#### 1.3 Install Docker Desktop

1. Download Docker Desktop for Mac (Apple Silicon): https://www.docker.com/products/docker-desktop
2. Install by dragging to Applications folder
3. Launch Docker Desktop
4. Ensure Docker is running:
   ```bash
   docker --version
   docker-compose --version
   ```

#### 1.4 Install Claude Desktop

1. Visit: https://claude.ai/desktop
2. Download macOS version
3. Install and sign in with your Anthropic account

---

### Step 2: Clone Repository

```bash
# Navigate to your projects directory
cd ~/Projects  # or wherever you keep projects

# Clone the repository
git clone https://github.com/contactabhishekbasu/AlgoTradingbot.git

# Enter project directory
cd AlgoTradingbot

# Verify you're on the correct branch
git branch
```

---

### Step 3: Python Environment Setup

#### 3.1 Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

To deactivate later: `deactivate`

#### 3.2 Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

#### 3.3 Install Dependencies

```bash
# Install development dependencies (includes all requirements)
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .

# Verify key packages
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import xgboost as xgb; print(f'XGBoost: {xgb.__version__}')"
python -c "import pandas as pd; print(f'Pandas: {pd.__version__}')"
```

#### 3.4 Install TA-Lib (Optional but Recommended)

TA-Lib requires C library installation:

```bash
# Install TA-Lib C library
brew install ta-lib

# Install Python wrapper
pip install TA-Lib
```

If you encounter issues, pandas-ta is included as a pure-Python alternative.

---

### Step 4: Configuration

#### 4.1 Create Environment File

```bash
# Copy example environment file
cp .env.example .env

# Open in editor
code .env  # or nano .env, or vim .env
```

#### 4.2 Configure Essential Variables

**REQUIRED - Must set these:**

```env
# Database password (REQUIRED!)
POSTGRES_PASSWORD=your_secure_password_here

# Alpaca API keys (for paper trading)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
```

**To get Alpaca API keys:**
1. Sign up at https://alpaca.markets
2. Go to Paper Trading dashboard
3. Generate API keys
4. Copy keys to .env file

**OPTIONAL - Recommended defaults:**

The .env.example file has sensible defaults for development. You typically only need to change:
- POSTGRES_PASSWORD
- ALPACA_API_KEY
- ALPACA_SECRET_KEY

---

### Step 5: Docker Setup

#### 5.1 Start Docker Services

```bash
# Start PostgreSQL and Redis (production mode)
docker-compose up -d

# OR start with development tools (PgAdmin, Redis Commander)
docker-compose --profile dev up -d
```

#### 5.2 Verify Services are Running

```bash
# Check service status
docker-compose ps

# Should show:
# - trading_postgres (healthy)
# - trading_redis (healthy)

# View logs
docker-compose logs -f
```

#### 5.3 Test Database Connection

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U trading_user -d trading

# You should see PostgreSQL prompt
# Type \q to exit

# Test Redis
docker-compose exec redis redis-cli ping
# Should return: PONG
```

---

### Step 6: Database Initialization

#### 6.1 Run Database Setup Script

```bash
# Make sure you're in project root and venv is activated
python scripts/init_database.py
```

Expected output:
```
==================================================
AlgoTradingbot Database Initialization
==================================================

Database Configuration:
  Host: localhost
  Port: 5432
  User: trading_user
  Database: trading

==================================================
Step 1: Creating database
==================================================
✓ Database 'trading' created successfully

==================================================
Step 2: Running migrations
==================================================
Running migration: 001_initial_schema.sql
✓ Migration 001_initial_schema.sql completed successfully

==================================================
Step 3: Verifying installation
==================================================
Verifying tables...
  ✓ market_data
  ✓ technical_indicators
  ✓ model_states
  ✓ predictions
  ✓ backtest_results
  ✓ backtest_trades
  ✓ system_logs
  ✓ configuration

==================================================
✓ Database initialization completed successfully!
==================================================
```

#### 6.2 Verify Database Schema

```bash
# Connect to database
docker-compose exec postgres psql -U trading_user -d trading

# List tables
\dt

# Describe a table
\d market_data

# View configuration
SELECT * FROM configuration;

# Exit
\q
```

---

### Step 7: Install Pre-commit Hooks

```bash
# Install pre-commit hooks (for code quality)
pre-commit install

# Test pre-commit hooks
pre-commit run --all-files
```

This will run code formatters and linters before each commit.

---

### Step 8: Configure Claude Desktop

#### 8.1 Locate Claude Desktop Config

```bash
# Config file location
~/.config/claude/claude_desktop_config.json
```

#### 8.2 Add MCP Server Configuration

Edit the file and add (or create if it doesn't exist):

```json
{
  "mcpServers": {
    "yfinance-trader": {
      "command": "python",
      "args": ["/full/path/to/AlgoTradingbot/src/mcp_servers/yfinance_trader_mcp.py"],
      "env": {
        "REDIS_URL": "redis://localhost:6379",
        "POSTGRES_URL": "postgresql://trading_user:your_password@localhost:5432/trading"
      }
    }
  }
}
```

**Note:** Replace `/full/path/to/AlgoTradingbot` with actual path. Get it with:
```bash
pwd  # When in AlgoTradingbot directory
```

---

## Verification

### Run Health Checks

```bash
# Comprehensive health check
make health-check

# Expected output:
# ✓ Python environment OK
# ✓ PostgreSQL connection OK
# ✓ Redis connection OK
# ✓ Required packages installed
# ✓ Docker services running
```

### Run Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
```

### Test Makefile Commands

```bash
# Show available commands
make help

# Check formatting
make format-check

# Run linters
make lint
```

### Verify Directory Structure

```bash
# Should see all these directories
tree -L 2 -d
```

Expected structure:
```
.
├── config
├── data
│   ├── processed
│   └── raw
├── docs
├── logs
├── models
├── scripts
├── sql
│   └── migrations
├── src
│   ├── backtesting
│   ├── cache
│   ├── data
│   ├── indicators
│   ├── mcp_servers
│   ├── ml
│   ├── streaming
│   └── utils
└── tests
    ├── e2e
    ├── integration
    └── unit
```

---

## Troubleshooting

### Common Issues

#### 1. Docker Services Won't Start

**Problem:** `docker-compose up -d` fails

**Solutions:**
```bash
# Check if Docker Desktop is running
docker info

# Restart Docker Desktop
# (Applications > Docker > Restart)

# Remove old volumes and try again
docker-compose down -v
docker-compose up -d
```

#### 2. PostgreSQL Connection Refused

**Problem:** Can't connect to PostgreSQL

**Solutions:**
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Check logs
docker-compose logs postgres

# Verify password in .env matches docker-compose
cat .env | grep POSTGRES_PASSWORD

# Restart PostgreSQL
docker-compose restart postgres
```

#### 3. Python Package Installation Fails

**Problem:** `pip install` fails for some packages

**Solutions:**
```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install packages one by one to identify problem
pip install tensorflow
pip install xgboost

# Check Python version
python --version  # Must be 3.11+

# For TA-Lib issues
brew install ta-lib
pip install TA-Lib
```

#### 4. Permission Denied Errors

**Problem:** Permission denied when running scripts

**Solutions:**
```bash
# Make script executable
chmod +x scripts/init_database.py
chmod +x scripts/*.py

# Or run with python directly
python scripts/init_database.py
```

#### 5. TensorFlow Metal Issues (M4)

**Problem:** TensorFlow not using GPU

**Solutions:**
```bash
# Verify tensorflow-metal is installed
pip install tensorflow-metal

# Check if MPS is available
python -c "import tensorflow as tf; print('MPS available:', tf.config.list_physical_devices('GPU'))"

# If issues persist, uninstall and reinstall
pip uninstall tensorflow tensorflow-metal
pip install tensorflow tensorflow-metal
```

#### 6. Port Already in Use

**Problem:** Port 5432 or 6379 already in use

**Solutions:**
```bash
# Find what's using the port
lsof -i :5432
lsof -i :6379

# Either stop that process or change ports in .env
POSTGRES_PORT=5433
REDIS_PORT=6380

# Restart Docker services
docker-compose down
docker-compose up -d
```

---

## Next Steps

### Phase 1: Data Foundation (Weeks 2-3)

Now that Phase 0 is complete, move on to:

1. **Implement YFinance Client** (`src/data/yfinance_client.py`)
2. **Implement Technical Indicators** (`src/indicators/technical_indicators.py`)
3. **Create First MCP Server** (`src/mcp_servers/yfinance_trader_mcp.py`)

### Development Workflow

```bash
# 1. Start development session
cd AlgoTradingbot
source venv/bin/activate
docker-compose up -d

# 2. Make changes to code

# 3. Run tests
make test

# 4. Format and lint
make format
make lint

# 5. Commit changes
git add .
git commit -m "Descriptive message"
git push

# 6. End session
docker-compose down
deactivate
```

### Useful Commands

```bash
# See all available make commands
make help

# Start Jupyter notebook for exploration
make notebook

# Connect to database
make db-shell

# Connect to Redis
make redis-cli

# View Docker logs
make docker-logs

# Clean up everything
make clean
```

### Learning Resources

- **PRD.md** - Full product requirements
- **ARCHITECTURE.md** - System architecture
- **DEVELOPMENT.md** - Development roadmap
- **research.md** - Academic foundation

---

## Support

### Getting Help

1. **Documentation**: Check PRD.md, ARCHITECTURE.md, DEVELOPMENT.md
2. **Logs**: `docker-compose logs` and `logs/` directory
3. **GitHub Issues**: https://github.com/contactabhishekbasu/AlgoTradingbot/issues

### Reporting Bugs

When reporting issues, include:
- Python version (`python --version`)
- macOS version
- Docker version
- Error messages and logs
- Steps to reproduce

---

## Appendix

### A. Useful Commands Cheat Sheet

```bash
# Docker
docker-compose up -d              # Start services
docker-compose down               # Stop services
docker-compose ps                 # List services
docker-compose logs -f            # View logs

# Database
make db-shell                     # PostgreSQL shell
make db-init                      # Initialize database

# Python
source venv/bin/activate          # Activate venv
deactivate                        # Deactivate venv
pip list                          # List packages

# Testing
make test                         # All tests
make test-unit                    # Unit tests only
make lint                         # Run linters
make format                       # Format code

# Cleanup
make clean                        # Clean Python artifacts
make docker-clean                 # Clean Docker (WARNING: removes data)
```

### B. Environment Variables Reference

See `.env.example` for complete list of environment variables with descriptions.

### C. Directory Structure

```
AlgoTradingbot/
├── src/                    # Source code
│   ├── data/              # Data fetching and storage
│   ├── indicators/        # Technical indicators
│   ├── ml/                # Machine learning models
│   ├── backtesting/       # Backtesting engine
│   ├── mcp_servers/       # MCP servers
│   └── ...
├── tests/                 # Test files
├── docs/                  # Documentation
├── sql/                   # Database migrations
├── scripts/               # Utility scripts
├── data/                  # Data storage (not in git)
├── models/                # Trained models (not in git)
├── logs/                  # Log files (not in git)
└── config/                # Configuration files
```

---

**Setup Complete!** 🎉

You're now ready to begin Phase 1: Data Foundation.

For next steps, see: [DEVELOPMENT.md](../DEVELOPMENT.md#phase-1-data-foundation-weeks-2-3)
