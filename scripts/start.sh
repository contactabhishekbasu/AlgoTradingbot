#!/bin/bash
# Startup script for AI Trading System
# Phase 1: Data Foundation

set -e

echo "🚀 Starting AI Trading System - Phase 1: Data Foundation"
echo "=========================================================="

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Load environment variables
if [ -f .env ]; then
    echo -e "${GREEN}✓${NC} Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo -e "${YELLOW}⚠${NC}  No .env file found. Using defaults."
    echo "   Create .env from .env.example: cp .env.example .env"
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}✗${NC} Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo -e "${GREEN}✓${NC} Docker is running"

# Start Docker containers
echo ""
echo "📦 Starting PostgreSQL and Redis containers..."
docker-compose up -d

# Wait for services to be ready
echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check PostgreSQL
echo -n "   PostgreSQL: "
if docker-compose exec -T postgres pg_isready -U ${POSTGRES_USER:-trading_user} > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ready${NC}"
else
    echo -e "${RED}✗ Not ready${NC}"
    exit 1
fi

# Check Redis
echo -n "   Redis: "
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ready${NC}"
else
    echo -e "${RED}✗ Not ready${NC}"
    exit 1
fi

# Initialize database if needed
echo ""
echo "🗄️  Checking database initialization..."
if docker-compose exec -T postgres psql -U ${POSTGRES_USER:-trading_user} -d ${POSTGRES_DB:-trading} -c "SELECT 1 FROM system_logs LIMIT 1;" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Database already initialized"
else
    echo "   Initializing database schema..."
    docker-compose exec -T postgres psql -U ${POSTGRES_USER:-trading_user} -d ${POSTGRES_DB:-trading} -f /docker-entrypoint-initdb.d/init.sql
    echo -e "${GREEN}✓${NC} Database initialized"
fi

# Health check
echo ""
echo "🏥 Running health checks..."

# Test database connection
echo -n "   Database connection: "
if docker-compose exec -T postgres psql -U ${POSTGRES_USER:-trading_user} -d ${POSTGRES_DB:-trading} -c "SELECT 1;" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ OK${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
fi

# Test Redis connection
echo -n "   Redis connection: "
if docker-compose exec -T redis redis-cli ping | grep -q PONG; then
    echo -e "${GREEN}✓ OK${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
fi

# Show running containers
echo ""
echo "📋 Running containers:"
docker-compose ps

echo ""
echo -e "${GREEN}✅ System started successfully!${NC}"
echo ""
echo "Next steps:"
echo "  1. Install Python dependencies: pip install -r requirements.txt"
echo "  2. Run tests: pytest"
echo "  3. Start MCP server: python -m src.mcp_servers.yfinance_trader_mcp"
echo ""
echo "To stop the system:"
echo "  docker-compose down"
echo ""
