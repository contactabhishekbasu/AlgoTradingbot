#!/bin/bash
# AlgoTradingbot System Startup Script
# Starts all required services and performs health checks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}           AlgoTradingbot System Startup${NC}"
echo -e "${BLUE}================================================================${NC}\n"

# Function to print step headers
print_step() {
    echo -e "\n${BLUE}→ $1${NC}"
    echo "----------------------------------------------------------------"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if .env file exists
print_step "Step 1: Checking configuration"
if [ -f .env ]; then
    print_success ".env file found"
    source .env
else
    print_warning ".env file not found, using .env.example as template"
    if [ -f .env.example ]; then
        cp .env.example .env
        print_warning "Created .env from .env.example"
        print_warning "Please edit .env and set POSTGRES_PASSWORD and API keys"
        echo ""
        read -p "Press Enter after updating .env file..."
    else
        print_error ".env.example not found"
        exit 1
    fi
fi

# Check if Docker is running
print_step "Step 2: Checking Docker"
if docker info > /dev/null 2>&1; then
    print_success "Docker is running"
else
    print_error "Docker is not running"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

# Start Docker Compose services
print_step "Step 3: Starting Docker services"
echo "Starting PostgreSQL and Redis..."
docker-compose up -d

# Wait for services to be healthy
print_step "Step 4: Waiting for services to be ready"
echo "This may take 10-30 seconds..."

max_wait=60
wait_time=0
postgres_ready=false
redis_ready=false

while [ $wait_time -lt $max_wait ]; do
    # Check PostgreSQL
    if ! $postgres_ready && docker-compose exec -T postgres pg_isready -U trading_user > /dev/null 2>&1; then
        postgres_ready=true
        print_success "PostgreSQL is ready"
    fi

    # Check Redis
    if ! $redis_ready && docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        redis_ready=true
        print_success "Redis is ready"
    fi

    # Break if both are ready
    if $postgres_ready && $redis_ready; then
        break
    fi

    sleep 2
    wait_time=$((wait_time + 2))
done

if ! $postgres_ready || ! $redis_ready; then
    print_error "Services did not become ready in time"
    echo "Check logs with: docker-compose logs"
    exit 1
fi

# Check if database is initialized
print_step "Step 5: Checking database"
if docker-compose exec -T postgres psql -U trading_user -d trading -c "SELECT 1 FROM configuration LIMIT 1" > /dev/null 2>&1; then
    print_success "Database already initialized"
else
    print_warning "Database not initialized, running init script..."

    # Check if Python venv exists
    if [ -d "venv" ]; then
        print_success "Virtual environment found"
        source venv/bin/activate
    else
        print_warning "Virtual environment not found"
        echo "Create it with: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    fi

    # Run database init script
    if command -v python &> /dev/null; then
        python scripts/init_database.py
        if [ $? -eq 0 ]; then
            print_success "Database initialized successfully"
        else
            print_error "Database initialization failed"
            exit 1
        fi
    else
        print_error "Python not found in PATH"
        exit 1
    fi
fi

# Run health checks
print_step "Step 6: Running health checks"
if command -v python &> /dev/null; then
    # Activate venv if it exists
    if [ -d "venv" ] && [ -z "$VIRTUAL_ENV" ]; then
        source venv/bin/activate
    fi

    python scripts/health_check.py
    health_check_status=$?
else
    print_warning "Python not available, skipping detailed health checks"
    health_check_status=0
fi

# Show service information
print_step "Step 7: System information"
echo ""
echo "Service URLs:"
echo "  PostgreSQL:    localhost:${POSTGRES_PORT:-5432}"
echo "  Redis:         localhost:${REDIS_PORT:-6379}"
if docker-compose ps | grep -q "pgadmin"; then
    echo "  PgAdmin:       http://localhost:8080"
fi
if docker-compose ps | grep -q "redis-commander"; then
    echo "  Redis UI:      http://localhost:8081"
fi

echo ""
echo "Useful commands:"
echo "  Stop services:        docker-compose down"
echo "  View logs:            docker-compose logs -f"
echo "  Database shell:       docker-compose exec postgres psql -U trading_user -d trading"
echo "  Redis CLI:            docker-compose exec redis redis-cli"
echo "  Health check:         python scripts/health_check.py"
echo "  All commands:         make help"

# Final message
echo ""
echo -e "${BLUE}================================================================${NC}"
if [ $health_check_status -eq 0 ]; then
    echo -e "${GREEN}✓ AlgoTradingbot system started successfully!${NC}"
    echo -e "${GREEN}  All services are running and healthy.${NC}"
else
    echo -e "${YELLOW}⚠ AlgoTradingbot system started with warnings${NC}"
    echo -e "${YELLOW}  Some health checks failed. Review the output above.${NC}"
fi
echo -e "${BLUE}================================================================${NC}"
echo ""

exit $health_check_status
