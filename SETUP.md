# Setup Guide - Phase 1: Data Foundation

Quick setup guide for the AI Trading System Phase 1 implementation.

## Prerequisites

- **macOS** (M4 chip recommended, but any Mac will work)
- **Python 3.11+**
- **Docker Desktop** (for PostgreSQL and Redis)
- **Git**

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/contactabhishekbasu/AlgoTradingbot.git
cd AlgoTradingbot
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
.\venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (use your preferred editor)
nano .env
```

**Key settings to configure in `.env`**:
```bash
# Database (defaults are fine for local development)
POSTGRES_PASSWORD=your_secure_password

# API Keys (optional for Phase 1, required for live trading later)
# ALPACA_API_KEY=your_key
# ALPACA_SECRET_KEY=your_secret
```

### 4. Start Services

```bash
# Make startup script executable (first time only)
chmod +x scripts/start.sh

# Start PostgreSQL and Redis
./scripts/start.sh
```

This will:
- Start Docker containers for PostgreSQL and Redis
- Initialize the database schema
- Run health checks
- Display service status

### 5. Verify Installation

```bash
# Run tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=term-missing

# Check services are running
docker-compose ps
```

Expected output:
```
       Name                     Command                  State           Ports
---------------------------------------------------------------------------------
trading_postgres   docker-entrypoint.sh postgres    Up (healthy)   0.0.0.0:5432->5432/tcp
trading_redis      docker-entrypoint.sh redis...    Up (healthy)   0.0.0.0:6379->6379/tcp
```

## Usage Examples

### Test YFinance Client

```python
from src.data.yfinance_client import YFinanceClient

# Create client
client = YFinanceClient()

# Get current price
price = client.get_current_price("AAPL")
print(f"AAPL Price: ${price['price']}")

# Get historical data
data = client.get_historical_data("AAPL", period="1mo")
print(data.head())

# Check market status
status = client.get_market_status()
print(f"Market Open: {status['is_open']}")
```

### Calculate Technical Indicators

```python
from src.data.yfinance_client import YFinanceClient
from src.indicators.technical_indicators import TechnicalIndicators

# Fetch data
client = YFinanceClient()
data = client.get_historical_data("AAPL", period="3mo")

# Calculate indicators
indicators = TechnicalIndicators()
result = indicators.calculate_all(
    data,
    indicators=["rsi", "williams_r", "bbands", "macd"]
)

# Display latest values
print("\nLatest Indicators for AAPL:")
print(f"RSI: {result['RSI'].iloc[-1]:.2f}")
print(f"Williams %R: {result['Williams_R'].iloc[-1]:.2f}")
print(f"Bollinger Bands:")
print(f"  Upper: {result['BB_Upper'].iloc[-1]:.2f}")
print(f"  Middle: {result['BB_Middle'].iloc[-1]:.2f}")
print(f"  Lower: {result['BB_Lower'].iloc[-1]:.2f}")
```

### Test Database Operations

```python
from src.data.database import DatabaseClient
from src.data.yfinance_client import YFinanceClient

# Create clients
db = DatabaseClient()
yf = YFinanceClient()

# Test connection
if db.test_connection():
    print("✅ Database connection successful")

# Fetch and store data
data = yf.get_historical_data("AAPL", period="1mo")
rows = db.store_market_data(data, "AAPL")
print(f"Stored {rows} rows in database")

# Retrieve data
stored_data = db.fetch_market_data("AAPL", limit=10)
print(stored_data)
```

### Test Redis Caching

```python
from src.cache.redis_client import RedisCache, CacheKeys

# Create cache client
cache = RedisCache()

# Test connection
if cache.ping():
    print("✅ Redis connection successful")

# Set and get data
cache.set("test_key", {"price": 150.50}, ttl=60)
value = cache.get("test_key")
print(f"Cached value: {value}")

# Get cache statistics
stats = cache.get_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2f}%")
```

## Start MCP Server

```bash
# Start the YFinance Trader MCP server
python -m src.mcp_servers.yfinance_trader_mcp
```

The server will:
- Initialize all components
- Connect to PostgreSQL and Redis
- Expose 8 tool endpoints
- Wait for MCP requests via stdio

**Available Tools**:
1. `get_current_price` - Real-time price
2. `get_historical_data` - Historical OHLCV
3. `calculate_indicators` - Technical indicators
4. `get_market_status` - Market open/close
5. `bulk_fetch` - Multiple symbols
6. `get_rsi` - RSI with signals
7. `get_williams_r` - Williams %R with signals
8. `get_bollinger_bands` - Bollinger Bands with signals

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_yfinance_client.py -v

# Run with coverage
pytest --cov=src --cov-report=html

# Run only unit tests
pytest tests/unit/ -v

# Run with specific markers
pytest -m "not slow" -v
```

## Troubleshooting

### Docker Issues

**Problem**: Docker containers won't start
```bash
# Check Docker is running
docker info

# Restart Docker Desktop
# Then try again:
docker-compose down
docker-compose up -d
```

**Problem**: Port conflicts (5432 or 6379 already in use)
```bash
# Find process using port
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis

# Stop conflicting service or change port in docker-compose.yml
```

### Database Issues

**Problem**: Database not initialized
```bash
# Manually initialize database
docker-compose exec postgres psql -U trading_user -d trading -f /docker-entrypoint-initdb.d/init.sql
```

**Problem**: Connection refused
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check logs
docker-compose logs postgres
```

### Redis Issues

**Problem**: Redis connection failed
```bash
# Check Redis is running
docker-compose ps redis

# Test connection
docker-compose exec redis redis-cli ping
```

### Python Issues

**Problem**: Module not found
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**Problem**: Import errors
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Stopping Services

```bash
# Stop all containers
docker-compose down

# Stop and remove volumes (WARNING: deletes all data)
docker-compose down -v
```

## Next Steps

1. ✅ **Phase 1 Complete** - Data Foundation
2. ⏳ **Phase 2 Next** - ML Core Implementation
   - Feature engineering pipeline
   - LSTM model implementation
   - XGBoost model implementation
   - Ensemble logic
   - Online learning

## Support

For issues or questions:
1. Check [PHASE1_COMPLETE.md](PHASE1_COMPLETE.md) for detailed documentation
2. Review [ARCHITECTURE.md](ARCHITECTURE.md) for system design
3. See [DEVELOPMENT.md](DEVELOPMENT.md) for development roadmap
4. Create an issue on GitHub

## Additional Resources

- **YFinance Documentation**: https://pypi.org/project/yfinance/
- **Model Context Protocol**: https://modelcontextprotocol.io
- **PostgreSQL Docs**: https://www.postgresql.org/docs/
- **Redis Docs**: https://redis.io/docs/
- **Pandas Documentation**: https://pandas.pydata.org/docs/

---

**Phase 1 Status**: ✅ COMPLETE
**Last Updated**: October 30, 2025
