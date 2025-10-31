# AlgoTradingbot Validation Checklist

**Quick Reference Guide for System Validation**

---

## Quick Start

1. **Open Terminal** on your Mac
2. **Navigate to project:** `cd /path/to/AlgoTradingbot`
3. **Activate environment:** `source venv/bin/activate`
4. **Follow checklist below**

---

## ☑️ Pre-Validation Requirements

Before you start, verify you have:

- [ ] MacBook M4 (or compatible Mac)
- [ ] macOS 13+ installed
- [ ] Python 3.11+ installed
- [ ] Docker Desktop running
- [ ] Internet connection active
- [ ] At least 10GB free disk space
- [ ] At least 8GB RAM available

**Check Python:** `python3 --version` (should show 3.11 or higher)
**Check Docker:** `docker --version` (should return version number)

---

## Phase 1: System Setup (15-20 minutes)

### Step 1: Environment Setup
- [ ] Navigate to AlgoTradingbot directory
- [ ] Activate virtual environment: `source venv/bin/activate`
- [ ] Verify prompt shows `(venv)` prefix

### Step 2: Install Dependencies
- [ ] Run: `pip list | grep -E "pandas|numpy|tensorflow|xgboost|scikit-learn|yfinance"`
- [ ] Confirm all packages are listed
- [ ] If missing, run: `pip install -r requirements.txt`

### Step 3: Start Services
- [ ] Run: `docker-compose up -d`
- [ ] Wait for services to start (10-15 seconds)
- [ ] Verify: `docker-compose ps` shows both services "Up"

### Step 4: Health Check
- [ ] Run: `python scripts/health_check.py`
- [ ] Verify all checks show ✅
- [ ] Confirm "Overall Status: HEALTHY"

**✅ Phase 1 Complete:** All systems operational

---

## Phase 2: Data Layer (20-30 minutes)

### Step 5: API Connectivity
```bash
python -c "
from src.data.yfinance_client import YFinanceClient
import asyncio
async def test():
    client = YFinanceClient()
    price = await client.get_current_price('AAPL')
    print(f'✅ Price: \${price:.2f}')
    await client.close()
asyncio.run(test())
"
```
- [ ] Command completes successfully
- [ ] Returns current AAPL price
- [ ] No timeout or errors

### Step 6: Historical Data
```bash
python -c "
from src.data.yfinance_client import YFinanceClient
import asyncio
async def test():
    client = YFinanceClient()
    data = await client.get_historical_data('AAPL', period='5d')
    print(f'✅ Fetched {len(data)} days')
    await client.close()
asyncio.run(test())
"
```
- [ ] Fetches 5 days of data
- [ ] No missing data errors

### Step 7: Data Validation
```bash
python -c "
from src.data.yfinance_client import YFinanceClient
from src.data.validators import DataValidator
import asyncio
async def test():
    client = YFinanceClient()
    data = await client.get_historical_data('AAPL', period='1mo')
    validator = DataValidator()
    is_valid, errors = validator.validate_ohlcv(data)
    print('✅ Validation PASSED' if is_valid else '❌ Validation FAILED')
    await client.close()
asyncio.run(test())
"
```
- [ ] Data validation passes
- [ ] No quality issues reported

### Step 8: Technical Indicators
```bash
python -c "
from src.data.technical_indicators import TechnicalIndicators
from src.data.yfinance_client import YFinanceClient
import asyncio
async def test():
    client = YFinanceClient()
    data = await client.get_historical_data('AAPL', period='3mo')
    ti = TechnicalIndicators()
    rsi = ti.calculate_rsi(data['Close'])
    print(f'✅ RSI: {rsi.iloc[-1]:.2f}')
    await client.close()
asyncio.run(test())
"
```
- [ ] RSI calculated (value between 0-100)
- [ ] No calculation errors

### Step 9: Database Test
```bash
python -c "
from src.data.database import DatabaseManager
import asyncio
async def test():
    db = DatabaseManager()
    await db.connect()
    print('✅ Database connected')
    await db.close()
asyncio.run(test())
"
```
- [ ] Database connection successful
- [ ] No connection errors

### Step 10: Cache Test
```bash
python -c "
from src.data.cache import CacheManager
import asyncio
async def test():
    cache = CacheManager()
    await cache.connect()
    await cache.set('test', {'value': 123}, ttl=60)
    result = await cache.get('test')
    print('✅ Cache working' if result else '❌ Cache failed')
    await cache.close()
asyncio.run(test())
"
```
- [ ] Cache set/get working
- [ ] No Redis connection errors

**✅ Phase 2 Complete:** Data layer operational

---

## Phase 3: ML Models (30-45 minutes)

### Step 11: Feature Engineering
```bash
python -c "
from src.ml.feature_engineering import FeatureEngineer
from src.data.yfinance_client import YFinanceClient
import asyncio
async def test():
    client = YFinanceClient()
    data = await client.get_historical_data('AAPL', period='6mo')
    fe = FeatureEngineer()
    features = fe.create_features(data)
    print(f'✅ Features: {len(features.columns)} columns')
    await client.close()
asyncio.run(test())
"
```
- [ ] 50+ features created
- [ ] No missing values (NaN)

### Step 12: LSTM Training (3-5 minutes)
```bash
python -c "
from src.ml.models.lstm_model import LSTMModel
from src.ml.feature_engineering import FeatureEngineer
from src.data.yfinance_client import YFinanceClient
import asyncio
async def test():
    print('Training LSTM...')
    client = YFinanceClient()
    data = await client.get_historical_data('AAPL', period='1y')
    fe = FeatureEngineer()
    features = fe.create_features(data)
    X, y = fe.create_sequences(features, sequence_length=60)
    model = LSTMModel(input_shape=(60, X.shape[2]))
    history = model.train(X, y, epochs=5, batch_size=32, validation_split=0.2)
    print(f'✅ Accuracy: {history.history[\"accuracy\"][-1]:.4f}')
    await client.close()
asyncio.run(test())
"
```
- [ ] Training completes successfully
- [ ] Final accuracy > 0.55
- [ ] No out-of-memory errors

### Step 13: XGBoost Training (< 1 minute)
```bash
python -c "
from src.ml.models.xgboost_model import XGBoostModel
from src.ml.feature_engineering import FeatureEngineer
from src.data.yfinance_client import YFinanceClient
import asyncio
async def test():
    print('Training XGBoost...')
    client = YFinanceClient()
    data = await client.get_historical_data('AAPL', period='1y')
    fe = FeatureEngineer()
    features = fe.create_features(data)
    X, y = fe.prepare_supervised_data(features)
    model = XGBoostModel()
    metrics = model.train(X, y, validation_split=0.2)
    print(f'✅ Accuracy: {metrics[\"val_accuracy\"]:.4f}')
    await client.close()
asyncio.run(test())
"
```
- [ ] Training completes in < 5 seconds
- [ ] Accuracy > 0.55
- [ ] Predictions successful

### Step 14: Model Persistence
```bash
python -c "
from src.ml.models.xgboost_model import XGBoostModel
from src.ml.feature_engineering import FeatureEngineer
from src.data.yfinance_client import YFinanceClient
import asyncio, os
async def test():
    client = YFinanceClient()
    data = await client.get_historical_data('AAPL', period='6mo')
    fe = FeatureEngineer()
    features = fe.create_features(data)
    X, y = fe.prepare_supervised_data(features)
    model = XGBoostModel()
    model.train(X, y)
    model.save('models/test_model.pkl')
    loaded = XGBoostModel()
    loaded.load('models/test_model.pkl')
    print('✅ Model save/load working')
    os.remove('models/test_model.pkl')
    await client.close()
asyncio.run(test())
"
```
- [ ] Model saves successfully
- [ ] Model loads successfully
- [ ] Predictions match

**✅ Phase 3 Complete:** ML pipeline operational

---

## Phase 4: Code Quality (10-15 minutes)

### Step 15: Unit Tests
```bash
pytest tests/unit/ -v --cov=src --cov-report=term-missing
```
- [ ] All tests pass (81/81)
- [ ] Coverage ≥ 80%
- [ ] Completes in < 2 minutes

### Step 16: Integration Tests
```bash
pytest tests/integration/ -v -m "not slow"
```
- [ ] All tests pass (6/6)
- [ ] Completes in < 5 minutes

### Step 17: Code Linting
```bash
flake8 src/ --max-line-length=120 --max-complexity=10 --statistics
```
- [ ] Zero errors reported
- [ ] No complexity warnings

### Step 18: Security Scan
```bash
bandit -r src/ -ll
```
- [ ] Zero high/medium issues
- [ ] Low issues < 5

### Step 19: Secret Detection
```bash
detect-secrets scan --baseline .secrets.baseline
```
- [ ] No new secrets detected

### Step 20: Pre-commit Hooks
```bash
pre-commit run --all-files
```
- [ ] All hooks pass
- [ ] No files modified

**✅ Phase 4 Complete:** Code quality verified

---

## Phase 5: Integration (15-20 minutes)

### Step 21: Full ML Pipeline
```bash
python scripts/run_ml_pipeline.py --symbol AAPL --period 1y --epochs 10
```
- [ ] All 5 steps complete
- [ ] LSTM accuracy > 65%
- [ ] XGBoost accuracy > 60%
- [ ] Ensemble accuracy ≥ best model
- [ ] Models saved successfully

### Step 22: Database Performance
```bash
python -c "
import asyncio, time
from src.data.database import DatabaseManager
from src.data.yfinance_client import YFinanceClient
async def test():
    db = DatabaseManager()
    client = YFinanceClient()
    await db.connect()
    data = await client.get_historical_data('AAPL', period='1mo')
    start = time.time()
    await db.store_market_data('AAPL', data)
    elapsed = (time.time() - start) * 1000
    print(f'✅ Write: {elapsed:.2f}ms')
    await db.close()
    await client.close()
asyncio.run(test())
"
```
- [ ] Write performance < 100ms
- [ ] No database errors

### Step 23: Cache Performance
```bash
python -c "
import asyncio, time
from src.data.cache import CacheManager
async def test():
    cache = CacheManager()
    await cache.connect()
    start = time.time()
    for i in range(100):
        await cache.set(f'test_{i}', {'value': i}, ttl=60)
    elapsed = (time.time() - start) * 1000
    print(f'✅ Cache write: {elapsed/100:.2f}ms avg')
    for i in range(100):
        await cache.delete(f'test_{i}')
    await cache.close()
asyncio.run(test())
"
```
- [ ] Average write < 5ms
- [ ] No connection errors

### Step 24: End-to-End Signals
```bash
python -c "
import asyncio
from src.ml.ensemble import EnsembleModel
async def test():
    ensemble = EnsembleModel()
    try:
        await ensemble.load('models/ensemble_model')
        signal = await ensemble.predict('AAPL')
        print(f'✅ Signal: {signal[\"signal\"]} ({signal[\"confidence\"]:.2%})')
    except:
        print('⚠️ No pre-trained model - run Step 21 first')
asyncio.run(test())
"
```
- [ ] Signal generated successfully
- [ ] Confidence 50-100%
- [ ] Signal is BUY/SELL/HOLD

**✅ Phase 5 Complete:** Integration verified

---

## Final Validation Summary

### ✅ Validation Complete

**System Status:** ☐ PASS  ☐ FAIL

**Phases Completed:**
- [ ] Phase 1: System Setup (4/4 steps)
- [ ] Phase 2: Data Layer (6/6 steps)
- [ ] Phase 3: ML Models (4/4 steps)
- [ ] Phase 4: Code Quality (6/6 steps)
- [ ] Phase 5: Integration (4/4 steps)

**Total Steps:** 24/24 ☐

---

## Performance Summary

Record your results:

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Unit Tests | 81/81 pass | ______ | ☐ |
| Test Coverage | ≥80% | ______% | ☐ |
| LSTM Accuracy | >65% | ______% | ☐ |
| XGBoost Accuracy | >60% | ______% | ☐ |
| Ensemble Accuracy | ≥Best | ______% | ☐ |
| DB Write Time | <100ms | ______ms | ☐ |
| Cache Avg Write | <5ms | ______ms | ☐ |
| Security Issues | 0 high/med | ______ | ☐ |

---

## Issues Encountered

Record any issues or notes:

1. _______________________________________________
2. _______________________________________________
3. _______________________________________________

---

## Quick Troubleshooting

**Docker not starting:**
```bash
docker-compose down -v
docker-compose up -d
```

**Missing packages:**
```bash
pip install -r requirements.txt
```

**Database connection failed:**
```bash
docker-compose ps  # Check if postgres is Up
docker-compose logs postgres  # Check logs
```

**Tests failing:**
```bash
pytest tests/unit/test_file.py::test_name -vv  # Run specific test
```

**Low model accuracy:**
- Use more data: Change '1y' to '2y' or '5y'
- Increase epochs: Change 5 to 50 or 100
- Try different stocks: AAPL, MSFT, GOOGL

**Out of memory:**
- Reduce batch_size to 16
- Reduce sequence_length to 30
- Use shorter time period

---

## Next Steps After Validation

1. [ ] Review validation report
2. [ ] Address any failed items
3. [ ] Document any issues
4. [ ] Save this completed checklist
5. [ ] Proceed to Phase 3 (Backtesting) when ready
6. [ ] Start generating trading signals

---

## Validation Sign-off

**Validated By:** _______________________

**Date:** _______________________

**Time Taken:** _______ hours _______ minutes

**Overall Result:** ☐ PASS  ☐ FAIL

**Notes:**
_________________________________________________
_________________________________________________
_________________________________________________

---

**For detailed instructions, see:** VALIDATION_GUIDE.md

**Version:** 1.0 | **Last Updated:** October 31, 2025
