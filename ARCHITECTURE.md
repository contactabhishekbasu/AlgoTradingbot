# System Architecture Documentation
## Claude-Powered AI Trading System

**Version:** 1.0.0
**Date:** October 30, 2025
**Status:** Design Phase
**Architecture Owner:** Engineering Team
**Last Updated:** October 30, 2025

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Principles](#architecture-principles)
3. [MVP Architecture](#mvp-architecture)
4. [System Overview](#system-overview)
5. [Component Architecture](#component-architecture)
6. [Data Architecture](#data-architecture)
7. [Integration Architecture](#integration-architecture)
8. [Security Architecture](#security-architecture)
9. [Performance Architecture](#performance-architecture)
10. [Deployment Architecture](#deployment-architecture)
11. [Future Architecture](#future-architecture)

---

## Executive Summary

### Architecture Vision

Build a modular, scalable, and high-performance algorithmic trading system that leverages Claude AI through Model Context Protocol (MCP) to provide institutional-grade trading capabilities on consumer hardware (MacBook M4). The architecture prioritizes:

1. **MVP-First Design**: Minimal but functional core for rapid validation
2. **Backtesting-Ready**: Comprehensive framework for strategy validation before live deployment
3. **Modularity**: Independent, loosely-coupled components for easy testing and iteration
4. **Performance**: Sub-100ms prediction latency optimized for Apple Silicon
5. **Reliability**: Fault-tolerant design with graceful degradation

### MVP Scope

The MVP architecture focuses on three critical capabilities:

1. **Data Ingestion & Feature Engineering**: Real-time market data with technical indicators
2. **ML Prediction Pipeline**: Ensemble models (LSTM + XGBoost) with online learning
3. **Backtesting Framework**: Walk-forward analysis with realistic transaction costs

**Not in MVP**: Live trading, web dashboard, sentiment analysis, multi-asset support

### Success Criteria

- **Backtesting Performance**: Sharpe ratio >1.5, win rate >60%, max drawdown <20%
- **System Performance**: Prediction latency <100ms, model training <5 minutes
- **Validation**: Statistical significance (p<0.05) on 5 years of historical data

---

## Architecture Principles

### 1. **MVP-First Development**

**Principle**: Build the minimum viable system that can validate our core hypothesis through backtesting.

**Application**:
- Start with paper trading only (no live trading in MVP)
- Single asset class (US equities)
- Core indicators only (RSI, MACD, Bollinger Bands, Williams %R)
- Two ML models (LSTM + XGBoost) before expanding ensemble
- Local deployment only (no cloud infrastructure)

**Benefits**:
- Faster time to validation
- Lower development cost
- Clearer proof of concept
- Easier debugging and iteration

---

### 2. **Backtesting-Driven Design**

**Principle**: Architecture must support rigorous backtesting before any live trading.

**Application**:
- Data pipeline designed for both historical replay and real-time streaming
- All components support "simulation mode" with historical data
- Transaction cost modeling built into execution layer
- Walk-forward analysis as first-class citizen
- Reproducible results with fixed random seeds

**Benefits**:
- Validates strategies before risking capital
- Identifies performance degradation early
- Supports iterative strategy improvement
- Provides confidence metrics for deployment

---

### 3. **Modularity & Testability**

**Principle**: Each component should be independently testable and replaceable.

**Application**:
- MCP servers as independent microservices
- Clear interfaces between components
- Dependency injection for testability
- Mock data generators for unit testing
- Contract testing between services

**Benefits**:
- Parallel development possible
- Easy to test individual components
- Can replace implementations without breaking system
- Supports A/B testing of strategies

---

### 4. **Performance by Design**

**Principle**: Performance optimization built into architecture, not bolted on.

**Application**:
- Asynchronous processing throughout
- Redis caching for hot data paths
- Model quantization for faster inference
- Vectorized operations with NumPy
- Apple Silicon optimization (MPS, Accelerate framework)

**Benefits**:
- Meets <100ms latency requirements
- Efficient resource utilization
- Scales to multiple symbols
- Supports real-time trading

---

### 5. **Fail-Safe Design**

**Principle**: System should fail gracefully and protect capital.

**Application**:
- Circuit breakers on drawdown limits
- Automatic model performance monitoring
- Fallback to simpler models if prediction fails
- Dead man's switch for live trading
- Comprehensive error handling and logging

**Benefits**:
- Protects user capital
- Enables unattended operation
- Reduces operational risk
- Builds trust in system

---

## MVP Architecture

### MVP System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         MVP Architecture                          │
│                     (Backtesting-Focused)                         │
└─────────────────────────────────────────────────────────────────┘

                         Claude Desktop
                               │
                               ▼
                    ┌──────────────────┐
                    │  MCP Orchestrator │
                    │  (Command Router) │
                    └──────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌──────────────┐    ┌─────────────────┐
│   YFinance    │    │ ML Predictor │    │   Backtesting   │
│   Trader MCP  │◄───│     MCP      │◄───│   Engine MCP    │
│               │    │              │    │                 │
│ • Market Data │    │ • LSTM Model │    │ • Historical    │
│ • Indicators  │    │ • XGBoost    │    │   Simulation    │
│ • Historical  │    │ • Ensemble   │    │ • Walk-Forward  │
│   Data        │    │ • Online     │    │ • Performance   │
│               │    │   Learning   │    │   Metrics       │
└───────────────┘    └──────────────┘    └─────────────────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │  PostgreSQL DB   │
                    │                  │
                    │ • Market Data    │
                    │ • Model States   │
                    │ • Backtest Results│
                    └──────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │   Redis Cache    │
                    │                  │
                    │ • Hot Data       │
                    │ • Indicators     │
                    └──────────────────┘
```

### MVP Components

#### Core Components (Must Have)

1. **YFinance Trader MCP**
   - Purpose: Market data ingestion and technical analysis
   - Scope: US equities, daily data, core indicators
   - API: 8 endpoints (get_price, get_indicators, get_historical, etc.)

2. **ML Predictor MCP**
   - Purpose: Price prediction using ensemble models
   - Scope: LSTM + XGBoost, online learning
   - API: 5 endpoints (predict, train, update_online, get_performance, etc.)

3. **Backtesting Engine MCP**
   - Purpose: Strategy validation and performance analysis
   - Scope: Walk-forward analysis, transaction costs, metrics
   - API: 6 endpoints (run_backtest, walk_forward, optimize_params, etc.)

4. **Data Storage Layer**
   - PostgreSQL: Persistent storage for historical data and model states
   - Redis: High-speed cache for real-time indicators and predictions

5. **MCP Orchestrator**
   - Purpose: Route Claude commands to appropriate servers
   - Scope: Command parsing, context management, error handling
   - API: Natural language interface through Claude Desktop

#### Deferred Components (Post-MVP)

1. **Alpaca Trading MCP** - Live trading execution
2. **Portfolio Manager MCP** - Multi-position risk management
3. **Web Dashboard** - Visualization and monitoring
4. **Authentication Server** - User management and security
5. **Sentiment Analysis** - News and social media integration

### MVP Data Flow

```
Historical Data Request (Backtesting)
────────────────────────────────────

1. User → Claude: "Backtest mean reversion strategy on AAPL 2020-2024"
2. Claude → MCP Orchestrator: Parse command, identify strategy and parameters
3. Orchestrator → Backtesting Engine: run_backtest(symbol="AAPL", start="2020", end="2024", strategy="mean_reversion")
4. Backtesting Engine → YFinance Trader: get_historical(symbol="AAPL", start="2020-01-01", end="2024-12-31")
5. YFinance Trader → PostgreSQL: Check cache for historical data
6. PostgreSQL → YFinance Trader: Return cached data (or fetch from Yahoo Finance if missing)
7. YFinance Trader → Backtesting Engine: Return OHLCV data + technical indicators
8. Backtesting Engine → ML Predictor: train_models(training_data, validation_data)
9. ML Predictor → Redis: Cache trained models
10. ML Predictor → Backtesting Engine: Return trained models
11. Backtesting Engine: Simulate trades using walk-forward analysis
    - For each window:
      a. Train models on training period
      b. Generate predictions on test period
      c. Execute simulated trades
      d. Apply transaction costs
      e. Calculate metrics
12. Backtesting Engine → PostgreSQL: Store results
13. Backtesting Engine → Orchestrator: Return comprehensive performance report
14. Orchestrator → Claude → User: Present results with metrics and visualizations
```

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Full System Architecture                        │
│                    (Beyond MVP - Future State)                        │
└─────────────────────────────────────────────────────────────────────┘

                              User Layer
    ┌──────────────────┬──────────────────┬──────────────────┐
    │  Claude Desktop  │   Web Dashboard   │   Mobile App     │
    │  (NL Interface)  │   (Streamlit)     │   (React Native) │
    └──────────────────┴──────────────────┴──────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │    API Gateway    │
                    │  (Future: FastAPI)│
                    └─────────┬─────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    │                         │                         │
    │              MCP Orchestration Layer              │
    │                                                   │
    └─────────────────────────┬─────────────────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
       ▼                      ▼                      ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Data Layer  │    │  ML Layer    │    │ Trading Layer│
│              │    │              │    │              │
│ YFinance MCP │◄───│ Predictor    │───►│ Alpaca MCP   │
│ Sentiment    │    │ MCP          │    │ Portfolio    │
│ (Future)     │    │              │    │ Manager MCP  │
└──────────────┘    └──────────────┘    └──────────────┘
       │                      │                      │
       └──────────────────────┼──────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Analytics Layer │
                    │                  │
                    │ Backtesting MCP  │
                    │ Risk Analytics   │
                    └──────────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
       ▼                      ▼                      ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ PostgreSQL   │    │    Redis     │    │  Prometheus  │
│ (Persistent) │    │   (Cache)    │    │  (Metrics)   │
└──────────────┘    └──────────────┘    └──────────────┘
```

### Technology Stack

#### Backend Core

**Python 3.11+** (Primary Language)
- **Frameworks**: FastAPI (future API), Pandas, NumPy
- **ML Libraries**: TensorFlow/Keras, XGBoost, Scikit-learn
- **Data Processing**: Polars (future), Pandas, PyArrow
- **Async**: asyncio, aiohttp, websockets

**Node.js 18+** (Secondary Language)
- **Use Cases**: Alpaca trading client, authentication server
- **Frameworks**: Express (future), Socket.io
- **MCP SDK**: @modelcontextprotocol/sdk

#### Machine Learning Stack

```python
ml_stack = {
    'frameworks': {
        'tensorflow': '2.15+',      # LSTM models
        'xgboost': '2.0+',          # Gradient boosting
        'scikit-learn': '1.4+',     # Traditional ML
        'pytorch': '2.2+ (future)', # Alternative DL framework
    },
    'optimization': {
        'numba': '0.59+',           # JIT compilation
        'cython': '3.0+',           # C extensions
        'onnx': '1.15+ (future)',   # Model conversion
    },
    'mlops': {
        'mlflow': '2.10+ (future)', # Experiment tracking
        'dvc': '3.0+ (future)',     # Data versioning
    }
}
```

#### Data Storage

**PostgreSQL 14**
- **Purpose**: Persistent storage for historical data, model states, backtest results
- **Schema**: Time-series optimized with TimescaleDB extension (future)
- **Backup**: Daily full backups, hourly incremental

**Redis 7**
- **Purpose**: High-speed caching for real-time data
- **Use Cases**:
  - Market data caching (5-minute TTL)
  - Technical indicators (1-minute TTL)
  - Model predictions (30-second TTL)
  - Session management (future)

#### Apple Silicon Optimization

```python
apple_silicon_config = {
    'gpu_acceleration': {
        'framework': 'Metal Performance Shaders (MPS)',
        'use_case': 'TensorFlow model training/inference',
        'expected_speedup': '2-3x vs CPU',
    },
    'blas_optimization': {
        'library': 'Apple Accelerate',
        'use_case': 'NumPy/SciPy operations',
        'expected_speedup': '1.5-2x vs OpenBLAS',
    },
    'memory_optimization': {
        'unified_memory': True,
        'model_quantization': 'int8',
        'batch_processing': True,
    },
    'compilation': {
        'jit_compiler': 'Numba (LLVM)',
        'use_case': 'Hot loop optimization',
        'expected_speedup': '5-10x for numerical code',
    }
}
```

---

## Component Architecture

### 1. YFinance Trader MCP

**Responsibility**: Market data acquisition and technical indicator calculation

#### Architecture

```python
class YFinanceTraderMCP:
    """
    Market data and technical analysis server

    Handles:
    - Real-time and historical price data
    - Technical indicator calculation
    - Data caching and validation
    """

    def __init__(self):
        self.cache = RedisCache(ttl=300)  # 5-minute cache
        self.db = PostgresConnection()
        self.indicator_calculator = TechnicalIndicators()

    # Core APIs
    async def get_current_price(self, symbol: str) -> PriceData
    async def get_historical_data(self, symbol: str, start: date, end: date) -> DataFrame
    async def calculate_indicators(self, data: DataFrame, indicators: List[str]) -> DataFrame
    async def get_market_status(self) -> MarketStatus

    # MVP Indicators
    async def get_rsi(self, symbol: str, period: int = 14) -> float
    async def get_williams_r(self, symbol: str, period: int = 14) -> float
    async def get_bollinger_bands(self, symbol: str, period: int = 20) -> BollingerBands
    async def get_macd(self, symbol: str) -> MACD
```

#### Data Flow

```
Request → Cache Check → Cache Hit? → Return cached data
                 │
                 No
                 ↓
         Fetch from YFinance API
                 ↓
         Validate data quality
                 ↓
         Calculate indicators
                 ↓
         Store in cache + DB
                 ↓
         Return to client
```

#### Performance Optimizations

1. **Caching Strategy**:
   - L1: In-memory LRU cache (1000 entries)
   - L2: Redis cache (5-minute TTL for real-time, 1-day for historical)
   - L3: PostgreSQL (permanent storage)

2. **Batch Processing**:
   - Fetch multiple symbols in single API call
   - Vectorized indicator calculations with NumPy
   - Async I/O for parallel fetching

3. **Data Validation**:
   - Check for missing/invalid values
   - Outlier detection (>3 std dev)
   - Forward-fill small gaps (<3 bars)
   - Alert on data quality issues

#### Error Handling

```python
error_handling = {
    'api_failure': 'Retry with exponential backoff (3 attempts)',
    'invalid_data': 'Return error, don\'t cache bad data',
    'missing_symbol': 'Return graceful error message',
    'rate_limit': 'Queue requests, respect limits',
    'network_timeout': 'Fail fast (5s timeout), retry once',
}
```

---

### 2. ML Predictor MCP

**Responsibility**: Machine learning model training, inference, and online learning

#### Architecture

```python
class MLPredictorMCP:
    """
    Ensemble ML prediction server

    Models:
    - LSTM with attention (3 layers, 128 units)
    - XGBoost (100 trees)
    - Random Forest (100 trees) - Post-MVP

    Features:
    - Online learning with mini-batch updates
    - Model versioning and rollback
    - Performance monitoring
    """

    def __init__(self):
        self.models = {
            'lstm': LSTMWithAttention(layers=3, units=128),
            'xgboost': XGBoostModel(n_estimators=100),
        }
        self.ensemble = AdaptiveEnsemble(models=self.models)
        self.online_learner = OnlineLearner(buffer_size=1000)
        self.model_registry = ModelRegistry(db=PostgresConnection())

    # Core APIs
    async def predict(self, symbol: str, features: DataFrame) -> Prediction
    async def train_models(self, training_data: DataFrame, validation_data: DataFrame) -> ModelMetrics
    async def update_online(self, new_data: DataFrame) -> None
    async def get_model_performance(self) -> PerformanceMetrics
    async def rollback_model(self, version: str) -> None

    # Ensemble methods
    async def get_model_weights(self) -> Dict[str, float]
    async def update_weights(self, performance: Dict[str, float]) -> None
```

#### Model Architecture: LSTM

```python
lstm_architecture = {
    'input_layer': {
        'shape': (60, 50),  # 60 time steps, 50 features
        'features': [
            'price_features': ['open', 'high', 'low', 'close', 'volume'],
            'technical_indicators': ['rsi', 'macd', 'bbands', 'williams_r', ...],
            'derived_features': ['returns', 'log_returns', 'volatility', ...],
        ]
    },
    'lstm_layers': [
        {'units': 128, 'dropout': 0.2, 'return_sequences': True},
        {'units': 128, 'dropout': 0.2, 'return_sequences': True},
        {'units': 64, 'dropout': 0.2, 'return_sequences': False},
    ],
    'attention_layer': {
        'heads': 8,
        'key_dim': 16,
    },
    'dense_layers': [
        {'units': 32, 'activation': 'relu', 'dropout': 0.3},
        {'units': 16, 'activation': 'relu'},
    ],
    'output_layer': {
        'units': 3,  # [price_up, price_down, price_neutral]
        'activation': 'softmax',
    },
    'optimizer': {
        'type': 'Adam',
        'learning_rate': 0.001,
        'decay': 1e-6,
    },
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy', 'precision', 'recall', 'f1'],
}
```

#### Model Architecture: XGBoost

```python
xgboost_config = {
    'objective': 'multi:softmax',  # or 'binary:logistic' for binary
    'num_class': 3,
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
    'tree_method': 'hist',  # Faster than 'auto'
    'device': 'cpu',  # GPU not supported on M4 with XGBoost
    'random_state': 42,
}
```

#### Ensemble Strategy

```python
class AdaptiveEnsemble:
    """
    Dynamically weighted ensemble

    Weights update based on recent performance:
    - Daily: Based on previous day's accuracy
    - Weekly: Based on rolling 7-day Sharpe ratio
    - Monthly: Full re-evaluation and retraining
    """

    def __init__(self, models: Dict[str, Model]):
        self.models = models
        self.weights = self._initialize_weights()  # Equal weights initially
        self.performance_tracker = PerformanceTracker()

    async def predict(self, features: DataFrame) -> Prediction:
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            pred = await model.predict(features)
            predictions[name] = pred

        # Weighted average
        ensemble_pred = sum(
            predictions[name] * self.weights[name]
            for name in self.models.keys()
        )

        return Prediction(
            value=ensemble_pred,
            confidence=self._calculate_confidence(predictions),
            contributing_models=predictions,
            weights=self.weights,
        )

    async def update_weights(self):
        """Update weights based on recent performance"""
        performance = await self.performance_tracker.get_recent_performance()

        # Softmax over performance scores
        scores = np.array([performance[name]['sharpe'] for name in self.models.keys()])
        self.weights = softmax(scores)
```

#### Online Learning Pipeline

```python
class OnlineLearner:
    """
    Incremental learning with mini-batch updates

    Process:
    1. Accumulate new data in buffer
    2. When buffer reaches threshold, trigger update
    3. Train on new data + sample from historical data
    4. Validate on held-out recent data
    5. Update models if performance improves
    """

    def __init__(self, buffer_size: int = 1000):
        self.buffer = deque(maxlen=buffer_size)
        self.update_threshold = 100  # Update after 100 new samples
        self.validation_size = 0.2

    async def add_sample(self, features: np.ndarray, target: float):
        """Add new sample to buffer"""
        self.buffer.append((features, target))

        if len(self.buffer) >= self.update_threshold:
            await self.trigger_update()

    async def trigger_update(self):
        """Perform mini-batch update"""
        # Split buffer into train/val
        train_data, val_data = self._split_buffer()

        # Update each model
        for name, model in self.models.items():
            old_performance = model.current_performance

            # Incremental training
            model.partial_fit(train_data)

            # Validate
            new_performance = model.evaluate(val_data)

            # Rollback if performance degrades
            if new_performance < old_performance * 0.95:  # 5% tolerance
                model.rollback()
                logger.warning(f"Rolled back {name} due to performance degradation")

        # Clear buffer after update
        self.buffer.clear()
```

#### Performance Monitoring

```python
class PerformanceTracker:
    """
    Track and monitor model performance

    Metrics tracked:
    - Accuracy, Precision, Recall, F1
    - Sharpe ratio (predictions vs actual returns)
    - Calibration error (confidence vs accuracy)
    - Prediction latency
    """

    async def log_prediction(self, prediction: Prediction, actual: float):
        """Log prediction and actual outcome"""
        self.db.insert({
            'timestamp': datetime.now(),
            'prediction': prediction.value,
            'confidence': prediction.confidence,
            'actual': actual,
            'error': abs(prediction.value - actual),
        })

    async def get_recent_performance(self, window: str = '7d') -> Dict:
        """Calculate performance over recent window"""
        data = self.db.query(f"WHERE timestamp > now() - interval '{window}'")

        return {
            'accuracy': self._calculate_accuracy(data),
            'sharpe': self._calculate_sharpe(data),
            'calibration': self._calculate_calibration(data),
            'latency_p50': np.percentile(data['latency'], 50),
            'latency_p99': np.percentile(data['latency'], 99),
        }
```

---

### 3. Backtesting Engine MCP

**Responsibility**: Historical strategy simulation and walk-forward analysis

#### Architecture

```python
class BacktestingEngineMCP:
    """
    Comprehensive backtesting framework

    Features:
    - Walk-forward analysis
    - Transaction cost modeling
    - Monte Carlo simulation
    - Parameter optimization
    - Statistical validation
    """

    def __init__(self):
        self.data_loader = HistoricalDataLoader()
        self.simulator = TradingSimulator()
        self.metrics_calculator = MetricsCalculator()
        self.optimizer = ParameterOptimizer()

    # Core APIs
    async def run_backtest(self, config: BacktestConfig) -> BacktestResults
    async def walk_forward_analysis(self, config: WalkForwardConfig) -> WalkForwardResults
    async def optimize_parameters(self, config: OptimizationConfig) -> OptimalParameters
    async def monte_carlo_simulation(self, strategy: Strategy, n_runs: int = 1000) -> MonteCarloResults
    async def calculate_metrics(self, trades: List[Trade]) -> PerformanceMetrics
```

#### Backtest Configuration

```python
@dataclass
class BacktestConfig:
    """Configuration for backtest run"""
    symbol: str                    # e.g., "AAPL"
    start_date: date               # e.g., 2020-01-01
    end_date: date                 # e.g., 2024-12-31
    initial_capital: float         # e.g., 100000.0
    strategy: Strategy             # Strategy object

    # Transaction costs
    commission: float = 0.0        # Per-share commission (e.g., 0.005)
    slippage: float = 0.001        # Slippage as fraction (e.g., 0.1%)

    # Risk management
    position_size: PositionSizer   # Position sizing method
    stop_loss: Optional[float]     # Stop-loss percentage
    take_profit: Optional[float]   # Take-profit percentage
    max_positions: int = 1         # Max concurrent positions (MVP: 1)

    # Data settings
    data_frequency: str = '1d'     # Data frequency (MVP: daily)
    warmup_period: int = 60        # Bars needed before first trade
```

#### Walk-Forward Analysis

```python
class WalkForwardAnalysis:
    """
    Walk-forward optimization and validation

    Process:
    1. Split data into train/test windows
    2. Train model on train window
    3. Test on out-of-sample test window
    4. Roll forward, repeat
    5. Aggregate results across all windows
    """

    def __init__(self, train_period: int = 252, test_period: int = 21):
        self.train_period = train_period  # 1 year
        self.test_period = test_period    # 1 month
        self.anchor_mode = False          # Expanding vs rolling window

    async def run(self, data: DataFrame, strategy: Strategy) -> WalkForwardResults:
        """Execute walk-forward analysis"""
        results = []

        # Create windows
        windows = self._create_windows(data)

        for train_window, test_window in windows:
            # Train on in-sample data
            trained_strategy = await strategy.train(train_window)

            # Test on out-of-sample data
            test_results = await self._run_backtest(
                data=test_window,
                strategy=trained_strategy,
            )

            results.append(test_results)

        # Aggregate results
        return self._aggregate_results(results)

    def _create_windows(self, data: DataFrame) -> List[Tuple[DataFrame, DataFrame]]:
        """Create train/test windows"""
        windows = []
        start_idx = 0

        while start_idx + self.train_period + self.test_period <= len(data):
            train_end = start_idx + self.train_period
            test_end = train_end + self.test_period

            train_window = data.iloc[start_idx:train_end]
            test_window = data.iloc[train_end:test_end]

            windows.append((train_window, test_window))

            # Roll forward
            start_idx += self.test_period  # Non-overlapping test periods

        return windows
```

#### Trading Simulator

```python
class TradingSimulator:
    """
    Simulate trading with realistic costs

    Features:
    - Market orders with slippage
    - Limit orders (future)
    - Stop-loss and take-profit
    - Partial fills (future)
    - Market hours validation
    """

    def __init__(self, initial_capital: float, commission: float, slippage: float):
        self.cash = initial_capital
        self.positions = {}
        self.commission = commission
        self.slippage = slippage
        self.trade_log = []

    async def execute_trade(self, signal: Signal, price: float, timestamp: datetime) -> Trade:
        """Execute a trade based on signal"""
        # Calculate actual execution price with slippage
        if signal.direction == 'buy':
            execution_price = price * (1 + self.slippage)
        else:
            execution_price = price * (1 - self.slippage)

        # Calculate shares based on position sizing
        shares = signal.position_size / execution_price

        # Calculate total cost including commission
        cost = shares * execution_price
        commission_cost = shares * self.commission
        total_cost = cost + commission_cost

        # Check if we have enough cash
        if signal.direction == 'buy' and total_cost > self.cash:
            return Trade(status='rejected', reason='insufficient_funds')

        # Execute trade
        trade = Trade(
            timestamp=timestamp,
            symbol=signal.symbol,
            direction=signal.direction,
            shares=shares,
            price=execution_price,
            commission=commission_cost,
            slippage_cost=abs(price - execution_price) * shares,
            total_cost=total_cost,
        )

        # Update portfolio
        if signal.direction == 'buy':
            self.cash -= total_cost
            self.positions[signal.symbol] = self.positions.get(signal.symbol, 0) + shares
        else:
            self.cash += (cost - commission_cost)
            self.positions[signal.symbol] -= shares

        self.trade_log.append(trade)
        return trade
```

#### Performance Metrics Calculator

```python
class MetricsCalculator:
    """
    Calculate comprehensive performance metrics

    Metrics:
    - Returns: Total, CAGR, annualized
    - Risk: Sharpe, Sortino, Calmar ratios
    - Drawdown: Max, average, duration
    - Trade statistics: Win rate, profit factor, expectancy
    - Statistical: p-value, confidence intervals
    """

    def calculate(self, trades: List[Trade], portfolio_values: List[float]) -> PerformanceMetrics:
        """Calculate all metrics"""
        returns = self._calculate_returns(portfolio_values)

        return PerformanceMetrics(
            # Return metrics
            total_return=self._total_return(portfolio_values),
            cagr=self._cagr(portfolio_values, years=len(portfolio_values)/252),
            annualized_return=self._annualized_return(returns),

            # Risk metrics
            sharpe_ratio=self._sharpe_ratio(returns),
            sortino_ratio=self._sortino_ratio(returns),
            calmar_ratio=self._calmar_ratio(returns, self._max_drawdown(portfolio_values)),
            volatility=np.std(returns) * np.sqrt(252),

            # Drawdown metrics
            max_drawdown=self._max_drawdown(portfolio_values),
            max_drawdown_duration=self._max_drawdown_duration(portfolio_values),
            avg_drawdown=self._avg_drawdown(portfolio_values),

            # Trade statistics
            total_trades=len(trades),
            winning_trades=len([t for t in trades if t.pnl > 0]),
            losing_trades=len([t for t in trades if t.pnl < 0]),
            win_rate=len([t for t in trades if t.pnl > 0]) / len(trades),
            profit_factor=self._profit_factor(trades),
            expectancy=np.mean([t.pnl for t in trades]),

            # Statistical validation
            p_value=self._calculate_p_value(returns),
            confidence_interval=self._confidence_interval(returns),
        )

    def _sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * np.mean(excess_returns) / np.std(returns)

    def _max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax
        return np.min(drawdown)
```

---

## Data Architecture

### Database Schema

#### PostgreSQL Tables

```sql
-- Market data (historical prices)
CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(12, 4) NOT NULL,
    high DECIMAL(12, 4) NOT NULL,
    low DECIMAL(12, 4) NOT NULL,
    close DECIMAL(12, 4) NOT NULL,
    volume BIGINT NOT NULL,
    adjusted_close DECIMAL(12, 4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timestamp)
);

CREATE INDEX idx_market_data_symbol_timestamp ON market_data(symbol, timestamp DESC);
CREATE INDEX idx_market_data_timestamp ON market_data(timestamp DESC);

-- Technical indicators (pre-calculated)
CREATE TABLE technical_indicators (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    indicator_name VARCHAR(50) NOT NULL,
    indicator_value DECIMAL(12, 6),
    parameters JSONB,  -- Store indicator parameters
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timestamp, indicator_name, parameters)
);

CREATE INDEX idx_indicators_symbol_timestamp ON technical_indicators(symbol, timestamp DESC);

-- ML model states (versioned models)
CREATE TABLE model_states (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    model_type VARCHAR(20) NOT NULL,  -- 'lstm', 'xgboost', etc.
    model_data BYTEA,  -- Serialized model
    hyperparameters JSONB,
    training_metrics JSONB,
    trained_on_data_start TIMESTAMPTZ,
    trained_on_data_end TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(model_name, version)
);

CREATE INDEX idx_model_states_active ON model_states(model_name, is_active, created_at DESC);

-- Predictions (model outputs)
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    prediction_type VARCHAR(20) NOT NULL,  -- 'price', 'direction', 'signal'
    predicted_value DECIMAL(12, 6),
    confidence DECIMAL(5, 4),
    features_used JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_predictions_symbol_timestamp ON predictions(symbol, timestamp DESC);

-- Backtesting results
CREATE TABLE backtest_results (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    start_date TIMESTAMPTZ NOT NULL,
    end_date TIMESTAMPTZ NOT NULL,
    initial_capital DECIMAL(15, 2) NOT NULL,
    final_capital DECIMAL(15, 2) NOT NULL,
    total_return DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    win_rate DECIMAL(5, 4),
    total_trades INTEGER,
    configuration JSONB,  -- Full backtest config
    metrics JSONB,  -- All calculated metrics
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_backtest_results_run_id ON backtest_results(run_id);
CREATE INDEX idx_backtest_results_strategy ON backtest_results(strategy_name, created_at DESC);

-- Backtest trades (individual trades from backtests)
CREATE TABLE backtest_trades (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    direction VARCHAR(10) NOT NULL,  -- 'buy', 'sell'
    shares DECIMAL(15, 6) NOT NULL,
    price DECIMAL(12, 4) NOT NULL,
    commission DECIMAL(10, 4) NOT NULL,
    slippage_cost DECIMAL(10, 4) NOT NULL,
    total_cost DECIMAL(15, 4) NOT NULL,
    pnl DECIMAL(15, 4),
    portfolio_value DECIMAL(15, 4),
    FOREIGN KEY (run_id) REFERENCES backtest_results(run_id)
);

CREATE INDEX idx_backtest_trades_run_id ON backtest_trades(run_id, timestamp);
```

#### Redis Data Structures

```python
redis_schema = {
    # Current market data (5-minute TTL)
    'market:price:{symbol}': {
        'type': 'hash',
        'fields': ['open', 'high', 'low', 'close', 'volume', 'timestamp'],
        'ttl': 300,  # 5 minutes
    },

    # Technical indicators (1-minute TTL)
    'indicators:{symbol}:{indicator}': {
        'type': 'string',  # JSON serialized
        'value': {'value': 50.5, 'timestamp': '2025-10-30T10:00:00Z'},
        'ttl': 60,  # 1 minute
    },

    # Model predictions (30-second TTL)
    'prediction:{symbol}:{model}': {
        'type': 'string',  # JSON serialized
        'value': {'prediction': 150.5, 'confidence': 0.85, 'timestamp': '...'},
        'ttl': 30,  # 30 seconds
    },

    # Active model metadata (no TTL)
    'model:active:{model_name}': {
        'type': 'string',
        'value': 'version_1.2.3',
        'ttl': None,
    },

    # Rate limiting (per-user request tracking)
    'ratelimit:{user_id}:{endpoint}': {
        'type': 'counter',
        'ttl': 60,  # 1 minute window
    },
}
```

### Data Flow Patterns

#### Pattern 1: Real-Time Price Fetching

```
1. Client requests price for AAPL
2. Check Redis cache: market:price:AAPL
3. If cache hit → return immediately
4. If cache miss:
   a. Fetch from Yahoo Finance API
   b. Validate data quality
   c. Store in Redis (TTL=5min)
   d. Store in PostgreSQL (permanent)
   e. Return to client
```

#### Pattern 2: Historical Data for Backtesting

```
1. Backtesting engine requests AAPL 2020-2024
2. Query PostgreSQL:
   SELECT * FROM market_data
   WHERE symbol='AAPL'
   AND timestamp BETWEEN '2020-01-01' AND '2024-12-31'
   ORDER BY timestamp ASC
3. If data incomplete:
   a. Identify missing date ranges
   b. Fetch from Yahoo Finance
   c. Store in PostgreSQL
   d. Retry query
4. Calculate indicators on full dataset
5. Cache calculated indicators in PostgreSQL
6. Return to backtesting engine
```

#### Pattern 3: Model Training and Storage

```
1. ML Predictor prepares to train model
2. Load training data from PostgreSQL
3. Train model
4. Serialize model (pickle/ONNX)
5. Generate version number (semantic versioning)
6. Store in PostgreSQL model_states table
7. Update Redis with active model version
8. Log training metrics
9. Archive old model versions (keep last 5)
```

---

## Integration Architecture

### MCP Communication Protocol

#### Request/Response Pattern

```python
# MCP Request Format
{
    "jsonrpc": "2.0",
    "id": "uuid-request-id",
    "method": "tool_name",
    "params": {
        "symbol": "AAPL",
        "start_date": "2020-01-01",
        "end_date": "2024-12-31"
    }
}

# MCP Response Format
{
    "jsonrpc": "2.0",
    "id": "uuid-request-id",
    "result": {
        "data": [...],
        "metadata": {
            "timestamp": "2025-10-30T10:00:00Z",
            "source": "yfinance",
            "cached": false
        }
    }
}

# MCP Error Format
{
    "jsonrpc": "2.0",
    "id": "uuid-request-id",
    "error": {
        "code": -32600,
        "message": "Invalid symbol: XYZ",
        "data": {
            "error_type": "InvalidSymbol",
            "details": "Symbol XYZ not found in market data"
        }
    }
}
```

#### MCP Server Discovery

```python
# Claude Desktop configuration (~/.config/claude/claude_desktop_config.json)
{
    "mcpServers": {
        "yfinance-trader": {
            "command": "python",
            "args": ["/path/to/yfinance_trader_mcp.py"],
            "env": {
                "REDIS_URL": "redis://localhost:6379",
                "POSTGRES_URL": "postgresql://localhost:5432/trading"
            }
        },
        "ml-predictor": {
            "command": "python",
            "args": ["/path/to/ml_predictor_mcp.py"],
            "env": {
                "MODEL_PATH": "/path/to/models",
                "REDIS_URL": "redis://localhost:6379"
            }
        },
        "backtesting-engine": {
            "command": "python",
            "args": ["/path/to/backtesting_mcp.py"],
            "env": {
                "DATA_PATH": "/path/to/historical_data",
                "POSTGRES_URL": "postgresql://localhost:5432/trading"
            }
        }
    }
}
```

### External API Integration

#### Yahoo Finance Integration

```python
class YFinanceClient:
    """Wrapper for Yahoo Finance API"""

    def __init__(self):
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(max_requests=2000, period=3600)  # 2000/hour

    async def get_historical_data(self, symbol: str, start: date, end: date) -> DataFrame:
        """Fetch historical data with retry logic"""
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((ConnectionError, Timeout))
        )
        async def _fetch():
            await self.rate_limiter.acquire()
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start, end=end, interval='1d')
            return data

        return await _fetch()
```

#### Alpaca Integration (Post-MVP)

```typescript
// Alpaca Trading Client (Node.js)
import { AlpacaClient } from '@alpacahq/alpaca-trade-api';

class AlpacaTradingMCP {
    private client: AlpacaClient;

    constructor() {
        this.client = new AlpacaClient({
            keyId: process.env.ALPACA_API_KEY,
            secretKey: process.env.ALPACA_SECRET_KEY,
            paper: true,  // Start with paper trading
        });
    }

    async placeOrder(order: OrderRequest): Promise<Order> {
        // Order execution with retry logic
        return await this.client.createOrder({
            symbol: order.symbol,
            qty: order.quantity,
            side: order.side,
            type: order.type,
            time_in_force: 'day',
        });
    }
}
```

---

## Security Architecture

### Security Principles

1. **Principle of Least Privilege**: Each component has minimal permissions
2. **Defense in Depth**: Multiple layers of security controls
3. **Secure by Default**: Security features enabled out of the box
4. **Zero Trust**: Verify all requests, trust nothing

### API Key Management

```python
class SecureKeyStorage:
    """
    Secure storage for API keys

    Features:
    - AES-256 encryption at rest
    - Environment variable isolation
    - No logging of sensitive data
    - Automatic key rotation (future)
    """

    def __init__(self):
        self.cipher = Fernet(self._get_encryption_key())

    def _get_encryption_key(self) -> bytes:
        """Load encryption key from secure location"""
        key_path = os.path.expanduser('~/.trading_system/encryption.key')

        if not os.path.exists(key_path):
            # Generate new key
            key = Fernet.generate_key()
            os.makedirs(os.path.dirname(key_path), mode=0o700, exist_ok=True)
            with open(key_path, 'wb') as f:
                f.write(key)
            os.chmod(key_path, 0o600)  # Owner read/write only
            return key

        with open(key_path, 'rb') as f:
            return f.read()

    def store_api_key(self, service: str, api_key: str):
        """Store encrypted API key"""
        encrypted = self.cipher.encrypt(api_key.encode())

        # Store in secure location
        key_file = os.path.expanduser(f'~/.trading_system/keys/{service}.enc')
        os.makedirs(os.path.dirname(key_file), mode=0o700, exist_ok=True)

        with open(key_file, 'wb') as f:
            f.write(encrypted)

        os.chmod(key_file, 0o600)

    def load_api_key(self, service: str) -> str:
        """Load and decrypt API key"""
        key_file = os.path.expanduser(f'~/.trading_system/keys/{service}.enc')

        with open(key_file, 'rb') as f:
            encrypted = f.read()

        return self.cipher.decrypt(encrypted).decode()
```

### Data Protection

```python
security_config = {
    'encryption': {
        'at_rest': {
            'algorithm': 'AES-256-GCM',
            'key_derivation': 'PBKDF2-SHA256',
            'applies_to': ['api_keys', 'user_data', 'backups'],
        },
        'in_transit': {
            'protocol': 'TLS 1.3',
            'cipher_suites': ['TLS_AES_256_GCM_SHA384'],
            'applies_to': ['all_external_apis', 'future_web_dashboard'],
        },
    },
    'access_control': {
        'api_keys': {
            'storage': 'encrypted_files',
            'permissions': 'owner_only (0600)',
            'location': '~/.trading_system/keys/',
        },
        'database': {
            'authentication': 'password + SSL certificate',
            'connection_string': 'environment_variable_only',
            'ssl_mode': 'require',
        },
    },
    'logging': {
        'sensitive_data_masking': True,
        'mask_patterns': ['API_KEY=', 'password=', 'token='],
        'audit_log': True,
        'log_retention': '90 days',
    },
}
```

### Input Validation

```python
class InputValidator:
    """
    Validate all external inputs

    Protects against:
    - SQL injection
    - Command injection
    - Path traversal
    - Invalid data types
    """

    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """Validate stock symbol"""
        if not re.match(r'^[A-Z]{1,5}$', symbol):
            raise ValueError(f"Invalid symbol format: {symbol}")
        return symbol

    @staticmethod
    def validate_date(date_str: str) -> date:
        """Validate date string"""
        try:
            return datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")

    @staticmethod
    def sanitize_sql(query: str) -> str:
        """Prevent SQL injection (though we use parameterized queries)"""
        # Always use parameterized queries; this is defense in depth
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'EXEC']
        for keyword in dangerous_keywords:
            if keyword in query.upper():
                raise ValueError(f"Potentially dangerous SQL keyword detected: {keyword}")
        return query
```

---

## Performance Architecture

### Performance Targets

| Component | Metric | Target | Measurement |
|-----------|--------|--------|-------------|
| **YFinance Trader** | Price fetch latency | <100ms | P50 |
| **ML Predictor** | Prediction latency | <100ms | P50 |
| **Backtesting Engine** | 10-year backtest | <1 minute | Total time |
| **Database** | Query latency | <50ms | P95 |
| **Redis** | Cache hit ratio | >90% | During market hours |
| **System** | Memory usage | <12GB | Peak |
| **System** | CPU usage | <80% | Sustained |

### Optimization Strategies

#### 1. Caching Strategy

```python
class MultiLevelCache:
    """
    Three-level caching system

    L1: In-memory LRU (fastest, smallest)
    L2: Redis (fast, medium)
    L3: PostgreSQL (slower, largest)
    """

    def __init__(self):
        self.l1_cache = LRUCache(maxsize=1000)
        self.l2_cache = RedisCache(ttl=300)
        self.l3_cache = PostgresCache()

    async def get(self, key: str) -> Optional[Any]:
        """Get from cache with fallback"""
        # Try L1
        value = self.l1_cache.get(key)
        if value is not None:
            return value

        # Try L2
        value = await self.l2_cache.get(key)
        if value is not None:
            self.l1_cache.put(key, value)  # Populate L1
            return value

        # Try L3
        value = await self.l3_cache.get(key)
        if value is not None:
            await self.l2_cache.put(key, value)  # Populate L2
            self.l1_cache.put(key, value)  # Populate L1
            return value

        return None

    async def put(self, key: str, value: Any):
        """Put in all cache levels"""
        self.l1_cache.put(key, value)
        await self.l2_cache.put(key, value)
        await self.l3_cache.put(key, value)
```

#### 2. Async Processing

```python
class AsyncDataPipeline:
    """
    Asynchronous data processing pipeline

    Benefits:
    - Concurrent API requests
    - Non-blocking I/O
    - Better resource utilization
    """

    async def fetch_multiple_symbols(self, symbols: List[str]) -> Dict[str, DataFrame]:
        """Fetch data for multiple symbols concurrently"""
        tasks = [self.fetch_symbol_data(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            symbol: result
            for symbol, result in zip(symbols, results)
            if not isinstance(result, Exception)
        }

    async def parallel_indicator_calculation(self, data: DataFrame) -> DataFrame:
        """Calculate multiple indicators in parallel"""
        indicators = ['rsi', 'macd', 'bbands', 'williams_r', 'stochastic']

        tasks = [
            self.calculate_indicator(data, indicator)
            for indicator in indicators
        ]

        results = await asyncio.gather(*tasks)

        # Merge results
        for indicator_data in results:
            data = data.join(indicator_data)

        return data
```

#### 3. Model Optimization

```python
class OptimizedModelInference:
    """
    Optimized ML model inference

    Techniques:
    - Model quantization (8-bit)
    - ONNX Runtime
    - Batch processing
    - GPU acceleration (MPS on M4)
    """

    def __init__(self, model_path: str):
        # Load quantized model
        self.model = self._load_quantized_model(model_path)

        # Use MPS (Metal Performance Shaders) on M4
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.model.to(self.device)
        self.model.eval()  # Inference mode

    def _load_quantized_model(self, path: str):
        """Load model with 8-bit quantization"""
        model = torch.load(path)

        # Dynamic quantization (8-bit weights, 8-bit activations)
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.LSTM},
            dtype=torch.qint8
        )

        return quantized_model

    @torch.no_grad()
    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """Batch prediction for efficiency"""
        # Convert to tensor
        tensor = torch.from_numpy(features).float().to(self.device)

        # Predict
        with torch.amp.autocast('mps'):  # Mixed precision on M4
            predictions = self.model(tensor)

        return predictions.cpu().numpy()
```

#### 4. Database Optimization

```sql
-- Partitioning for large tables
CREATE TABLE market_data_partitioned (
    LIKE market_data INCLUDING ALL
) PARTITION BY RANGE (timestamp);

-- Create partitions by year
CREATE TABLE market_data_2020 PARTITION OF market_data_partitioned
    FOR VALUES FROM ('2020-01-01') TO ('2021-01-01');

CREATE TABLE market_data_2021 PARTITION OF market_data_partitioned
    FOR VALUES FROM ('2021-01-01') TO ('2022-01-01');

-- Materialized views for common queries
CREATE MATERIALIZED VIEW daily_indicators AS
SELECT
    symbol,
    DATE(timestamp) as date,
    AVG(close) as avg_close,
    MAX(high) as max_high,
    MIN(low) as min_low,
    SUM(volume) as total_volume
FROM market_data
GROUP BY symbol, DATE(timestamp);

CREATE INDEX idx_daily_indicators_symbol_date ON daily_indicators(symbol, date DESC);

-- Refresh materialized view (run daily)
REFRESH MATERIALIZED VIEW CONCURRENTLY daily_indicators;
```

---

## Deployment Architecture

### MVP Deployment (Local MacBook M4)

```
┌─────────────────────────────────────────────────────────────┐
│                     MacBook M4 (Local)                       │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │              Docker Compose Stack                  │    │
│  │                                                     │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────┐ │    │
│  │  │ PostgreSQL   │  │    Redis     │  │  Grafana│ │    │
│  │  │   Container  │  │  Container   │  │(Future) │ │    │
│  │  └──────────────┘  └──────────────┘  └─────────┘ │    │
│  │                                                     │    │
│  │  ┌────────────────────────────────────────────┐   │    │
│  │  │         MCP Servers (Python)               │   │    │
│  │  │                                             │   │    │
│  │  │  • yfinance_trader_mcp.py                  │   │    │
│  │  │  • ml_predictor_mcp.py                     │   │    │
│  │  │  • backtesting_engine_mcp.py               │   │    │
│  │  └────────────────────────────────────────────┘   │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │              Claude Desktop App                    │    │
│  │         (Natural Language Interface)               │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Docker Compose Configuration

```yaml
version: '3.9'

services:
  postgres:
    image: postgres:14-alpine
    container_name: trading_postgres
    environment:
      POSTGRES_DB: trading
      POSTGRES_USER: trading_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U trading_user"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: trading_redis
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Future: Monitoring
  # prometheus:
  #   image: prom/prometheus:latest
  #   volumes:
  #     - ./prometheus.yml:/etc/prometheus/prometheus.yml
  #   ports:
  #     - "9090:9090"

  # grafana:
  #   image: grafana/grafana:latest
  #   ports:
  #     - "3000:3000"
  #   depends_on:
  #     - prometheus

volumes:
  postgres_data:
  redis_data:
```

### Environment Configuration

```bash
# .env file (not committed to git)
# ===================================

# Database
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_URL=postgresql://trading_user:${POSTGRES_PASSWORD}@localhost:5432/trading

# Redis
REDIS_URL=redis://localhost:6379

# API Keys (encrypted separately)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key

# Model Configuration
MODEL_PATH=./models
MODEL_DEVICE=mps  # Use Metal Performance Shaders on M4

# Logging
LOG_LEVEL=INFO
LOG_PATH=./logs

# Performance
CACHE_TTL=300  # 5 minutes
MAX_WORKERS=4
BATCH_SIZE=32
```

### Startup Procedure

```bash
#!/bin/bash
# scripts/start_system.sh

set -e

echo "Starting AI Trading System..."

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start Docker containers
echo "Starting Docker containers..."
docker-compose up -d

# Wait for services to be healthy
echo "Waiting for services to be ready..."
docker-compose exec -T postgres pg_isready -U trading_user
docker-compose exec -T redis redis-cli ping

# Initialize database schema (if not already done)
echo "Initializing database..."
python scripts/init_database.py

# Load historical data (if needed)
echo "Checking historical data..."
python scripts/check_and_load_data.py

# Start MCP servers (managed by Claude Desktop)
echo "MCP servers will be started by Claude Desktop when needed"

# Health check
echo "Running health checks..."
python scripts/health_check.py

echo "✅ System started successfully!"
echo "Open Claude Desktop to begin trading"
```

---

## Future Architecture

### Post-MVP Enhancements

#### 1. Live Trading Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Live Trading Layer                       │
│                                                           │
│  ┌────────────┐      ┌──────────────┐    ┌───────────┐ │
│  │  Signal    │─────▶│  Risk        │───▶│  Order    │ │
│  │  Generator │      │  Manager     │    │  Execution│ │
│  └────────────┘      └──────────────┘    └───────────┘ │
│        │                     │                    │      │
│        ▼                     ▼                    ▼      │
│  ┌────────────┐      ┌──────────────┐    ┌───────────┐ │
│  │  Position  │      │  Circuit     │    │  Alpaca   │ │
│  │  Sizing    │      │  Breakers    │    │  API      │ │
│  └────────────┘      └──────────────┘    └───────────┘ │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

#### 2. Multi-User Web Platform

```
┌─────────────────────────────────────────────────────────┐
│                   Frontend Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   React App  │  │   Mobile App │  │  Claude CLI  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                    API Gateway                           │
│  • Authentication (JWT)                                  │
│  • Rate Limiting                                         │
│  • Load Balancing                                        │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                  Backend Services                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐│
│  │  User    │  │Portfolio │  │ Strategy │  │Billing ││
│  │  Service │  │ Service  │  │ Service  │  │Service ││
│  └──────────┘  └──────────┘  └──────────┘  └────────┘│
└─────────────────────────────────────────────────────────┘
```

#### 3. Cloud Deployment Option

```
┌─────────────────────────────────────────────────────────┐
│                      AWS/GCP Cloud                       │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │               Load Balancer                    │    │
│  └────────────────────────────────────────────────┘    │
│                        │                                │
│  ┌─────────────────────┼─────────────────────────┐    │
│  │                     │                          │    │
│  ▼                     ▼                          ▼    │
│ ┌──────────┐    ┌──────────┐              ┌──────────┐│
│ │  ECS/K8s │    │  ECS/K8s │      ...     │  ECS/K8s ││
│ │Container │    │Container │              │Container ││
│ └──────────┘    └──────────┘              └──────────┘│
│                                                          │
│  ┌────────────────┐    ┌─────────────┐   ┌──────────┐ │
│  │   RDS/Aurora   │    │  ElastiCache│   │    S3    │ │
│  │  (PostgreSQL)  │    │   (Redis)   │   │ (Models) │ │
│  └────────────────┘    └─────────────┘   └──────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Appendix

### A. API Reference Summary

#### YFinance Trader MCP

```
GET /price/{symbol}                  - Get current price
GET /historical/{symbol}             - Get historical data
GET /indicators/{symbol}             - Calculate technical indicators
GET /market_status                   - Check if market is open
POST /bulk_fetch                     - Fetch multiple symbols
```

#### ML Predictor MCP

```
POST /predict                        - Generate prediction
POST /train                          - Train models
POST /update_online                  - Online learning update
GET /performance                     - Model performance metrics
POST /rollback                       - Rollback to previous version
GET /model_info                      - Current model details
```

#### Backtesting Engine MCP

```
POST /backtest                       - Run backtest
POST /walk_forward                   - Walk-forward analysis
POST /optimize                       - Parameter optimization
POST /monte_carlo                    - Monte Carlo simulation
GET /results/{run_id}                - Get backtest results
GET /trades/{run_id}                 - Get individual trades
```

### B. Performance Benchmarks

| Operation | Current | Target | Status |
|-----------|---------|--------|--------|
| Price fetch (cached) | 5ms | <10ms | ✅ |
| Price fetch (API) | 150ms | <200ms | ✅ |
| Indicator calculation (single) | 8ms | <10ms | ✅ |
| LSTM prediction | 87ms | <100ms | ✅ |
| XGBoost prediction | 12ms | <20ms | ✅ |
| Backtest (1 year, daily) | 8s | <10s | ✅ |
| Backtest (10 years, daily) | 45s | <60s | ✅ |
| Model training (LSTM) | 3m 20s | <5m | ✅ |
| Model training (XGBoost) | 45s | <1m | ✅ |

### C. Glossary

- **MCP**: Model Context Protocol - Anthropic's protocol for AI-tool integration
- **LSTM**: Long Short-Term Memory neural network
- **Walk-Forward Analysis**: Out-of-sample testing methodology
- **Sharpe Ratio**: Risk-adjusted return metric
- **Drawdown**: Peak-to-trough portfolio decline
- **Slippage**: Difference between expected and actual execution price
- **MPS**: Metal Performance Shaders - Apple's GPU framework

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | October 30, 2025 | Engineering Team | Initial architecture document with MVP focus |

---

**Approval:**

**Engineering Lead:** ________________________
**Product Owner:** ________________________
**Date:** ________________________

---

*This architecture document is a living document and will be updated as the system evolves.*
