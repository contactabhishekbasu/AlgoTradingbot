# System Architecture Documentation
## Claude-Powered AI Trading System

**Version:** 1.0.0
**Last Updated:** October 30, 2025
**Status:** Design Phase

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Principles](#architecture-principles)
3. [Component Architecture](#component-architecture)
4. [Data Architecture](#data-architecture)
5. [Machine Learning Architecture](#machine-learning-architecture)
6. [API Design](#api-design)
7. [Security Architecture](#security-architecture)
8. [Deployment Architecture](#deployment-architecture)
9. [Performance Optimization](#performance-optimization)
10. [Monitoring & Observability](#monitoring--observability)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interaction Layer                    │
│  ┌──────────────────┐         ┌─────────────────────────────┐  │
│  │  Claude Desktop  │◄────────┤   Streamlit Web Dashboard   │  │
│  │  (Natural Lang)  │         │   (Real-time Monitoring)    │  │
│  └────────┬─────────┘         └──────────────┬──────────────┘  │
└───────────┼────────────────────────────────────┼─────────────────┘
            │                                    │
            ▼                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MCP Orchestration Layer                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │               MCP Orchestrator (Python)                    │  │
│  │  - Request routing        - Load balancing                 │  │
│  │  - Context management     - Error handling                 │  │
│  └───────────┬───────────────────────────────────────────────┘  │
└──────────────┼──────────────────────────────────────────────────┘
               │
    ┌──────────┴─────────┬──────────┬──────────┬─────────────┐
    ▼                    ▼          ▼          ▼             ▼
┌─────────────┐  ┌──────────────┐ ┌────────┐ ┌──────────┐ ┌─────────┐
│  YFinance   │  │ ML Predictor │ │ Alpaca │ │Portfolio │ │  Auth   │
│   Trader    │  │    Server    │ │Trading │ │ Manager  │ │ Server  │
│  (Python)   │  │   (Python)   │ │(Node.js)│ │(Python) │ │(Node.js)│
└──────┬──────┘  └──────┬───────┘ └───┬────┘ └────┬─────┘ └────┬────┘
       │                │              │           │            │
       └────────────────┴──────────────┴───────────┴────────────┘
                                 │
                    ┌────────────┴─────────────┐
                    ▼                          ▼
          ┌──────────────────┐      ┌──────────────────┐
          │   PostgreSQL     │      │      Redis       │
          │  (Persistence)   │      │    (Caching)     │
          └──────────────────┘      └──────────────────┘
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
    ┌──────────┐        ┌──────────┐
    │  Models  │        │   Data   │
    │ Storage  │        │  Cache   │
    └──────────┘        └──────────┘
```

### Core Design Philosophy

**1. Microservices via MCP**: Each functional domain is an independent MCP server
**2. Real-time First**: Stream processing with event-driven architecture
**3. ML-Centric**: Machine learning as a first-class citizen, not an afterthought
**4. Local-First**: Optimized for MacBook M4, cloud-optional
**5. Fail-Safe**: Multiple layers of risk management and circuit breakers

---

## Architecture Principles

### 1. Separation of Concerns

Each MCP server has a single, well-defined responsibility:

| Server | Responsibility | Why Separate |
|--------|---------------|--------------|
| **yfinance_trader** | Market data acquisition | Isolate external API dependencies |
| **ml_predictor** | Price prediction | Heavy computation, independent scaling |
| **alpaca_trading** | Order execution | Critical path, needs isolation |
| **portfolio_manager** | Risk & position management | Complex business logic |
| **auth_server** | Authentication | Security boundary |

### 2. Event-Driven Architecture

```python
# Event Flow Example
Market Data Event → Feature Engineering → Model Prediction → Signal Generation
                                                              ↓
                                                        Risk Check
                                                              ↓
                                                    Order Placement ← User Approval
                                                              ↓
                                                      Position Update
                                                              ↓
                                                    Portfolio Rebalance
```

### 3. Idempotency

All operations are idempotent to handle retries:
- Order placement uses unique `client_order_id`
- Predictions cached with TTL
- Database operations use `INSERT ... ON CONFLICT`

### 4. Circuit Breakers

Fail fast to protect capital:
```python
circuit_breakers = {
    'daily_loss': -5%,      # Stop all trading
    'drawdown': -15%,       # Reduce position sizes
    'api_errors': 5,        # Switch to backup data source
    'model_confidence': 0.6 # Require higher confidence
}
```

### 5. Data Immutability

- Historical data is immutable (append-only)
- Trades cannot be deleted (only marked canceled)
- Model predictions stored with version and timestamp
- Audit trail for all actions

---

## Component Architecture

### MCP Orchestrator

**Technology:** Python 3.11+
**Role:** Central coordinator for all MCP servers
**File:** `mcp_orchestrator/server.py`

```python
class MCPOrchestrator:
    """
    Routes requests from Claude to appropriate MCP servers.
    Manages context, handles errors, implements retry logic.
    """

    def __init__(self):
        self.servers = {
            'yfinance': YFinanceConnection(),
            'ml_predictor': MLPredictorConnection(),
            'alpaca': AlpacaConnection(),
            'portfolio': PortfolioConnection(),
            'auth': AuthConnection()
        }
        self.context = ConversationContext()
        self.cache = RedisCache()

    async def route_request(self, request: Request) -> Response:
        """Route request to appropriate server with retry logic."""
        intent = self.classify_intent(request)
        target_server = self.intent_to_server[intent]

        # Circuit breaker check
        if not self.circuit_breakers[target_server].is_closed():
            return fallback_response(request)

        # Execute with retry
        response = await self.execute_with_retry(
            target_server,
            request,
            max_retries=3
        )

        # Update context
        self.context.add_interaction(request, response)

        return response
```

**Key Responsibilities:**
1. **Intent Classification**: Parse natural language to determine action
2. **Request Routing**: Send to correct MCP server
3. **Context Management**: Maintain conversation state
4. **Error Handling**: Retry, fallback, graceful degradation
5. **Response Aggregation**: Combine multi-server responses

---

### YFinance Trader MCP Server

**Technology:** Python 3.11+ with yfinance, pandas, ta-lib
**Port:** 8001
**File:** `mcp_servers/yfinance_trader/server.py`

```python
class YFinanceTraderServer:
    """
    Provides market data and technical analysis.
    """

    async def get_historical_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data with caching."""
        cache_key = f"hist:{symbol}:{period}:{interval}"

        # Check cache first
        if cached := self.cache.get(cache_key):
            return cached

        # Fetch from yfinance
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        # Cache for 1 hour (longer for daily data)
        ttl = 3600 if interval == "1d" else 300
        self.cache.set(cache_key, df, ttl=ttl)

        return df

    async def calculate_indicators(
        self,
        symbol: str,
        indicators: List[str]
    ) -> Dict[str, pd.Series]:
        """Calculate technical indicators."""
        df = await self.get_historical_data(symbol)

        results = {}
        for indicator in indicators:
            if indicator == "RSI":
                results["RSI"] = ta.momentum.RSIIndicator(
                    df['Close'], window=14
                ).rsi()
            elif indicator == "MACD":
                macd = ta.trend.MACD(df['Close'])
                results["MACD"] = macd.macd()
                results["MACD_signal"] = macd.macd_signal()
            # ... more indicators

        return results

    async def get_realtime_quote(self, symbol: str) -> Dict:
        """Get real-time quote data."""
        ticker = yf.Ticker(symbol)
        info = ticker.info

        return {
            'symbol': symbol,
            'price': info.get('currentPrice'),
            'bid': info.get('bid'),
            'ask': info.get('ask'),
            'volume': info.get('volume'),
            'timestamp': datetime.now()
        }
```

**Endpoints:**
- `get_historical_data(symbol, period, interval)` - OHLCV data
- `calculate_indicators(symbol, indicators)` - Technical indicators
- `get_realtime_quote(symbol)` - Current price
- `screen_stocks(criteria)` - Stock screening
- `get_fundamentals(symbol)` - Company fundamentals

**Caching Strategy:**
- Historical daily data: 1 hour TTL
- Intraday data: 5 minute TTL
- Real-time quotes: 10 second TTL
- Fundamentals: 24 hour TTL

---

### ML Predictor MCP Server

**Technology:** Python 3.11+ with TensorFlow, XGBoost, scikit-learn
**Port:** 8002
**File:** `mcp_servers/ml_predictor/server.py`

```python
class MLPredictorServer:
    """
    Provides machine learning predictions for price movements.
    """

    def __init__(self):
        self.models = {
            'lstm': self.load_model('lstm_model.h5'),
            'xgboost': self.load_model('xgboost_model.pkl'),
            'random_forest': self.load_model('rf_model.pkl')
        }
        self.ensemble_weights = self.load_weights()
        self.feature_engineer = FeatureEngineer()

    async def predict(
        self,
        symbol: str,
        horizon: str = "1d",
        models: List[str] = None
    ) -> Prediction:
        """Generate prediction using ensemble of models."""

        # Get features
        features = await self.feature_engineer.get_features(symbol)

        # Get predictions from each model
        predictions = {}
        confidences = {}

        for model_name in (models or self.models.keys()):
            model = self.models[model_name]
            pred = model.predict(features)
            conf = self.calculate_confidence(model, features)

            predictions[model_name] = pred
            confidences[model_name] = conf

        # Ensemble with weighted average
        final_prediction = self.ensemble_predict(
            predictions,
            confidences,
            self.ensemble_weights
        )

        return Prediction(
            symbol=symbol,
            direction=final_prediction['direction'],
            price_target=final_prediction['price'],
            confidence=final_prediction['confidence'],
            horizon=horizon,
            timestamp=datetime.now(),
            models_used=list(predictions.keys())
        )

    async def online_update(
        self,
        symbol: str,
        actual_price: float,
        prediction: Prediction
    ):
        """Update models with new data (online learning)."""

        # Calculate prediction error
        error = actual_price - prediction.price_target

        # Update ensemble weights
        self.update_ensemble_weights(prediction, error)

        # Partial fit for adaptive models
        for model_name in ['xgboost', 'random_forest']:
            model = self.models[model_name]
            if hasattr(model, 'partial_fit'):
                features = await self.feature_engineer.get_features(symbol)
                model.partial_fit(features, [actual_price])

        # Store for batch retraining
        self.store_training_example(symbol, prediction, actual_price)
```

**Model Architecture Details:**

#### LSTM Model
```python
# Architecture: Sequential LSTM with Attention
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(60, n_features)),
    Dropout(0.2),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Price prediction
])

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae', 'mape']
)
```

#### XGBoost Model
```python
model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    tree_method='hist',  # Faster for large datasets
    device='cpu'  # Apple Silicon optimization pending
)
```

**Feature Engineering Pipeline:**

```python
class FeatureEngineer:
    """Generate features for ML models."""

    async def get_features(self, symbol: str) -> np.ndarray:
        """
        Generate 100+ features from raw market data.
        """
        # Get raw data
        df = await self.yfinance_client.get_historical_data(
            symbol, period="6mo", interval="1d"
        )

        features = {}

        # Price features (10)
        features['returns_1d'] = df['Close'].pct_change(1)
        features['returns_5d'] = df['Close'].pct_change(5)
        features['returns_20d'] = df['Close'].pct_change(20)
        features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Momentum indicators (15)
        features['rsi_14'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
        features['rsi_21'] = ta.momentum.RSIIndicator(df['Close'], 21).rsi()
        features['stoch_k'] = ta.momentum.StochasticOscillator(
            df['High'], df['Low'], df['Close']
        ).stoch()
        features['williams_r'] = ta.momentum.WilliamsRIndicator(
            df['High'], df['Low'], df['Close']
        ).williams_r()

        # Trend indicators (20)
        features['sma_20'] = ta.trend.SMAIndicator(df['Close'], 20).sma_indicator()
        features['ema_12'] = ta.trend.EMAIndicator(df['Close'], 12).ema_indicator()
        features['macd'] = ta.trend.MACD(df['Close']).macd()
        features['adx'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()

        # Volatility indicators (15)
        features['bb_high'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
        features['bb_low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
        features['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()

        # Volume indicators (10)
        features['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        features['vwap'] = (df['Volume'] * df['Close']).cumsum() / df['Volume'].cumsum()

        # Pattern features (30+)
        features.update(self.detect_candlestick_patterns(df))

        # Combine into array
        return self.normalize_features(pd.DataFrame(features))
```

---

### Alpaca Trading MCP Server

**Technology:** Node.js 18+ with @alpacahq/alpaca-trade-api
**Port:** 8003
**File:** `mcp_servers/alpaca_trading/index.js`

```javascript
class AlpacaTradingServer {
    constructor() {
        this.alpaca = new Alpaca({
            keyId: process.env.ALPACA_API_KEY,
            secretKey: process.env.ALPACA_SECRET,
            paper: process.env.ALPACA_PAPER === 'true'
        });
    }

    async placeOrder(params) {
        /**
         * Place order with comprehensive validation and error handling.
         *
         * @param {Object} params
         * @param {string} params.symbol - Stock symbol
         * @param {number} params.qty - Quantity
         * @param {string} params.side - 'buy' or 'sell'
         * @param {string} params.type - 'market', 'limit', 'stop', 'stop_limit'
         * @param {string} params.time_in_force - 'day', 'gtc', 'ioc', 'fok'
         * @param {number} params.limit_price - For limit orders
         * @param {number} params.stop_price - For stop orders
         */

        // Pre-flight checks
        await this.validateOrder(params);
        await this.checkBuyingPower(params);
        await this.checkRiskLimits(params);

        // Generate unique client order ID
        const clientOrderId = `claude_${Date.now()}_${uuidv4()}`;

        try {
            const order = await this.alpaca.createOrder({
                symbol: params.symbol,
                qty: params.qty,
                side: params.side,
                type: params.type,
                time_in_force: params.time_in_force || 'day',
                limit_price: params.limit_price,
                stop_price: params.stop_price,
                client_order_id: clientOrderId
            });

            // Log to database
            await this.logOrder(order);

            // Start monitoring order status
            this.monitorOrder(order.id);

            return {
                success: true,
                order_id: order.id,
                client_order_id: clientOrderId,
                status: order.status,
                filled_qty: order.filled_qty,
                filled_avg_price: order.filled_avg_price
            };

        } catch (error) {
            await this.handleOrderError(error, params, clientOrderId);
            throw error;
        }
    }

    async getPositions() {
        /**
         * Get all open positions.
         */
        const positions = await this.alpaca.getPositions();

        return positions.map(pos => ({
            symbol: pos.symbol,
            qty: parseFloat(pos.qty),
            side: pos.side,
            market_value: parseFloat(pos.market_value),
            cost_basis: parseFloat(pos.cost_basis),
            unrealized_pl: parseFloat(pos.unrealized_pl),
            unrealized_plpc: parseFloat(pos.unrealized_plpc),
            current_price: parseFloat(pos.current_price),
            lastday_price: parseFloat(pos.lastday_price),
            change_today: parseFloat(pos.change_today)
        }));
    }

    async getAccount() {
        /**
         * Get account information.
         */
        const account = await this.alpaca.getAccount();

        return {
            buying_power: parseFloat(account.buying_power),
            cash: parseFloat(account.cash),
            portfolio_value: parseFloat(account.portfolio_value),
            equity: parseFloat(account.equity),
            last_equity: parseFloat(account.last_equity),
            long_market_value: parseFloat(account.long_market_value),
            short_market_value: parseFloat(account.short_market_value),
            initial_margin: parseFloat(account.initial_margin),
            maintenance_margin: parseFloat(account.maintenance_margin),
            daytrade_count: parseInt(account.daytrade_count),
            pattern_day_trader: account.pattern_day_trader
        };
    }

    async monitorOrder(orderId) {
        /**
         * Monitor order status and send updates.
         */
        const order = await this.alpaca.getOrder(orderId);

        // Check if order is complete
        if (['filled', 'canceled', 'rejected'].includes(order.status)) {
            await this.handleOrderComplete(order);
            return;
        }

        // Check again in 1 second
        setTimeout(() => this.monitorOrder(orderId), 1000);
    }
}
```

**Order Flow:**
```
User Command → MCP Orchestrator → Risk Check → Alpaca API → Order Placed
                                      ↓                          ↓
                                   Rejected                  Monitoring
                                      ↓                          ↓
                                 User Notified            Status Updates
                                                                 ↓
                                                            Filled/Canceled
                                                                 ↓
                                                         Portfolio Updated
```

---

### Portfolio Manager MCP Server

**Technology:** Python 3.11+ with numpy, scipy
**Port:** 8004
**File:** `mcp_servers/portfolio_manager/server.py`

```python
class PortfolioManagerServer:
    """
    Manages portfolio, calculates risk metrics, determines position sizes.
    """

    async def calculate_position_size(
        self,
        symbol: str,
        signal_confidence: float,
        strategy: str = "kelly"
    ) -> int:
        """
        Calculate optimal position size based on risk parameters.
        """
        # Get account info
        account = await self.alpaca_client.get_account()
        portfolio_value = account['portfolio_value']

        # Get current positions
        positions = await self.get_positions()
        current_exposure = sum(p['market_value'] for p in positions)

        # Get volatility
        volatility = await self.calculate_volatility(symbol)

        if strategy == "kelly":
            # Kelly Criterion: f = (bp - q) / b
            # f = fraction to bet
            # b = odds (payoff ratio)
            # p = probability of win
            # q = probability of loss (1-p)

            win_rate = signal_confidence
            avg_win = 0.05  # 5% average win
            avg_loss = 0.03  # 3% average loss

            kelly_fraction = (
                (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            )

            # Use half-Kelly for safety
            kelly_fraction *= 0.5

            # Apply Kelly to available capital
            position_value = portfolio_value * kelly_fraction

        elif strategy == "risk_parity":
            # Equal risk contribution
            target_risk = 0.10  # 10% portfolio volatility
            position_value = (
                target_risk * portfolio_value / volatility
            )

        elif strategy == "equal_weight":
            # Equal weight across N positions
            max_positions = 10
            position_value = portfolio_value / max_positions

        # Apply constraints
        position_value = self.apply_constraints(
            position_value,
            portfolio_value,
            current_exposure
        )

        # Convert to shares
        current_price = await self.get_current_price(symbol)
        shares = int(position_value / current_price)

        return shares

    async def check_risk_limits(
        self,
        symbol: str,
        qty: int,
        side: str
    ) -> Tuple[bool, str]:
        """
        Verify order doesn't violate risk limits.
        """
        account = await self.alpaca_client.get_account()
        positions = await self.get_positions()

        # Calculate new exposure
        price = await self.get_current_price(symbol)
        order_value = qty * price

        # Check 1: Single position limit (max 20% of portfolio)
        if order_value > account['portfolio_value'] * 0.20:
            return False, "Exceeds single position limit (20%)"

        # Check 2: Total exposure limit (max 90% of portfolio)
        current_exposure = sum(p['market_value'] for p in positions)
        if side == 'buy':
            new_exposure = current_exposure + order_value
            if new_exposure > account['portfolio_value'] * 0.90:
                return False, "Exceeds total exposure limit (90%)"

        # Check 3: Sector concentration (max 40% per sector)
        sector = await self.get_sector(symbol)
        sector_exposure = sum(
            p['market_value'] for p in positions
            if await self.get_sector(p['symbol']) == sector
        )
        if side == 'buy':
            new_sector_exposure = sector_exposure + order_value
            if new_sector_exposure > account['portfolio_value'] * 0.40:
                return False, f"Exceeds sector limit for {sector} (40%)"

        # Check 4: Maximum drawdown (circuit breaker)
        drawdown = await self.calculate_current_drawdown()
        if drawdown > 0.15:  # 15% max drawdown
            return False, f"Circuit breaker: drawdown at {drawdown:.1%}"

        # Check 5: Daily loss limit
        daily_pnl = await self.calculate_daily_pnl()
        if daily_pnl < -account['portfolio_value'] * 0.05:  # -5% daily limit
            return False, "Daily loss limit reached (-5%)"

        return True, "All risk checks passed"

    async def calculate_portfolio_metrics(self) -> Dict:
        """
        Calculate comprehensive portfolio performance metrics.
        """
        # Get historical equity curve
        equity_history = await self.get_equity_history(period='1y')
        returns = equity_history.pct_change().dropna()

        # Calculate metrics
        metrics = {
            # Returns
            'total_return': (equity_history[-1] / equity_history[0]) - 1,
            'annualized_return': self.annualize_return(returns),
            'daily_return': returns.iloc[-1],
            'mtd_return': self.period_return(equity_history, 'month'),
            'ytd_return': self.period_return(equity_history, 'year'),

            # Risk
            'volatility': returns.std() * np.sqrt(252),
            'downside_deviation': self.downside_deviation(returns),
            'max_drawdown': self.max_drawdown(equity_history),
            'current_drawdown': self.current_drawdown(equity_history),

            # Risk-adjusted returns
            'sharpe_ratio': self.sharpe_ratio(returns),
            'sortino_ratio': self.sortino_ratio(returns),
            'calmar_ratio': self.calmar_ratio(equity_history, returns),

            # Win rate metrics
            'win_rate': self.calculate_win_rate(),
            'profit_factor': self.profit_factor(),
            'expectancy': self.expectancy(),

            # Exposure
            'long_exposure': self.long_exposure(),
            'short_exposure': self.short_exposure(),
            'net_exposure': self.net_exposure(),
            'gross_exposure': self.gross_exposure()
        }

        return metrics
```

---

### Auth Server MCP Server

**Technology:** Node.js 18+ with Firebase Admin SDK
**Port:** 8005
**File:** `mcp_servers/auth_server/index.js`

```javascript
class AuthServer {
    constructor() {
        this.firebaseApp = admin.initializeApp({
            credential: admin.credential.cert(
                JSON.parse(process.env.FIREBASE_CONFIG)
            )
        });
        this.auth = admin.auth();
        this.db = admin.firestore();
    }

    async verifyToken(idToken) {
        /**
         * Verify Firebase JWT token.
         */
        try {
            const decodedToken = await this.auth.verifyIdToken(idToken);
            const user = await this.getUser(decodedToken.uid);
            return { valid: true, user };
        } catch (error) {
            return { valid: false, error: error.message };
        }
    }

    async createUser(email, password, profile) {
        /**
         * Create new user account.
         */
        const userRecord = await this.auth.createUser({
            email,
            password,
            emailVerified: false
        });

        // Store additional profile data
        await this.db.collection('users').doc(userRecord.uid).set({
            email,
            ...profile,
            created_at: admin.firestore.FieldValue.serverTimestamp(),
            subscription_tier: 'free',
            risk_profile: 'moderate'
        });

        return userRecord;
    }

    async getUserSettings(userId) {
        /**
         * Get user trading settings and preferences.
         */
        const doc = await this.db.collection('users').doc(userId).get();

        if (!doc.exists) {
            throw new Error('User not found');
        }

        return doc.data();
    }

    async updateRiskProfile(userId, riskProfile) {
        /**
         * Update user risk preferences.
         */
        await this.db.collection('users').doc(userId).update({
            risk_profile: riskProfile,
            updated_at: admin.firestore.FieldValue.serverTimestamp()
        });
    }
}
```

---

## Data Architecture

### Database Schema (PostgreSQL)

```sql
-- Trades table
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL,
    order_id VARCHAR(128) UNIQUE NOT NULL,
    client_order_id VARCHAR(128) UNIQUE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
    qty DECIMAL(18, 8) NOT NULL,
    price DECIMAL(18, 8) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL,
    filled_qty DECIMAL(18, 8) DEFAULT 0,
    filled_avg_price DECIMAL(18, 8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    filled_at TIMESTAMP,
    canceled_at TIMESTAMP,
    INDEX idx_user_trades (user_id, created_at DESC),
    INDEX idx_symbol_trades (symbol, created_at DESC)
);

-- Predictions table
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    prediction_type VARCHAR(20) NOT NULL,
    predicted_price DECIMAL(18, 8) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL,
    horizon VARCHAR(10) NOT NULL,
    features JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    actual_price DECIMAL(18, 8),
    actual_time TIMESTAMP,
    error DECIMAL(18, 8),
    INDEX idx_symbol_predictions (symbol, created_at DESC),
    INDEX idx_model_predictions (model_name, created_at DESC)
);

-- Portfolio snapshots table
CREATE TABLE portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL,
    equity DECIMAL(18, 2) NOT NULL,
    cash DECIMAL(18, 2) NOT NULL,
    buying_power DECIMAL(18, 2) NOT NULL,
    long_market_value DECIMAL(18, 2) NOT NULL,
    short_market_value DECIMAL(18, 2) NOT NULL,
    positions JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_snapshots (user_id, created_at DESC)
);

-- Performance metrics table
CREATE TABLE performance_metrics (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL,
    date DATE NOT NULL,
    total_return DECIMAL(10, 6),
    daily_return DECIMAL(10, 6),
    sharpe_ratio DECIMAL(10, 6),
    max_drawdown DECIMAL(10, 6),
    win_rate DECIMAL(5, 4),
    profit_factor DECIMAL(10, 6),
    num_trades INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, date),
    INDEX idx_user_metrics (user_id, date DESC)
);

-- Model training history
CREATE TABLE model_training (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    training_start TIMESTAMP NOT NULL,
    training_end TIMESTAMP NOT NULL,
    dataset_size INTEGER NOT NULL,
    train_score DECIMAL(10, 6),
    test_score DECIMAL(10, 6),
    hyperparameters JSONB,
    metrics JSONB,
    model_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_model_history (model_name, created_at DESC)
);

-- Market data cache (for commonly accessed data)
CREATE TABLE market_data_cache (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    interval VARCHAR(10),
    data JSONB NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_symbol_cache (symbol, data_type, expires_at)
);

-- System events log
CREATE TABLE system_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    component VARCHAR(50) NOT NULL,
    message TEXT,
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_events_time (created_at DESC),
    INDEX idx_events_type (event_type, created_at DESC)
);
```

### Redis Caching Strategy

```python
# Cache keys structure
CACHE_KEYS = {
    # Market data (TTL: varies)
    'market:quote:{symbol}': 10,              # 10 seconds
    'market:hist:{symbol}:{period}': 3600,    # 1 hour
    'market:indicators:{symbol}': 300,        # 5 minutes

    # Predictions (TTL: varies by horizon)
    'prediction:{symbol}:1h': 1800,           # 30 minutes
    'prediction:{symbol}:1d': 3600,           # 1 hour
    'prediction:{symbol}:1w': 86400,          # 24 hours

    # User data (TTL: 5 minutes)
    'user:{user_id}:portfolio': 300,
    'user:{user_id}:positions': 300,
    'user:{user_id}:settings': 300,

    # Model data (TTL: 1 hour)
    'model:{model_name}:weights': 3600,
    'model:ensemble:weights': 3600,

    # System state (TTL: 1 minute)
    'system:health': 60,
    'system:circuit_breakers': 60
}
```

### Data Flow

```
External APIs → YFinance Server → Raw Data → Feature Engineer → ML Models
                                      ↓                             ↓
                                 Redis Cache                   Predictions
                                      ↓                             ↓
                              Database (PostgreSQL)         Trading Signals
                                      ↓                             ↓
                               Audit Trail                Risk Manager
                                                                    ↓
                                                            Order Execution
```

---

## Machine Learning Architecture

### Model Lifecycle

```
Research → Development → Training → Validation → Deployment → Monitoring → Retraining
    ↓          ↓            ↓           ↓            ↓            ↓            ↓
Papers    Prototypes    Dataset    Backtest      API      Performance    Improve
                                   Results                  Tracking
```

### Training Pipeline

```python
class ModelTrainingPipeline:
    """
    End-to-end ML model training pipeline.
    """

    async def train_model(
        self,
        model_name: str,
        symbols: List[str],
        start_date: str,
        end_date: str
    ):
        """Full training pipeline."""

        # 1. Data Collection
        logger.info("Collecting training data...")
        data = await self.collect_data(symbols, start_date, end_date)

        # 2. Feature Engineering
        logger.info("Engineering features...")
        features, targets = await self.engineer_features(data)

        # 3. Train/Test Split (time-series aware)
        X_train, X_test, y_train, y_test = self.time_series_split(
            features, targets, test_size=0.3
        )

        # 4. Model Training
        logger.info(f"Training {model_name}...")
        model = self.get_model(model_name)

        if model_name == 'lstm':
            model = await self.train_lstm(X_train, y_train, X_test, y_test)
        elif model_name == 'xgboost':
            model = await self.train_xgboost(X_train, y_train)
        elif model_name == 'random_forest':
            model = await self.train_rf(X_train, y_train)

        # 5. Validation
        logger.info("Validating model...")
        metrics = await self.validate_model(model, X_test, y_test)

        # 6. Backtesting
        logger.info("Backtesting strategy...")
        backtest_results = await self.backtest_model(
            model, symbols, start_date, end_date
        )

        # 7. Save Model
        if metrics['test_score'] > 0.70 and backtest_results['sharpe'] > 1.5:
            logger.info("Model passes thresholds, saving...")
            model_path = await self.save_model(model, model_name)

            # Log to database
            await self.log_training(
                model_name=model_name,
                metrics=metrics,
                backtest_results=backtest_results,
                model_path=model_path
            )
        else:
            logger.warning("Model does not meet performance thresholds")

        return model, metrics, backtest_results
```

### Online Learning

```python
class OnlineLearner:
    """
    Continuously update models with new data.
    """

    def __init__(self, update_frequency='daily'):
        self.update_frequency = update_frequency
        self.buffer = ExperienceReplayBuffer(max_size=1000)

    async def update_loop(self):
        """Main update loop."""
        while True:
            # Wait for next update time
            await self.wait_for_next_update()

            # Collect new data since last update
            new_data = await self.collect_new_data()

            # Add to buffer
            self.buffer.add(new_data)

            # Update models
            for model_name in ['xgboost', 'random_forest']:
                await self.partial_update(model_name, self.buffer.sample(100))

            # Retrain LSTM less frequently (weekly)
            if self.should_retrain_lstm():
                await self.retrain_lstm()

            # Update ensemble weights
            await self.update_ensemble_weights()

            # Validate performance
            performance = await self.validate_online_performance()

            # Alert if performance degrades
            if performance['accuracy'] < 0.65:
                await self.alert_performance_degradation()

    async def partial_update(self, model_name: str, batch: List):
        """Partial fit on new batch."""
        model = self.models[model_name]
        X, y = self.prepare_batch(batch)

        if hasattr(model, 'partial_fit'):
            model.partial_fit(X, y)
        else:
            # Incremental update for XGBoost
            model = xgb.train(
                params,
                xgb.DMatrix(X, y),
                xgb_model=model
            )

        # Save updated model
        await self.save_model(model, model_name)
```

---

## API Design

### REST API Endpoints

#### Trading Endpoints

```
POST   /api/v1/orders                 - Place new order
GET    /api/v1/orders                 - List orders
GET    /api/v1/orders/:id             - Get order details
DELETE /api/v1/orders/:id             - Cancel order
GET    /api/v1/positions              - Get all positions
GET    /api/v1/positions/:symbol      - Get position for symbol
GET    /api/v1/account                - Get account info
```

#### Market Data Endpoints

```
GET    /api/v1/quotes/:symbol         - Get real-time quote
GET    /api/v1/bars/:symbol           - Get historical bars
GET    /api/v1/indicators/:symbol     - Get technical indicators
POST   /api/v1/screen                 - Screen stocks by criteria
```

#### ML Prediction Endpoints

```
POST   /api/v1/predict                - Get price prediction
GET    /api/v1/predictions/:symbol    - Get cached predictions
GET    /api/v1/models                 - List available models
GET    /api/v1/models/:name/performance - Get model performance metrics
```

#### Portfolio Endpoints

```
GET    /api/v1/portfolio              - Get portfolio summary
GET    /api/v1/portfolio/metrics      - Get performance metrics
GET    /api/v1/portfolio/risk         - Get risk analysis
POST   /api/v1/portfolio/rebalance    - Rebalance portfolio
```

### WebSocket Streams

```javascript
// Real-time market data
ws://localhost:8080/stream/quotes/:symbol

// Order status updates
ws://localhost:8080/stream/orders

// Portfolio updates
ws://localhost:8080/stream/portfolio

// ML predictions
ws://localhost:8080/stream/predictions/:symbol
```

---

## Security Architecture

### Authentication Flow

```
User → Firebase Auth → JWT Token → API Gateway → MCP Server
                           ↓
                    Token Validation
                           ↓
                    User Authorization
                           ↓
                    Rate Limiting
                           ↓
                    Request Processing
```

### API Key Management

```python
class SecureKeyManager:
    """Secure storage and retrieval of API keys."""

    def __init__(self):
        self.encryption_key = self.load_encryption_key()
        self.cipher = Fernet(self.encryption_key)

    def encrypt_key(self, plaintext: str) -> str:
        """Encrypt API key."""
        return self.cipher.encrypt(plaintext.encode()).decode()

    def decrypt_key(self, encrypted: str) -> str:
        """Decrypt API key."""
        return self.cipher.decrypt(encrypted.encode()).decode()

    def store_key(self, user_id: str, service: str, key: str):
        """Store encrypted API key."""
        encrypted = self.encrypt_key(key)
        # Store in database or secure key vault
        self.db.execute(
            "INSERT INTO api_keys (user_id, service, encrypted_key) "
            "VALUES (%s, %s, %s)",
            (user_id, service, encrypted)
        )

    def get_key(self, user_id: str, service: str) -> str:
        """Retrieve and decrypt API key."""
        result = self.db.execute(
            "SELECT encrypted_key FROM api_keys "
            "WHERE user_id = %s AND service = %s",
            (user_id, service)
        )
        encrypted = result.fetchone()[0]
        return self.decrypt_key(encrypted)
```

### Rate Limiting

```python
# Redis-based rate limiting
rate_limits = {
    'free_tier': {
        'requests_per_minute': 10,
        'orders_per_day': 50,
        'predictions_per_hour': 100
    },
    'premium_tier': {
        'requests_per_minute': 100,
        'orders_per_day': 500,
        'predictions_per_hour': 1000
    }
}
```

---

## Deployment Architecture

### Docker Compose Setup

```yaml
version: '3.8'

services:
  # MCP Orchestrator
  orchestrator:
    build: ./mcp_orchestrator
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - POSTGRES_URL=postgresql://postgres:5432/trading
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  # YFinance Trader
  yfinance-trader:
    build: ./mcp_servers/yfinance_trader
    ports:
      - "8001:8001"
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    restart: unless-stopped

  # ML Predictor
  ml-predictor:
    build: ./mcp_servers/ml_predictor
    ports:
      - "8002:8002"
    volumes:
      - ./models:/app/models
    environment:
      - REDIS_URL=redis://redis:6379
      - POSTGRES_URL=postgresql://postgres:5432/trading
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  # Alpaca Trading
  alpaca-trading:
    build: ./mcp_servers/alpaca_trading
    ports:
      - "8003:8003"
    environment:
      - ALPACA_API_KEY=${ALPACA_API_KEY}
      - ALPACA_SECRET=${ALPACA_SECRET}
      - ALPACA_PAPER=true
    restart: unless-stopped

  # Portfolio Manager
  portfolio-manager:
    build: ./mcp_servers/portfolio_manager
    ports:
      - "8004:8004"
    environment:
      - POSTGRES_URL=postgresql://postgres:5432/trading
    depends_on:
      - postgres
    restart: unless-stopped

  # Auth Server
  auth-server:
    build: ./mcp_servers/auth_server
    ports:
      - "8005:8005"
    environment:
      - FIREBASE_CONFIG=${FIREBASE_CONFIG}
    restart: unless-stopped

  # Streamlit Dashboard
  dashboard:
    build: ./web_ui
    ports:
      - "8501:8501"
    environment:
      - ORCHESTRATOR_URL=http://orchestrator:8000
    depends_on:
      - orchestrator
    restart: unless-stopped

  # PostgreSQL
  postgres:
    image: postgres:14-alpine
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=trading
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped

  # Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    restart: unless-stopped

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - ./monitoring/grafana:/etc/grafana/provisioning
      - grafana-data:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  postgres-data:
  redis-data:
  prometheus-data:
  grafana-data:
```

---

## Performance Optimization

### MacBook M4 Specific Optimizations

```python
# Use Apple's Metal Performance Shaders for neural networks
import tensorflow as tf

# Enable Metal GPU acceleration
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Use Apple Accelerate framework for NumPy
import numpy as np
np.show_config()  # Verify using Accelerate BLAS

# Memory management
import psutil

def optimize_batch_size():
    """Dynamically adjust batch size based on available memory."""
    available_memory = psutil.virtual_memory().available / (1024**3)  # GB

    if available_memory > 8:
        return 64
    elif available_memory > 4:
        return 32
    else:
        return 16

# Use Numba for JIT compilation
from numba import jit

@jit(nopython=True)
def fast_indicator_calculation(prices):
    """JIT-compiled indicator calculation."""
    # Fast computation
    return result
```

### Caching Strategy

```python
class MultiLevelCache:
    """
    Multi-level caching: Memory → Redis → Database
    """

    def __init__(self):
        self.memory_cache = {}  # L1: In-memory
        self.redis_cache = Redis()  # L2: Redis
        self.db = Database()  # L3: PostgreSQL

    async def get(self, key: str):
        # Try L1 (memory)
        if key in self.memory_cache:
            return self.memory_cache[key]

        # Try L2 (Redis)
        value = await self.redis_cache.get(key)
        if value:
            self.memory_cache[key] = value
            return value

        # Try L3 (database)
        value = await self.db.get(key)
        if value:
            await self.redis_cache.set(key, value, ttl=3600)
            self.memory_cache[key] = value
            return value

        return None
```

---

## Monitoring & Observability

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge

# Business metrics
orders_placed = Counter('orders_placed_total', 'Total orders placed')
predictions_made = Counter('predictions_made_total', 'Total predictions made')
portfolio_value = Gauge('portfolio_value_dollars', 'Current portfolio value')

# Performance metrics
prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)
order_latency = Histogram(
    'order_latency_seconds',
    'Order placement latency',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

# ML metrics
model_accuracy = Gauge('model_accuracy', 'Model accuracy', ['model_name'])
model_confidence = Gauge('model_confidence', 'Average model confidence')
```

### Logging Strategy

```python
import logging
import json

class StructuredLogger:
    """JSON structured logging for better parsing."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        self.logger.addHandler(handler)

    def log_trade(self, order: Dict):
        """Log trade execution."""
        self.logger.info("trade_executed", extra={
            "event_type": "trade",
            "order_id": order['id'],
            "symbol": order['symbol'],
            "side": order['side'],
            "qty": order['qty'],
            "price": order['price'],
            "timestamp": datetime.now().isoformat()
        })

    def log_prediction(self, prediction: Dict):
        """Log ML prediction."""
        self.logger.info("prediction_made", extra={
            "event_type": "prediction",
            "symbol": prediction['symbol'],
            "model": prediction['model'],
            "prediction": prediction['price'],
            "confidence": prediction['confidence'],
            "timestamp": datetime.now().isoformat()
        })
```

---

## Appendix

### Technology Stack Summary

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Backend** | Python | 3.11+ | MCP servers, ML |
| **Backend** | Node.js | 18+ | Trading, auth |
| **Database** | PostgreSQL | 14 | Primary storage |
| **Cache** | Redis | 7 | Fast data access |
| **ML** | TensorFlow | 2.15+ | LSTM models |
| **ML** | XGBoost | 2.0+ | Gradient boosting |
| **ML** | scikit-learn | 1.3+ | Random forest |
| **Frontend** | Streamlit | 1.30+ | Web dashboard |
| **Monitoring** | Prometheus | Latest | Metrics |
| **Monitoring** | Grafana | Latest | Visualization |
| **Container** | Docker | Latest | Deployment |

### Reference Links

- **MCP Protocol**: https://modelcontextprotocol.io
- **Alpaca API**: https://alpaca.markets/docs
- **TensorFlow**: https://www.tensorflow.org
- **XGBoost**: https://xgboost.readthedocs.io
- **Streamlit**: https://streamlit.io

---

*Last Updated: October 30, 2025*
*Version: 1.0.0*
