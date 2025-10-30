-- Initial database schema for AI Trading System
-- Version: 1.0.0
-- Created: 2025-10-30

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Market data (historical prices)
CREATE TABLE IF NOT EXISTS market_data (
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
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timestamp)
);

-- Indexes for market_data
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp
    ON market_data(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp
    ON market_data(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol
    ON market_data(symbol);

-- Technical indicators (pre-calculated)
CREATE TABLE IF NOT EXISTS technical_indicators (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    indicator_name VARCHAR(50) NOT NULL,
    indicator_value DECIMAL(12, 6),
    parameters JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timestamp, indicator_name, parameters)
);

-- Index for technical_indicators
CREATE INDEX IF NOT EXISTS idx_indicators_symbol_timestamp
    ON technical_indicators(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_indicators_name
    ON technical_indicators(indicator_name);
CREATE INDEX IF NOT EXISTS idx_indicators_params
    ON technical_indicators USING GIN(parameters);

-- ML model states (versioned models)
CREATE TABLE IF NOT EXISTS model_states (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    model_type VARCHAR(20) NOT NULL,
    model_data BYTEA,
    hyperparameters JSONB,
    training_metrics JSONB,
    trained_on_data_start TIMESTAMPTZ,
    trained_on_data_end TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(model_name, version)
);

-- Index for model_states
CREATE INDEX IF NOT EXISTS idx_model_states_active
    ON model_states(model_name, is_active, created_at DESC);

-- Predictions (model outputs)
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    prediction_type VARCHAR(20) NOT NULL,
    predicted_value DECIMAL(12, 6),
    confidence DECIMAL(5, 4),
    features_used JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for predictions
CREATE INDEX IF NOT EXISTS idx_predictions_symbol_timestamp
    ON predictions(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_model
    ON predictions(model_name, model_version);

-- Backtesting results
CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL DEFAULT uuid_generate_v4(),
    strategy_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    start_date TIMESTAMPTZ NOT NULL,
    end_date TIMESTAMPTZ NOT NULL,
    initial_capital DECIMAL(15, 2) NOT NULL,
    final_capital DECIMAL(15, 2) NOT NULL,
    total_return DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    sortino_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    win_rate DECIMAL(5, 4),
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    avg_win DECIMAL(12, 4),
    avg_loss DECIMAL(12, 4),
    profit_factor DECIMAL(10, 4),
    configuration JSONB,
    metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(run_id)
);

-- Index for backtest_results
CREATE INDEX IF NOT EXISTS idx_backtest_results_run_id
    ON backtest_results(run_id);
CREATE INDEX IF NOT EXISTS idx_backtest_results_strategy
    ON backtest_results(strategy_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_results_symbol
    ON backtest_results(symbol, created_at DESC);

-- Backtest trades (individual trades from backtests)
CREATE TABLE IF NOT EXISTS backtest_trades (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    direction VARCHAR(10) NOT NULL,
    shares DECIMAL(15, 6) NOT NULL,
    price DECIMAL(12, 4) NOT NULL,
    commission DECIMAL(10, 4) NOT NULL,
    slippage_cost DECIMAL(10, 4) NOT NULL,
    total_cost DECIMAL(15, 4) NOT NULL,
    pnl DECIMAL(15, 4),
    portfolio_value DECIMAL(15, 4),
    trade_type VARCHAR(20),
    stop_loss DECIMAL(12, 4),
    take_profit DECIMAL(12, 4),
    FOREIGN KEY (run_id) REFERENCES backtest_results(run_id) ON DELETE CASCADE
);

-- Index for backtest_trades
CREATE INDEX IF NOT EXISTS idx_backtest_trades_run_id
    ON backtest_trades(run_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_symbol
    ON backtest_trades(symbol, timestamp DESC);

-- Data quality logs
CREATE TABLE IF NOT EXISTS data_quality_logs (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    check_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    issue_level VARCHAR(10) NOT NULL,
    issue_category VARCHAR(50) NOT NULL,
    issue_message TEXT,
    details JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for data_quality_logs
CREATE INDEX IF NOT EXISTS idx_data_quality_logs_symbol
    ON data_quality_logs(symbol, check_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_data_quality_logs_level
    ON data_quality_logs(issue_level);

-- System logs (for application-level logging)
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    level VARCHAR(10) NOT NULL,
    logger_name VARCHAR(100),
    message TEXT,
    context JSONB,
    error_trace TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for system_logs
CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp
    ON system_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_logs_level
    ON system_logs(level, timestamp DESC);

-- Performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15, 4),
    metric_unit VARCHAR(20),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tags JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for performance_metrics
CREATE INDEX IF NOT EXISTS idx_performance_metrics_name
    ON performance_metrics(metric_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp
    ON performance_metrics(timestamp DESC);

-- Update updated_at timestamp function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for market_data updated_at
CREATE TRIGGER update_market_data_updated_at
    BEFORE UPDATE ON market_data
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (assuming trading_user exists)
DO $$
BEGIN
    IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'trading_user') THEN
        GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_user;
        GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_user;
    END IF;
END
$$;

-- Create views for common queries

-- Latest prices view
CREATE OR REPLACE VIEW latest_prices AS
SELECT DISTINCT ON (symbol)
    symbol,
    timestamp,
    close as price,
    volume,
    created_at
FROM market_data
ORDER BY symbol, timestamp DESC;

-- Model performance view
CREATE OR REPLACE VIEW model_performance AS
SELECT
    model_name,
    model_version,
    COUNT(*) as total_predictions,
    AVG(confidence) as avg_confidence,
    MAX(created_at) as last_prediction
FROM predictions
GROUP BY model_name, model_version
ORDER BY last_prediction DESC;

-- Backtest summary view
CREATE OR REPLACE VIEW backtest_summary AS
SELECT
    strategy_name,
    symbol,
    COUNT(*) as total_runs,
    AVG(total_return) as avg_return,
    AVG(sharpe_ratio) as avg_sharpe,
    AVG(max_drawdown) as avg_drawdown,
    AVG(win_rate) as avg_win_rate,
    MAX(created_at) as last_run
FROM backtest_results
GROUP BY strategy_name, symbol
ORDER BY avg_sharpe DESC;

-- Insert initial data (optional)
INSERT INTO system_logs (level, logger_name, message, context)
VALUES ('INFO', 'database', 'Database initialized successfully', '{"version": "1.0.0"}'::jsonb)
ON CONFLICT DO NOTHING;

-- Database initialization complete
SELECT 'Database schema initialized successfully' as status;
