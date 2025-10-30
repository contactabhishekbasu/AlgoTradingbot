-- AlgoTradingbot Database Schema
-- Initial setup for PostgreSQL 14+
-- Version: 1.0.0
-- Date: 2025-10-30

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Set timezone
SET timezone = 'UTC';

-- ============================================================================
-- TABLE: market_data
-- Purpose: Store historical OHLCV price data
-- ============================================================================
CREATE TABLE IF NOT EXISTS market_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(12, 4) NOT NULL CHECK (open > 0),
    high DECIMAL(12, 4) NOT NULL CHECK (high > 0),
    low DECIMAL(12, 4) NOT NULL CHECK (low > 0),
    close DECIMAL(12, 4) NOT NULL CHECK (close > 0),
    volume BIGINT NOT NULL CHECK (volume >= 0),
    adjusted_close DECIMAL(12, 4),
    data_source VARCHAR(50) DEFAULT 'yfinance',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timestamp)
);

-- Indexes for market_data
CREATE INDEX idx_market_data_symbol_timestamp ON market_data(symbol, timestamp DESC);
CREATE INDEX idx_market_data_timestamp ON market_data(timestamp DESC);
CREATE INDEX idx_market_data_symbol_date ON market_data(symbol, DATE(timestamp));

-- Add constraint to ensure OHLC relationship
ALTER TABLE market_data ADD CONSTRAINT chk_ohlc
    CHECK (low <= open AND low <= close AND low <= high AND high >= open AND high >= close);

-- ============================================================================
-- TABLE: technical_indicators
-- Purpose: Store pre-calculated technical indicators
-- ============================================================================
CREATE TABLE IF NOT EXISTS technical_indicators (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    indicator_name VARCHAR(50) NOT NULL,
    indicator_value DECIMAL(12, 6),
    parameters JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_technical_indicators UNIQUE(symbol, timestamp, indicator_name, parameters)
);

-- Indexes for technical_indicators
CREATE INDEX idx_indicators_symbol_timestamp ON technical_indicators(symbol, timestamp DESC);
CREATE INDEX idx_indicators_name ON technical_indicators(indicator_name);
CREATE INDEX idx_indicators_parameters ON technical_indicators USING GIN (parameters);

-- ============================================================================
-- TABLE: model_states
-- Purpose: Store ML model versions and metadata
-- ============================================================================
CREATE TABLE IF NOT EXISTS model_states (
    id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    model_type VARCHAR(20) NOT NULL CHECK (model_type IN ('lstm', 'xgboost', 'random_forest', 'ensemble')),
    model_path TEXT,  -- Path to model file (instead of storing in DB)
    model_size_mb DECIMAL(10, 2),
    hyperparameters JSONB DEFAULT '{}'::jsonb,
    training_metrics JSONB DEFAULT '{}'::jsonb,
    training_dataset_hash VARCHAR(64),
    trained_on_data_start TIMESTAMPTZ,
    trained_on_data_end TIMESTAMPTZ,
    training_duration_seconds INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(100) DEFAULT 'system',
    is_active BOOLEAN DEFAULT TRUE,
    notes TEXT,
    CONSTRAINT uq_model_version UNIQUE(model_name, version)
);

-- Indexes for model_states
CREATE INDEX idx_model_states_active ON model_states(model_name, is_active, created_at DESC);
CREATE INDEX idx_model_states_type ON model_states(model_type);
CREATE INDEX idx_model_states_created ON model_states(created_at DESC);

-- ============================================================================
-- TABLE: predictions
-- Purpose: Store model predictions for analysis
-- ============================================================================
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    prediction_uuid UUID DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    prediction_type VARCHAR(20) NOT NULL CHECK (prediction_type IN ('price', 'direction', 'signal', 'probability')),
    predicted_value DECIMAL(12, 6),
    predicted_direction VARCHAR(10) CHECK (predicted_direction IN ('up', 'down', 'neutral')),
    confidence DECIMAL(5, 4) CHECK (confidence >= 0 AND confidence <= 1),
    features_used JSONB DEFAULT '{}'::jsonb,
    actual_value DECIMAL(12, 6),  -- Filled in later for validation
    prediction_error DECIMAL(12, 6),
    is_accurate BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    validated_at TIMESTAMPTZ
);

-- Indexes for predictions
CREATE INDEX idx_predictions_symbol_timestamp ON predictions(symbol, timestamp DESC);
CREATE INDEX idx_predictions_model ON predictions(model_name, model_version);
CREATE INDEX idx_predictions_uuid ON predictions(prediction_uuid);
CREATE INDEX idx_predictions_accuracy ON predictions(is_accurate) WHERE is_accurate IS NOT NULL;

-- ============================================================================
-- TABLE: backtest_results
-- Purpose: Store backtest run metadata and summary metrics
-- ============================================================================
CREATE TABLE IF NOT EXISTS backtest_results (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID DEFAULT uuid_generate_v4() UNIQUE,
    run_name VARCHAR(200),
    strategy_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    start_date TIMESTAMPTZ NOT NULL,
    end_date TIMESTAMPTZ NOT NULL,
    initial_capital DECIMAL(15, 2) NOT NULL CHECK (initial_capital > 0),
    final_capital DECIMAL(15, 2) NOT NULL,
    total_return DECIMAL(10, 4),
    cagr DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    sortino_ratio DECIMAL(10, 4),
    calmar_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    max_drawdown_duration_days INTEGER,
    volatility DECIMAL(10, 4),
    win_rate DECIMAL(5, 4),
    profit_factor DECIMAL(10, 4),
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    average_win DECIMAL(12, 4),
    average_loss DECIMAL(12, 4),
    largest_win DECIMAL(12, 4),
    largest_loss DECIMAL(12, 4),
    expectancy DECIMAL(12, 4),
    configuration JSONB DEFAULT '{}'::jsonb,
    metrics JSONB DEFAULT '{}'::jsonb,
    execution_time_seconds INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(100) DEFAULT 'system',
    notes TEXT
);

-- Indexes for backtest_results
CREATE INDEX idx_backtest_results_run_id ON backtest_results(run_id);
CREATE INDEX idx_backtest_results_strategy ON backtest_results(strategy_name, created_at DESC);
CREATE INDEX idx_backtest_results_symbol ON backtest_results(symbol);
CREATE INDEX idx_backtest_results_date_range ON backtest_results(start_date, end_date);
CREATE INDEX idx_backtest_results_sharpe ON backtest_results(sharpe_ratio DESC);

-- ============================================================================
-- TABLE: backtest_trades
-- Purpose: Store individual trades from backtest runs
-- ============================================================================
CREATE TABLE IF NOT EXISTS backtest_trades (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES backtest_results(run_id) ON DELETE CASCADE,
    trade_number INTEGER NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    direction VARCHAR(10) NOT NULL CHECK (direction IN ('buy', 'sell', 'short', 'cover')),
    shares DECIMAL(15, 6) NOT NULL CHECK (shares > 0),
    price DECIMAL(12, 4) NOT NULL CHECK (price > 0),
    commission DECIMAL(10, 4) NOT NULL DEFAULT 0,
    slippage_cost DECIMAL(10, 4) NOT NULL DEFAULT 0,
    total_cost DECIMAL(15, 4) NOT NULL,
    pnl DECIMAL(15, 4),
    pnl_percent DECIMAL(10, 4),
    cumulative_pnl DECIMAL(15, 4),
    portfolio_value DECIMAL(15, 4),
    position_size_percent DECIMAL(5, 4),
    hold_period_days INTEGER,
    entry_reason TEXT,
    exit_reason TEXT,
    signal_strength DECIMAL(5, 4)
);

-- Indexes for backtest_trades
CREATE INDEX idx_backtest_trades_run_id ON backtest_trades(run_id, timestamp);
CREATE INDEX idx_backtest_trades_symbol ON backtest_trades(symbol);
CREATE INDEX idx_backtest_trades_direction ON backtest_trades(direction);
CREATE INDEX idx_backtest_trades_pnl ON backtest_trades(pnl DESC);

-- ============================================================================
-- TABLE: system_logs
-- Purpose: Store system events and errors
-- ============================================================================
CREATE TABLE IF NOT EXISTS system_logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    log_level VARCHAR(10) NOT NULL CHECK (log_level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    component VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    details JSONB DEFAULT '{}'::jsonb,
    user_id VARCHAR(100),
    session_id VARCHAR(100),
    error_traceback TEXT
);

-- Indexes for system_logs
CREATE INDEX idx_system_logs_timestamp ON system_logs(timestamp DESC);
CREATE INDEX idx_system_logs_level ON system_logs(log_level);
CREATE INDEX idx_system_logs_component ON system_logs(component);

-- ============================================================================
-- TABLE: configuration
-- Purpose: Store system configuration key-value pairs
-- ============================================================================
CREATE TABLE IF NOT EXISTS configuration (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(100) NOT NULL UNIQUE,
    config_value TEXT,
    config_type VARCHAR(20) DEFAULT 'string' CHECK (config_type IN ('string', 'integer', 'float', 'boolean', 'json')),
    description TEXT,
    is_secret BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    updated_by VARCHAR(100) DEFAULT 'system'
);

-- Indexes for configuration
CREATE INDEX idx_configuration_key ON configuration(config_key);

-- ============================================================================
-- VIEWS
-- ============================================================================

-- View: Latest model versions
CREATE OR REPLACE VIEW v_latest_models AS
SELECT DISTINCT ON (model_name)
    id,
    model_name,
    version,
    model_type,
    created_at,
    training_metrics,
    is_active
FROM model_states
WHERE is_active = TRUE
ORDER BY model_name, created_at DESC;

-- View: Backtest performance summary
CREATE OR REPLACE VIEW v_backtest_summary AS
SELECT
    strategy_name,
    COUNT(*) as total_runs,
    AVG(sharpe_ratio) as avg_sharpe_ratio,
    MAX(sharpe_ratio) as best_sharpe_ratio,
    AVG(total_return) as avg_return,
    AVG(max_drawdown) as avg_max_drawdown,
    AVG(win_rate) as avg_win_rate
FROM backtest_results
GROUP BY strategy_name;

-- View: Recent predictions accuracy
CREATE OR REPLACE VIEW v_recent_predictions_accuracy AS
SELECT
    model_name,
    model_version,
    DATE(timestamp) as prediction_date,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN is_accurate THEN 1 ELSE 0 END) as accurate_predictions,
    ROUND(SUM(CASE WHEN is_accurate THEN 1 ELSE 0 END)::numeric / COUNT(*)::numeric, 4) as accuracy_rate,
    AVG(confidence) as avg_confidence
FROM predictions
WHERE timestamp > NOW() - INTERVAL '30 days'
    AND is_accurate IS NOT NULL
GROUP BY model_name, model_version, DATE(timestamp)
ORDER BY prediction_date DESC;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function: Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger: Auto-update updated_at for market_data
CREATE TRIGGER update_market_data_updated_at BEFORE UPDATE ON market_data
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Trigger: Auto-update updated_at for configuration
CREATE TRIGGER update_configuration_updated_at BEFORE UPDATE ON configuration
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function: Calculate trade statistics for a backtest run
CREATE OR REPLACE FUNCTION calculate_backtest_stats(p_run_id UUID)
RETURNS TABLE (
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    win_rate DECIMAL,
    avg_win DECIMAL,
    avg_loss DECIMAL,
    profit_factor DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::INTEGER as total_trades,
        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::INTEGER as winning_trades,
        SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END)::INTEGER as losing_trades,
        CASE
            WHEN COUNT(*) > 0 THEN
                ROUND(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::numeric / COUNT(*)::numeric, 4)
            ELSE 0
        END as win_rate,
        AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
        AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss,
        CASE
            WHEN SUM(CASE WHEN pnl < 0 THEN ABS(pnl) END) > 0 THEN
                ABS(SUM(CASE WHEN pnl > 0 THEN pnl END) / SUM(CASE WHEN pnl < 0 THEN ABS(pnl) END))
            ELSE NULL
        END as profit_factor
    FROM backtest_trades
    WHERE run_id = p_run_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Insert default configuration
INSERT INTO configuration (config_key, config_value, config_type, description) VALUES
    ('system_version', '0.1.0', 'string', 'Current system version'),
    ('market_hours_start', '09:30', 'string', 'Market opening time (ET)'),
    ('market_hours_end', '16:00', 'string', 'Market closing time (ET)'),
    ('default_commission', '0.005', 'float', 'Default commission per share ($)'),
    ('default_slippage', '0.001', 'float', 'Default slippage (%)'),
    ('max_position_size', '0.10', 'float', 'Maximum position size as fraction of portfolio'),
    ('risk_free_rate', '0.04', 'float', 'Risk-free rate for Sharpe calculation'),
    ('cache_ttl_seconds', '300', 'integer', 'Redis cache TTL in seconds'),
    ('model_update_frequency', 'daily', 'string', 'How often to retrain models')
ON CONFLICT (config_key) DO NOTHING;

-- ============================================================================
-- GRANTS (adjust based on your security requirements)
-- ============================================================================

-- Grant permissions to trading_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO trading_user;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE market_data IS 'Historical OHLCV price data from various sources';
COMMENT ON TABLE technical_indicators IS 'Pre-calculated technical indicators for performance optimization';
COMMENT ON TABLE model_states IS 'ML model versions, metadata, and training information';
COMMENT ON TABLE predictions IS 'Model predictions with confidence scores and validation results';
COMMENT ON TABLE backtest_results IS 'Summary results from backtesting runs';
COMMENT ON TABLE backtest_trades IS 'Individual trade details from backtesting';
COMMENT ON TABLE system_logs IS 'Application logs and error tracking';
COMMENT ON TABLE configuration IS 'System configuration key-value store';

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE 'AlgoTradingbot database schema initialized successfully!';
    RAISE NOTICE 'Schema version: 1.0.0';
    RAISE NOTICE 'Tables created: 8';
    RAISE NOTICE 'Views created: 3';
    RAISE NOTICE 'Functions created: 2';
END $$;
