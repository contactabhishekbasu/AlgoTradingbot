"""
Unit Tests for Backtesting Engine

Tests for:
- Backtest engine core functionality
- Transaction cost modeling
- Mean reversion strategy
- Metrics calculation
- Walk-forward analysis

Author: Trading System Team
Date: October 2025
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtesting.engine import (
    BacktestEngine,
    BacktestConfig,
    Order,
    Position,
    Trade,
    OrderSide,
    OrderType,
    OrderStatus
)
from src.backtesting.strategies.mean_reversion import (
    MeanReversionStrategy,
    MeanReversionConfig,
    MLEnhancedMeanReversion
)
from src.backtesting.metrics import MetricsCalculator, PerformanceMetrics
from src.backtesting.walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WindowResult
)


class TestBacktestEngine:
    """Tests for BacktestEngine"""

    def test_engine_initialization(self):
        """Test engine initializes with correct config"""
        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)

        assert engine.config.initial_capital == 100000.0
        assert engine.state.cash == 100000.0
        assert engine.state.equity == 100000.0
        assert len(engine.state.positions) == 0

    def test_engine_with_invalid_config(self):
        """Test engine rejects invalid config"""
        with pytest.raises(ValueError):
            BacktestConfig(initial_capital=-1000)

        with pytest.raises(ValueError):
            BacktestConfig(initial_capital=100000, commission_rate=-0.01)

    def test_commission_calculation(self):
        """Test commission calculation"""
        config = BacktestConfig(commission_rate=0.005)
        engine = BacktestEngine(config)

        # Test: 100 shares should cost 100 * 0.005 = $0.50
        commission = engine._calculate_commission(100, 150.0)
        assert commission == 0.50

        # Test: 1000 shares should cost 1000 * 0.005 = $5.00
        commission = engine._calculate_commission(1000, 150.0)
        assert commission == 5.00

    def test_slippage_calculation(self):
        """Test slippage calculation"""
        config = BacktestConfig(slippage_rate=0.001)  # 0.1%
        engine = BacktestEngine(config)

        # Test: 100 shares @ $150 with 0.1% slippage = $15.00
        slippage = engine._calculate_slippage(100, 150.0, OrderSide.BUY)
        assert slippage == 15.0

    def test_position_opening(self):
        """Test opening a position"""
        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)

        # Open position
        engine._open_position("AAPL", 100, 150.0, datetime.now())

        # Verify position created
        assert "AAPL" in engine.state.positions
        assert engine.state.positions["AAPL"].quantity == 100
        assert engine.state.positions["AAPL"].entry_price == 150.0

        # Verify cash reduced
        assert engine.state.cash < 100000.0

    def test_position_closing(self):
        """Test closing a position"""
        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)

        # Open position
        entry_time = datetime.now()
        engine._open_position("AAPL", 100, 150.0, entry_time)
        initial_cash = engine.state.cash

        # Close position at profit
        exit_time = entry_time + timedelta(days=5)
        engine._close_position("AAPL", 160.0, exit_time)

        # Verify position closed
        assert "AAPL" not in engine.state.positions
        assert len(engine.state.trades) == 1

        # Verify trade recorded
        trade = engine.state.trades[0]
        assert trade.entry_price == 150.0
        assert trade.exit_price == 160.0
        assert trade.quantity == 100
        assert trade.pnl > 0  # Should be profitable

        # Verify cash increased
        assert engine.state.cash > initial_cash

    def test_insufficient_capital(self):
        """Test that engine rejects trades without sufficient capital"""
        config = BacktestConfig(initial_capital=1000.0)
        engine = BacktestEngine(config)

        # Try to buy more than we can afford
        engine._open_position("AAPL", 1000, 150.0, datetime.now())

        # Verify position was not created
        assert "AAPL" not in engine.state.positions

    def test_stop_loss_trigger(self):
        """Test stop loss is triggered"""
        config = BacktestConfig(
            initial_capital=100000.0,
            stop_loss_pct=0.05  # 5% stop loss
        )
        engine = BacktestEngine(config)

        # Open position
        entry_time = datetime.now()
        engine._open_position("AAPL", 100, 150.0, entry_time)

        # Create bar with 6% loss
        bar = pd.Series({
            'close': 141.0,  # -6% from 150
            'high': 145.0,
            'low': 140.0,
            'open': 145.0
        }, name=entry_time + timedelta(days=1))

        # Check exit conditions
        engine._check_exit_conditions("AAPL", bar, bar.name)

        # Position should be closed due to stop loss
        assert "AAPL" not in engine.state.positions
        assert len(engine.state.trades) == 1
        assert engine.state.trades[0].pnl < 0


class TestMeanReversionStrategy:
    """Tests for MeanReversionStrategy"""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance"""
        config = MeanReversionConfig(
            rsi_oversold=30.0,
            rsi_overbought=70.0,
            use_ml_confirmation=False
        )
        return MeanReversionStrategy(config)

    @pytest.fixture
    def mock_state(self):
        """Create mock backtest state"""
        from src.backtesting.engine import BacktestState
        return BacktestState(cash=100000.0, equity=100000.0)

    def test_strategy_initialization(self, strategy):
        """Test strategy initializes correctly"""
        assert strategy.config.rsi_oversold == 30.0
        assert strategy.config.rsi_overbought == 70.0
        assert strategy.signals_generated == 0

    def test_entry_signal_rsi_oversold(self, strategy, mock_state):
        """Test entry signal on RSI oversold"""
        bar = pd.Series({
            'close': 100.0,
            'rsi': 25.0,  # Oversold
            'williams_r': -50.0,
            'volume': 1000000
        }, name=datetime.now())

        signal = strategy.generate_signal(bar, mock_state)

        assert signal is not None
        assert signal['action'] == 'buy'
        assert 'rsi' in signal['reason']

    def test_entry_signal_williams_oversold(self, strategy, mock_state):
        """Test entry signal on Williams %R oversold"""
        bar = pd.Series({
            'close': 100.0,
            'rsi': 50.0,
            'williams_r': -85.0,  # Oversold
            'volume': 1000000
        }, name=datetime.now())

        signal = strategy.generate_signal(bar, mock_state)

        assert signal is not None
        assert signal['action'] == 'buy'
        assert 'williams' in signal['reason']

    def test_no_entry_signal_neutral(self, strategy, mock_state):
        """Test no entry signal when indicators are neutral"""
        bar = pd.Series({
            'close': 100.0,
            'rsi': 50.0,  # Neutral
            'williams_r': -50.0,  # Neutral
            'volume': 1000000
        }, name=datetime.now())

        signal = strategy.generate_signal(bar, mock_state)

        assert signal is None

    def test_exit_signal_rsi_overbought(self, strategy, mock_state):
        """Test exit signal on RSI overbought"""
        # First enter a position
        mock_state.positions['AAPL'] = Position(
            symbol='AAPL',
            quantity=100,
            entry_price=100.0,
            entry_time=datetime.now() - timedelta(days=3),
            current_price=110.0
        )

        bar = pd.Series({
            'close': 110.0,
            'rsi': 75.0,  # Overbought
            'williams_r': -10.0,
            'volume': 1000000
        }, name=datetime.now())

        signal = strategy.generate_signal(bar, mock_state)

        assert signal is not None
        assert signal['action'] == 'sell'


class TestMetricsCalculator:
    """Tests for MetricsCalculator"""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance"""
        return MetricsCalculator(risk_free_rate=0.02)

    @pytest.fixture
    def sample_results(self):
        """Create sample backtest results"""
        # Create equity curve
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        equity = 100000 * (1 + np.random.randn(252).cumsum() * 0.01)
        equity_curve = pd.DataFrame({'equity': equity}, index=dates)

        # Create sample trades
        trades = [
            Trade(
                symbol='AAPL',
                entry_time=dates[i],
                exit_time=dates[i+5],
                entry_price=100.0,
                exit_price=105.0 if i % 2 == 0 else 98.0,
                quantity=100,
                side=OrderSide.BUY,
                pnl=500.0 if i % 2 == 0 else -200.0,
                pnl_pct=0.05 if i % 2 == 0 else -0.02,
                commission=0.50,
                slippage=10.0,
                holding_period=timedelta(days=5)
            )
            for i in range(0, 100, 10)
        ]

        config = BacktestConfig(initial_capital=100000.0)

        return {
            'equity_curve': equity_curve,
            'trades': trades,
            'final_equity': equity[-1],
            'config': config
        }

    def test_metrics_calculation(self, calculator, sample_results):
        """Test comprehensive metrics calculation"""
        metrics = calculator.calculate_metrics(sample_results)

        assert isinstance(metrics, PerformanceMetrics)
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'total_return')
        assert hasattr(metrics, 'max_drawdown')
        assert hasattr(metrics, 'win_rate')

    def test_sharpe_ratio_calculation(self, calculator):
        """Test Sharpe ratio calculation"""
        # Create returns with positive mean
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.001)  # Mean ~0.1% daily

        sharpe = calculator._calculate_sharpe_ratio(returns)

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_max_drawdown_calculation(self, calculator):
        """Test maximum drawdown calculation"""
        # Create equity curve with known drawdown
        equity = pd.DataFrame({
            'equity': [100000, 110000, 105000, 95000, 100000, 110000]
        }, index=pd.date_range('2024-01-01', periods=6, freq='D'))

        dd_metrics = calculator._calculate_drawdown_metrics(equity)

        assert dd_metrics['max_drawdown'] < 0  # Drawdown is negative
        assert abs(dd_metrics['max_drawdown']) > 0.10  # >10% drawdown
        assert dd_metrics['max_drawdown_duration'] > 0

    def test_statistical_validation(self, calculator):
        """Test statistical validation"""
        # Create returns with positive mean
        returns = pd.Series(np.random.randn(100) * 0.02 + 0.005)

        validation = calculator._calculate_statistical_validation(returns)

        assert 'p_value' in validation
        assert 't_statistic' in validation
        assert 'confidence_interval_95' in validation
        assert isinstance(validation['confidence_interval_95'], tuple)


class TestWalkForwardAnalyzer:
    """Tests for WalkForwardAnalyzer"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        config = WalkForwardConfig(
            train_period_days=60,
            test_period_days=20,
            step_size_days=20
        )
        return WalkForwardAnalyzer(config)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for walk-forward analysis"""
        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        data = pd.DataFrame({
            'open': 100 + np.random.randn(200) * 5,
            'high': 105 + np.random.randn(200) * 5,
            'low': 95 + np.random.randn(200) * 5,
            'close': 100 + np.random.randn(200) * 5,
            'volume': 1000000 + np.random.randint(-100000, 100000, 200),
            'rsi': 50 + np.random.randn(200) * 15,
            'williams_r': -50 + np.random.randn(200) * 20
        }, index=dates)
        return data

    def test_window_generation(self, analyzer, sample_data):
        """Test generation of train/test windows"""
        windows = analyzer._generate_windows(sample_data)

        assert len(windows) > 0
        assert len(windows) <= 5  # Based on config and data size

        # Verify window structure
        for train_data, test_data, train_start, train_end, test_start, test_end in windows:
            assert len(train_data) == analyzer.config.train_period_days
            assert len(test_data) == analyzer.config.test_period_days
            assert train_end < test_start  # No overlap

    def test_insufficient_data(self):
        """Test handling of insufficient data"""
        config = WalkForwardConfig(
            train_period_days=100,
            test_period_days=50,
            step_size_days=50
        )
        analyzer = WalkForwardAnalyzer(config)

        # Create data that's too short
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'close': np.random.randn(50) * 10 + 100
        }, index=dates)

        windows = analyzer._generate_windows(data)
        assert len(windows) == 0  # Not enough data


class TestIntegration:
    """Integration tests for complete backtesting workflow"""

    def test_full_backtest_workflow(self):
        """Test complete backtest from data to results"""
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'open': 100 + np.random.randn(100) * 2,
            'high': 105 + np.random.randn(100) * 2,
            'low': 95 + np.random.randn(100) * 2,
            'close': 100 + np.random.randn(100) * 2,
            'volume': 1000000 + np.random.randint(-100000, 100000, 100),
            'rsi': 50 + np.random.randn(100) * 20,
            'williams_r': -50 + np.random.randn(100) * 25
        }, index=dates)

        # Ensure valid range for indicators
        data['rsi'] = data['rsi'].clip(0, 100)
        data['williams_r'] = data['williams_r'].clip(-100, 0)

        # Create strategy and engine
        strategy = MeanReversionStrategy()
        config = BacktestConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)

        # Run backtest
        results = engine.run_backtest(data, strategy, "AAPL")

        # Verify results structure
        assert 'final_equity' in results
        assert 'total_return' in results
        assert 'total_trades' in results
        assert 'equity_curve' in results
        assert 'trades' in results

        # Calculate metrics
        calculator = MetricsCalculator()
        metrics = calculator.calculate_metrics(results)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_trades >= 0
        assert -1 <= metrics.total_return <= 10  # Reasonable range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
