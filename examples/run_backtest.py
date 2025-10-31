"""
Complete Backtesting Example

This script demonstrates a complete backtesting workflow:
1. Load historical data
2. Calculate technical indicators
3. Generate ML predictions (optional)
4. Run backtest with mean reversion strategy
5. Calculate performance metrics
6. Perform walk-forward analysis
7. Generate reports and visualizations

Author: Trading System Team
Date: October 2025
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from src.data.yfinance_client import YFinanceClient
from src.indicators.technical_indicators import TechnicalIndicators
from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.backtesting.strategies.mean_reversion import MeanReversionStrategy, MeanReversionConfig
from src.backtesting.metrics import MetricsCalculator
from src.backtesting.walk_forward import WalkForwardAnalyzer, WalkForwardConfig
from src.backtesting.visualizer import BacktestVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load historical data and calculate indicators

    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data and indicators
    """
    logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")

    # Load data
    client = YFinanceClient()
    data = client.get_historical_data(symbol, start_date, end_date)

    logger.info(f"Loaded {len(data)} bars")

    # Calculate indicators
    logger.info("Calculating technical indicators...")
    indicators = TechnicalIndicators()

    data['rsi'] = indicators.rsi(data['close'])
    data['williams_r'] = indicators.williams_r(data['high'], data['low'], data['close'])
    data['macd'], data['macd_signal'], data['macd_hist'] = indicators.macd(data['close'])
    data['bb_upper'], data['bb_middle'], data['bb_lower'] = indicators.bollinger_bands(data['close'])

    # Drop NaN values
    data = data.dropna()
    logger.info(f"Data ready: {len(data)} bars with indicators")

    return data


def run_simple_backtest(
    data: pd.DataFrame,
    symbol: str,
    initial_capital: float = 100000.0
) -> tuple:
    """
    Run a simple backtest

    Args:
        data: Historical data with indicators
        symbol: Stock symbol
        initial_capital: Starting capital

    Returns:
        Tuple of (results, metrics)
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("RUNNING SIMPLE BACKTEST")
    logger.info(f"{'=' * 60}")

    # Configure backtest
    backtest_config = BacktestConfig(
        initial_capital=initial_capital,
        commission_rate=0.005,  # $0.005 per share
        slippage_rate=0.001,  # 0.1% slippage
        position_size_pct=0.10,  # 10% of capital per position
        max_positions=1,
        stop_loss_pct=0.05,  # 5% stop loss
        risk_free_rate=0.02
    )

    # Configure strategy
    strategy_config = MeanReversionConfig(
        rsi_oversold=30.0,
        rsi_overbought=70.0,
        williams_oversold=-80.0,
        williams_overbought=-20.0,
        use_ml_confirmation=False,  # No ML for simple backtest
        hold_period_min=1,
        hold_period_max=10
    )

    # Create strategy and engine
    strategy = MeanReversionStrategy(strategy_config)
    engine = BacktestEngine(backtest_config)

    # Run backtest
    logger.info(f"Running backtest on {len(data)} bars...")
    results = engine.run_backtest(data, strategy, symbol)

    # Calculate metrics
    logger.info("Calculating metrics...")
    calculator = MetricsCalculator(risk_free_rate=0.02)
    metrics = calculator.calculate_metrics(results)

    # Print results
    logger.info(f"\n{metrics}")

    return results, metrics


def run_walk_forward_analysis(
    data: pd.DataFrame,
    symbol: str,
    initial_capital: float = 100000.0
) -> Any:
    """
    Run walk-forward analysis

    Args:
        data: Historical data with indicators
        symbol: Stock symbol
        initial_capital: Starting capital

    Returns:
        WalkForwardResults
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("RUNNING WALK-FORWARD ANALYSIS")
    logger.info(f"{'=' * 60}")

    # Configure walk-forward
    wf_config = WalkForwardConfig(
        train_period_days=252,  # 1 year train
        test_period_days=21,  # 1 month test
        step_size_days=21,  # Roll forward 1 month
        retrain_models=False,  # No ML retraining for now
        anchored=False  # Rolling window
    )

    # Configure backtest
    backtest_config = BacktestConfig(
        initial_capital=initial_capital,
        commission_rate=0.005,
        slippage_rate=0.001,
        position_size_pct=0.10,
        max_positions=1,
        stop_loss_pct=0.05,
        risk_free_rate=0.02
    )

    # Configure strategy
    strategy_config = MeanReversionConfig(
        rsi_oversold=30.0,
        rsi_overbought=70.0,
        use_ml_confirmation=False
    )
    strategy = MeanReversionStrategy(strategy_config)

    # Run walk-forward analysis
    analyzer = WalkForwardAnalyzer(wf_config)
    wf_results = analyzer.run_analysis(
        data,
        strategy,
        backtest_config,
        symbol
    )

    # Print results
    logger.info(f"\n{wf_results}")

    # Generate detailed report
    report = analyzer.generate_summary_report(wf_results)
    logger.info(f"\n{report}")

    return wf_results


def generate_visualizations(
    results: dict,
    metrics: Any,
    symbol: str,
    wf_results: Any = None
):
    """
    Generate all visualizations

    Args:
        results: Backtest results
        metrics: Performance metrics
        symbol: Stock symbol
        wf_results: Optional walk-forward results
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("GENERATING VISUALIZATIONS")
    logger.info(f"{'=' * 60}")

    visualizer = BacktestVisualizer(output_dir="reports")

    # Equity curve
    logger.info("Generating equity curve...")
    visualizer.plot_equity_curve(
        results['equity_curve'],
        title=f"{symbol} - Equity Curve",
        save_path="reports/equity_curve.png"
    )

    # Drawdown
    logger.info("Generating drawdown chart...")
    visualizer.plot_drawdown(
        results['equity_curve'],
        title=f"{symbol} - Drawdown Analysis",
        save_path="reports/drawdown.png"
    )

    # Monthly returns
    logger.info("Generating monthly returns heatmap...")
    visualizer.plot_monthly_returns(
        results['equity_curve'],
        title=f"{symbol} - Monthly Returns",
        save_path="reports/monthly_returns.png"
    )

    # Trade distribution
    if results['trades']:
        logger.info("Generating trade distribution...")
        visualizer.plot_trade_distribution(
            results['trades'],
            title=f"{symbol} - Trade P&L Distribution",
            save_path="reports/trade_distribution.png"
        )

    # Rolling Sharpe
    logger.info("Generating rolling Sharpe ratio...")
    visualizer.plot_rolling_sharpe(
        results['equity_curve'],
        window=60,
        title=f"{symbol} - Rolling Sharpe Ratio (60-day)",
        save_path="reports/rolling_sharpe.png"
    )

    # Walk-forward results
    if wf_results:
        logger.info("Generating walk-forward analysis charts...")
        visualizer.plot_walk_forward_results(
            wf_results,
            title=f"{symbol} - Walk-Forward Analysis",
            save_path="reports/walk_forward.png"
        )

    # HTML report
    logger.info("Generating HTML report...")
    visualizer.generate_html_report(
        results,
        metrics,
        symbol,
        "Mean Reversion Strategy",
        output_path="reports/backtest_report.html"
    )

    logger.info("✓ All visualizations generated in reports/ directory")


def main():
    """Main execution function"""
    logger.info("=" * 70)
    logger.info("PHASE 3: BACKTESTING ENGINE - COMPREHENSIVE EXAMPLE")
    logger.info("=" * 70)

    # Configuration
    SYMBOL = "AAPL"
    START_DATE = "2020-01-01"
    END_DATE = "2024-10-30"
    INITIAL_CAPITAL = 100000.0

    try:
        # Step 1: Load and prepare data
        data = load_and_prepare_data(SYMBOL, START_DATE, END_DATE)

        # Step 2: Run simple backtest
        results, metrics = run_simple_backtest(data, SYMBOL, INITIAL_CAPITAL)

        # Step 3: Run walk-forward analysis
        wf_results = run_walk_forward_analysis(data, SYMBOL, INITIAL_CAPITAL)

        # Step 4: Generate visualizations
        generate_visualizations(results, metrics, SYMBOL, wf_results)

        # Final summary
        logger.info(f"\n{'=' * 70}")
        logger.info("BACKTESTING COMPLETE")
        logger.info(f"{'=' * 70}")
        logger.info(f"\nFinal Results for {SYMBOL}:")
        logger.info(f"  Total Return:     {metrics.total_return:>10.2%}")
        logger.info(f"  Sharpe Ratio:     {metrics.sharpe_ratio:>10.2f}")
        logger.info(f"  Max Drawdown:     {metrics.max_drawdown:>10.2%}")
        logger.info(f"  Win Rate:         {metrics.win_rate:>10.2%}")
        logger.info(f"  Total Trades:     {metrics.total_trades:>10}")
        logger.info(f"\nWalk-Forward Validation:")
        logger.info(f"  Windows Tested:   {wf_results.total_windows:>10}")
        logger.info(f"  Profitable:       {wf_results.profitable_windows:>10} ({wf_results.profitable_pct:.1%})")
        logger.info(f"  Avg Sharpe:       {wf_results.avg_sharpe:>10.2f}")
        logger.info(f"  p-value:          {wf_results.returns_p_value:>10.4f}")

        # Validation status
        if wf_results._is_valid():
            logger.info(f"\n✅ VALIDATION PASSED - Strategy meets all criteria")
            logger.info("   Recommend proceeding to paper trading")
        else:
            logger.info(f"\n❌ VALIDATION FAILED - Strategy needs optimization")
            logger.info("   Review metrics and adjust parameters")

        logger.info(f"\nReports saved to: reports/")
        logger.info(f"  - reports/equity_curve.png")
        logger.info(f"  - reports/drawdown.png")
        logger.info(f"  - reports/monthly_returns.png")
        logger.info(f"  - reports/trade_distribution.png")
        logger.info(f"  - reports/rolling_sharpe.png")
        logger.info(f"  - reports/walk_forward.png")
        logger.info(f"  - reports/backtest_report.html")

    except Exception as e:
        logger.error(f"Error during backtesting: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
