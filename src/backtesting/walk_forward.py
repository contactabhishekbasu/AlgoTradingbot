"""
Walk-Forward Analysis Framework

This module implements walk-forward analysis for robust strategy validation.
Walk-forward analysis tests the strategy on rolling out-of-sample periods
to prevent overfitting and validate robustness.

Key Features:
- Rolling window approach (no look-ahead bias)
- Train/test split for each window
- Model retraining for each period
- Comprehensive aggregation of results
- Statistical validation across windows

Author: Trading System Team
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats

from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.backtesting.metrics import MetricsCalculator, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis"""
    train_period_days: int = 252  # 1 year training period
    test_period_days: int = 21  # 1 month test period
    step_size_days: int = 21  # Roll forward by 1 month
    min_train_samples: int = 100  # Minimum samples for training
    retrain_models: bool = True  # Whether to retrain models each window
    anchored: bool = False  # If True, training window grows; if False, rolls

    def __post_init__(self):
        """Validate configuration"""
        if self.train_period_days < self.min_train_samples:
            raise ValueError("Train period must be >= min_train_samples")
        if self.test_period_days <= 0:
            raise ValueError("Test period must be positive")
        if self.step_size_days <= 0:
            raise ValueError("Step size must be positive")


@dataclass
class WindowResult:
    """Results for a single walk-forward window"""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    metrics: PerformanceMetrics
    trades: List[Any]
    equity_curve: pd.DataFrame
    model_trained: bool = False


@dataclass
class WalkForwardResults:
    """Aggregated results from walk-forward analysis"""
    config: WalkForwardConfig
    windows: List[WindowResult]

    # Aggregated metrics
    total_windows: int = 0
    profitable_windows: int = 0
    profitable_pct: float = 0.0

    # Performance across windows
    avg_sharpe: float = 0.0
    std_sharpe: float = 0.0
    min_sharpe: float = 0.0
    max_sharpe: float = 0.0

    avg_return: float = 0.0
    std_return: float = 0.0

    avg_win_rate: float = 0.0
    avg_drawdown: float = 0.0

    # Statistical validation
    returns_t_statistic: float = 0.0
    returns_p_value: float = 0.0
    sharpe_t_statistic: float = 0.0
    sharpe_p_value: float = 0.0
    consistency_score: float = 0.0  # % windows with Sharpe > 1.0

    # Combined equity curve
    combined_equity_curve: Optional[pd.DataFrame] = None

    def __str__(self) -> str:
        """String representation of results"""
        return f"""
Walk-Forward Analysis Results
{'=' * 60}

OVERVIEW
  Total Windows:           {self.total_windows:>10}
  Profitable Windows:      {self.profitable_windows:>10} ({self.profitable_pct:.1%})
  Consistency Score:       {self.consistency_score:>10.1%} (Sharpe > 1.0)

PERFORMANCE ACROSS WINDOWS
  Average Sharpe:          {self.avg_sharpe:>10.2f} ± {self.std_sharpe:.2f}
  Sharpe Range:            [{self.min_sharpe:.2f}, {self.max_sharpe:.2f}]
  Average Return:          {self.avg_return:>10.2%} ± {self.std_return:.2%}
  Average Win Rate:        {self.avg_win_rate:>10.2%}
  Average Drawdown:        {self.avg_drawdown:>10.2%}

STATISTICAL VALIDATION
  Returns t-statistic:     {self.returns_t_statistic:>10.2f}
  Returns p-value:         {self.returns_p_value:>10.4f}
  Sharpe t-statistic:      {self.sharpe_t_statistic:>10.2f}
  Sharpe p-value:          {self.sharpe_p_value:>10.4f}

VALIDATION: {'✅ PASSED' if self._is_valid() else '❌ FAILED'}
{'=' * 60}
        """

    def _is_valid(self) -> bool:
        """Check if results meet validation criteria"""
        criteria = [
            self.profitable_pct >= 0.70,  # 70%+ windows profitable
            self.avg_sharpe >= 1.5,  # Average Sharpe > 1.5
            self.returns_p_value < 0.05,  # Statistically significant
            self.consistency_score >= 0.70  # 70%+ windows with Sharpe > 1.0
        ]
        return all(criteria)


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis Implementation

    This analyzer performs rigorous out-of-sample testing by:
    1. Splitting data into train/test windows
    2. Training strategy on in-sample data
    3. Testing on out-of-sample data
    4. Rolling forward and repeating
    5. Aggregating and validating results

    Example:
        >>> config = WalkForwardConfig(train_period_days=252, test_period_days=21)
        >>> analyzer = WalkForwardAnalyzer(config)
        >>> results = analyzer.run_analysis(data, strategy, backtest_config)
    """

    def __init__(self, config: WalkForwardConfig):
        """
        Initialize walk-forward analyzer

        Args:
            config: Walk-forward configuration
        """
        self.config = config
        self.metrics_calculator = MetricsCalculator()
        logger.info(f"Initialized WalkForwardAnalyzer")
        logger.info(f"Train period: {config.train_period_days} days")
        logger.info(f"Test period: {config.test_period_days} days")
        logger.info(f"Step size: {config.step_size_days} days")

    def run_analysis(
        self,
        data: pd.DataFrame,
        strategy: Any,
        backtest_config: BacktestConfig,
        symbol: str = "STOCK",
        ml_trainer: Optional[Any] = None
    ) -> WalkForwardResults:
        """
        Run walk-forward analysis

        Args:
            data: Full dataset with features and indicators
            strategy: Trading strategy object
            backtest_config: Backtest configuration
            symbol: Symbol being tested
            ml_trainer: Optional ML trainer for model retraining

        Returns:
            WalkForwardResults with comprehensive analysis
        """
        logger.info(f"Starting walk-forward analysis on {symbol}")
        logger.info(f"Data range: {data.index[0]} to {data.index[-1]} ({len(data)} bars)")

        # Generate windows
        windows = self._generate_windows(data)
        logger.info(f"Generated {len(windows)} windows")

        if len(windows) == 0:
            logger.error("No valid windows generated. Check data length and configuration.")
            raise ValueError("Insufficient data for walk-forward analysis")

        # Run backtest for each window
        window_results = []
        for i, (train_data, test_data, train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"\nWindow {i+1}/{len(windows)}")
            logger.info(f"  Train: {train_start} to {train_end} ({len(train_data)} bars)")
            logger.info(f"  Test:  {test_start} to {test_end} ({len(test_data)} bars)")

            # Retrain models if enabled
            model_trained = False
            if self.config.retrain_models and ml_trainer is not None:
                try:
                    logger.info("  Retraining models...")
                    ml_trainer.train_on_window(train_data, symbol)
                    model_trained = True
                    logger.info("  ✓ Models retrained")
                except Exception as e:
                    logger.warning(f"  ⚠ Model retraining failed: {e}")

            # Run backtest on test period
            try:
                engine = BacktestEngine(backtest_config)
                backtest_results = engine.run_backtest(test_data, strategy, symbol)

                # Calculate metrics
                metrics = self.metrics_calculator.calculate_metrics(backtest_results)

                # Store window result
                window_result = WindowResult(
                    window_id=i + 1,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    metrics=metrics,
                    trades=backtest_results['trades'],
                    equity_curve=backtest_results['equity_curve'],
                    model_trained=model_trained
                )
                window_results.append(window_result)

                logger.info(f"  Results: Return={metrics.total_return:.2%}, "
                          f"Sharpe={metrics.sharpe_ratio:.2f}, "
                          f"Trades={metrics.total_trades}")

            except Exception as e:
                logger.error(f"  ❌ Window {i+1} failed: {e}")
                continue

        # Aggregate results
        logger.info(f"\nAggregating results from {len(window_results)} windows...")
        aggregated_results = self._aggregate_results(window_results)

        logger.info("Walk-forward analysis complete!")
        return aggregated_results

    def _generate_windows(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, datetime, datetime, datetime, datetime]]:
        """
        Generate train/test windows

        Returns:
            List of (train_data, test_data, train_start, train_end, test_start, test_end)
        """
        windows = []
        data_len = len(data)

        # Starting index
        start_idx = 0
        anchor_idx = 0  # For anchored windows

        while True:
            # Calculate window indices
            if self.config.anchored:
                train_start_idx = anchor_idx
            else:
                train_start_idx = start_idx

            train_end_idx = start_idx + self.config.train_period_days
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + self.config.test_period_days

            # Check if we have enough data
            if test_end_idx > data_len:
                break

            # Extract data
            train_data = data.iloc[train_start_idx:train_end_idx]
            test_data = data.iloc[test_start_idx:test_end_idx]

            # Validate minimum samples
            if len(train_data) < self.config.min_train_samples:
                logger.warning(f"Skipping window: insufficient training data ({len(train_data)} < {self.config.min_train_samples})")
                break

            if len(test_data) == 0:
                break

            # Store window
            windows.append((
                train_data,
                test_data,
                train_data.index[0],
                train_data.index[-1],
                test_data.index[0],
                test_data.index[-1]
            ))

            # Move to next window
            start_idx += self.config.step_size_days

        return windows

    def _aggregate_results(self, window_results: List[WindowResult]) -> WalkForwardResults:
        """Aggregate results across all windows"""
        if not window_results:
            raise ValueError("No window results to aggregate")

        # Extract metrics
        sharpe_ratios = [w.metrics.sharpe_ratio for w in window_results]
        returns = [w.metrics.total_return for w in window_results]
        win_rates = [w.metrics.win_rate for w in window_results]
        drawdowns = [w.metrics.max_drawdown for w in window_results]

        # Count profitable windows
        profitable_windows = sum(1 for r in returns if r > 0)
        profitable_pct = profitable_windows / len(window_results)

        # Sharpe statistics
        avg_sharpe = np.mean(sharpe_ratios)
        std_sharpe = np.std(sharpe_ratios)
        min_sharpe = np.min(sharpe_ratios)
        max_sharpe = np.max(sharpe_ratios)

        # Return statistics
        avg_return = np.mean(returns)
        std_return = np.std(returns)

        # Other metrics
        avg_win_rate = np.mean(win_rates)
        avg_drawdown = np.mean(drawdowns)

        # Consistency score (% windows with Sharpe > 1.0)
        consistency_score = sum(1 for s in sharpe_ratios if s > 1.0) / len(sharpe_ratios)

        # Statistical validation
        # T-test: null hypothesis is that mean return = 0
        returns_t_stat, returns_p_value = stats.ttest_1samp(returns, 0)

        # T-test: null hypothesis is that mean Sharpe = 0
        sharpe_t_stat, sharpe_p_value = stats.ttest_1samp(sharpe_ratios, 0)

        # Combine equity curves
        combined_equity = self._combine_equity_curves(window_results)

        results = WalkForwardResults(
            config=self.config,
            windows=window_results,
            total_windows=len(window_results),
            profitable_windows=profitable_windows,
            profitable_pct=profitable_pct,
            avg_sharpe=avg_sharpe,
            std_sharpe=std_sharpe,
            min_sharpe=min_sharpe,
            max_sharpe=max_sharpe,
            avg_return=avg_return,
            std_return=std_return,
            avg_win_rate=avg_win_rate,
            avg_drawdown=avg_drawdown,
            returns_t_statistic=returns_t_stat,
            returns_p_value=returns_p_value,
            sharpe_t_statistic=sharpe_t_stat,
            sharpe_p_value=sharpe_p_value,
            consistency_score=consistency_score,
            combined_equity_curve=combined_equity
        )

        return results

    def _combine_equity_curves(self, window_results: List[WindowResult]) -> pd.DataFrame:
        """Combine equity curves from all windows"""
        combined_equity = []

        for window in window_results:
            equity_curve = window.equity_curve.copy()
            equity_curve['window_id'] = window.window_id
            combined_equity.append(equity_curve)

        if combined_equity:
            return pd.concat(combined_equity)
        else:
            return pd.DataFrame()

    def generate_summary_report(self, results: WalkForwardResults) -> str:
        """Generate detailed summary report"""
        report = [str(results)]

        report.append("\nPER-WINDOW RESULTS")
        report.append("=" * 100)
        report.append(f"{'Window':<8} {'Test Period':<25} {'Return':<12} {'Sharpe':<10} {'Win Rate':<12} {'Trades':<8}")
        report.append("-" * 100)

        for window in results.windows:
            test_period = f"{window.test_start.strftime('%Y-%m-%d')} to {window.test_end.strftime('%Y-%m-%d')}"
            report.append(
                f"{window.window_id:<8} "
                f"{test_period:<25} "
                f"{window.metrics.total_return:<12.2%} "
                f"{window.metrics.sharpe_ratio:<10.2f} "
                f"{window.metrics.win_rate:<12.2%} "
                f"{window.metrics.total_trades:<8}"
            )

        report.append("=" * 100)

        # Recommendations
        report.append("\nRECOMMENDATIONS")
        report.append("=" * 60)

        if results._is_valid():
            report.append("✅ Strategy PASSED all validation criteria")
            report.append("✅ Recommend proceeding to paper trading")
        else:
            report.append("❌ Strategy FAILED validation criteria")

            if results.profitable_pct < 0.70:
                report.append(f"  - Profitable windows: {results.profitable_pct:.1%} (target: ≥70%)")
            if results.avg_sharpe < 1.5:
                report.append(f"  - Average Sharpe: {results.avg_sharpe:.2f} (target: ≥1.5)")
            if results.returns_p_value >= 0.05:
                report.append(f"  - Returns p-value: {results.returns_p_value:.4f} (target: <0.05)")
            if results.consistency_score < 0.70:
                report.append(f"  - Consistency: {results.consistency_score:.1%} (target: ≥70%)")

            report.append("\n⚠ Recommend strategy optimization before proceeding")

        report.append("=" * 60)

        return "\n".join(report)
