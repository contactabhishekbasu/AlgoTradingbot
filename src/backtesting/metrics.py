"""
Performance Metrics Calculator

This module calculates comprehensive performance metrics for backtesting results,
including return metrics, risk metrics, drawdown analysis, and statistical validation.

Metrics include:
- Return metrics: Total return, CAGR, annualized return
- Risk metrics: Sharpe ratio, Sortino ratio, Calmar ratio, volatility
- Drawdown metrics: Max drawdown, average drawdown, recovery time
- Trade statistics: Win rate, profit factor, expectancy
- Statistical validation: p-values, confidence intervals, t-statistics

Author: Trading System Team
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics"""
    # Return metrics
    total_return: float
    cagr: float
    annualized_return: float
    cumulative_return: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    volatility: float
    downside_deviation: float

    # Drawdown metrics
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float
    recovery_factor: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_holding_period: float

    # Statistical validation
    p_value: float
    t_statistic: float
    confidence_interval_95: Tuple[float, float]
    sharpe_significance: float

    # Risk-adjusted metrics
    beta: Optional[float] = None
    alpha: Optional[float] = None
    information_ratio: Optional[float] = None

    def __str__(self) -> str:
        """String representation of metrics"""
        return f"""
Performance Metrics Summary
{'=' * 50}

RETURNS
  Total Return:        {self.total_return:>10.2%}
  CAGR:                {self.cagr:>10.2%}
  Annualized Return:   {self.annualized_return:>10.2%}

RISK METRICS
  Sharpe Ratio:        {self.sharpe_ratio:>10.2f}
  Sortino Ratio:       {self.sortino_ratio:>10.2f}
  Calmar Ratio:        {self.calmar_ratio:>10.2f}
  Volatility (Annual): {self.volatility:>10.2%}

DRAWDOWN
  Max Drawdown:        {self.max_drawdown:>10.2%}
  Avg Drawdown:        {self.avg_drawdown:>10.2%}
  Recovery Factor:     {self.recovery_factor:>10.2f}

TRADES
  Total Trades:        {self.total_trades:>10}
  Win Rate:            {self.win_rate:>10.2%}
  Profit Factor:       {self.profit_factor:>10.2f}
  Expectancy:          ${self.expectancy:>9.2f}

STATISTICAL VALIDATION
  p-value:             {self.p_value:>10.4f}
  t-statistic:         {self.t_statistic:>10.2f}
  95% CI:              ({self.confidence_interval_95[0]:.4f}, {self.confidence_interval_95[1]:.4f})
{'=' * 50}
        """


class MetricsCalculator:
    """
    Calculate comprehensive performance metrics

    This calculator computes 30+ metrics for evaluating backtest performance
    and validating trading strategies.

    Example:
        >>> calculator = MetricsCalculator()
        >>> metrics = calculator.calculate_metrics(backtest_results)
        >>> print(metrics)
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize metrics calculator

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        logger.info(f"Initialized MetricsCalculator (risk-free rate: {risk_free_rate:.2%})")

    def calculate_metrics(
        self,
        results: Dict[str, Any],
        benchmark_returns: Optional[pd.Series] = None
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics

        Args:
            results: Backtest results dictionary from BacktestEngine
            benchmark_returns: Optional benchmark returns for beta/alpha calculation

        Returns:
            PerformanceMetrics object with all computed metrics
        """
        logger.info("Calculating performance metrics...")

        equity_curve = results['equity_curve']
        trades = results['trades']
        config = results['config']

        # Calculate returns
        returns = equity_curve['equity'].pct_change().dropna()
        initial_capital = config.initial_capital
        final_equity = results['final_equity']

        # Return metrics
        total_return = self._calculate_total_return(initial_capital, final_equity)
        cagr = self._calculate_cagr(equity_curve, initial_capital, final_equity)
        annualized_return = self._calculate_annualized_return(returns)
        cumulative_return = total_return

        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        volatility = self._calculate_volatility(returns)
        downside_deviation = self._calculate_downside_deviation(returns)

        # Drawdown metrics
        drawdown_metrics = self._calculate_drawdown_metrics(equity_curve)
        max_drawdown = drawdown_metrics['max_drawdown']
        max_drawdown_duration = drawdown_metrics['max_drawdown_duration']
        avg_drawdown = drawdown_metrics['avg_drawdown']

        # Calmar ratio
        calmar_ratio = self._calculate_calmar_ratio(cagr, max_drawdown)

        # Recovery factor
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        trade_stats = self._calculate_trade_statistics(trades)

        # Statistical validation
        stat_validation = self._calculate_statistical_validation(returns)

        # Benchmark-adjusted metrics
        beta = None
        alpha = None
        information_ratio = None
        if benchmark_returns is not None:
            beta = self._calculate_beta(returns, benchmark_returns)
            alpha = self._calculate_alpha(returns, benchmark_returns, beta)
            information_ratio = self._calculate_information_ratio(returns, benchmark_returns)

        metrics = PerformanceMetrics(
            # Returns
            total_return=total_return,
            cagr=cagr,
            annualized_return=annualized_return,
            cumulative_return=cumulative_return,
            # Risk
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            volatility=volatility,
            downside_deviation=downside_deviation,
            # Drawdown
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            avg_drawdown=avg_drawdown,
            recovery_factor=recovery_factor,
            # Trades
            **trade_stats,
            # Statistical
            **stat_validation,
            # Benchmark-adjusted
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio
        )

        logger.info(f"Metrics calculated: Sharpe={sharpe_ratio:.2f}, Win Rate={trade_stats['win_rate']:.2%}")
        return metrics

    def _calculate_total_return(self, initial_capital: float, final_equity: float) -> float:
        """Calculate total return"""
        return (final_equity - initial_capital) / initial_capital

    def _calculate_cagr(
        self,
        equity_curve: pd.DataFrame,
        initial_capital: float,
        final_equity: float
    ) -> float:
        """Calculate Compound Annual Growth Rate"""
        num_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        if num_years <= 0:
            return 0.0
        cagr = ((final_equity / initial_capital) ** (1 / num_years)) - 1
        return cagr

    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        if len(returns) == 0:
            return 0.0
        mean_return = returns.mean()
        annualized = mean_return * 252  # 252 trading days
        return annualized

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sharpe ratio

        Sharpe = (Return - Risk-free rate) / Volatility
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
        if excess_returns.std() == 0:
            return 0.0

        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        return sharpe

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sortino ratio

        Sortino = (Return - Risk-free rate) / Downside deviation
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / 252)
        downside_dev = self._calculate_downside_deviation(returns)

        if downside_dev == 0:
            return 0.0

        sortino = excess_returns.mean() / downside_dev * np.sqrt(252)
        return sortino

    def _calculate_calmar_ratio(self, cagr: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio (CAGR / Max Drawdown)"""
        if max_drawdown == 0:
            return 0.0
        return cagr / abs(max_drawdown)

    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        if len(returns) == 0:
            return 0.0
        return returns.std() * np.sqrt(252)

    def _calculate_downside_deviation(self, returns: pd.Series) -> float:
        """Calculate downside deviation (volatility of negative returns)"""
        if len(returns) == 0:
            return 0.0
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        return downside_returns.std()

    def _calculate_drawdown_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive drawdown metrics"""
        equity = equity_curve['equity']

        # Calculate running maximum
        running_max = equity.expanding().max()

        # Calculate drawdown
        drawdown = (equity - running_max) / running_max

        # Max drawdown
        max_drawdown = drawdown.min()

        # Drawdown duration
        is_in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0

        for in_dd in is_in_drawdown:
            if in_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0

        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0

        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_drawdown': avg_drawdown
        }

    def _calculate_trade_statistics(self, trades: List) -> Dict[str, Any]:
        """Calculate trade statistics"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'avg_holding_period': 0.0
            }

        # Categorize trades
        winning_trades = [t for t in trades if t.net_pnl > 0]
        losing_trades = [t for t in trades if t.net_pnl <= 0]

        # Win rate
        win_rate = len(winning_trades) / len(trades)

        # Profit factor
        total_wins = sum(t.net_pnl for t in winning_trades)
        total_losses = abs(sum(t.net_pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Expectancy
        expectancy = np.mean([t.net_pnl for t in trades])

        # Average win/loss
        avg_win = np.mean([t.net_pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.net_pnl for t in losing_trades]) if losing_trades else 0

        # Largest win/loss
        largest_win = max([t.net_pnl for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.net_pnl for t in losing_trades]) if losing_trades else 0

        # Average holding period
        avg_holding_period = np.mean([t.holding_period.days for t in trades])

        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_holding_period': avg_holding_period
        }

    def _calculate_statistical_validation(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate statistical validation metrics"""
        if len(returns) < 2:
            return {
                'p_value': 1.0,
                't_statistic': 0.0,
                'confidence_interval_95': (0.0, 0.0),
                'sharpe_significance': 1.0
            }

        # T-test: null hypothesis is that mean return = 0
        t_stat, p_value = stats.ttest_1samp(returns, 0)

        # Confidence interval (95%)
        confidence_level = 0.95
        degrees_freedom = len(returns) - 1
        confidence_interval = stats.t.interval(
            confidence_level,
            degrees_freedom,
            loc=returns.mean(),
            scale=stats.sem(returns)
        )

        # Sharpe ratio significance
        # Test if Sharpe ratio is significantly different from 0
        sharpe = self._calculate_sharpe_ratio(returns)
        sharpe_se = np.sqrt((1 + (sharpe ** 2) / 2) / len(returns))
        sharpe_t = sharpe / sharpe_se if sharpe_se > 0 else 0
        sharpe_p_value = 2 * (1 - stats.norm.cdf(abs(sharpe_t)))

        return {
            'p_value': p_value,
            't_statistic': t_stat,
            'confidence_interval_95': confidence_interval,
            'sharpe_significance': sharpe_p_value
        }

    def _calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta (correlation with benchmark)"""
        # Align returns
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')

        if len(aligned_returns) < 2:
            return 1.0

        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)

        if benchmark_variance == 0:
            return 1.0

        beta = covariance / benchmark_variance
        return beta

    def _calculate_alpha(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        beta: float
    ) -> float:
        """Calculate alpha (excess return over benchmark)"""
        # Align returns
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')

        if len(aligned_returns) < 2:
            return 0.0

        # Annualized alpha
        strategy_return = aligned_returns.mean() * 252
        benchmark_return = aligned_benchmark.mean() * 252
        risk_free = self.risk_free_rate

        alpha = strategy_return - (risk_free + beta * (benchmark_return - risk_free))
        return alpha

    def _calculate_information_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate information ratio"""
        # Align returns
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')

        if len(aligned_returns) < 2:
            return 0.0

        # Excess returns
        excess_returns = aligned_returns - aligned_benchmark
        tracking_error = excess_returns.std() * np.sqrt(252)

        if tracking_error == 0:
            return 0.0

        information_ratio = (excess_returns.mean() * 252) / tracking_error
        return information_ratio

    def calculate_rolling_metrics(
        self,
        equity_curve: pd.DataFrame,
        window: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics

        Args:
            equity_curve: Equity curve DataFrame
            window: Rolling window size (default: 252 days = 1 year)

        Returns:
            DataFrame with rolling metrics
        """
        returns = equity_curve['equity'].pct_change().dropna()

        rolling_metrics = pd.DataFrame(index=returns.index)

        # Rolling Sharpe ratio
        rolling_metrics['sharpe'] = (
            (returns.rolling(window).mean() - self.risk_free_rate / 252) /
            returns.rolling(window).std() * np.sqrt(252)
        )

        # Rolling volatility
        rolling_metrics['volatility'] = returns.rolling(window).std() * np.sqrt(252)

        # Rolling max drawdown
        equity = equity_curve['equity']
        rolling_max = equity.rolling(window).max()
        rolling_metrics['drawdown'] = (equity - rolling_max) / rolling_max

        return rolling_metrics
