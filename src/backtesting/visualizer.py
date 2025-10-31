"""
Backtesting Visualizer

This module provides visualization and reporting capabilities for backtest results,
including equity curves, drawdown charts, trade distributions, and performance reports.

Features:
- Equity curve plots
- Drawdown visualization
- Monthly returns heatmap
- Trade distribution analysis
- Rolling metrics charts
- Comprehensive HTML reports

Author: Trading System Team
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class BacktestVisualizer:
    """
    Visualize backtest results

    Creates comprehensive visualizations and reports for backtest analysis.

    Example:
        >>> visualizer = BacktestVisualizer()
        >>> visualizer.plot_equity_curve(results)
        >>> visualizer.generate_report(results, metrics)
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize visualizer

        Args:
            output_dir: Directory for saving plots and reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized BacktestVisualizer (output: {output_dir})")

    def plot_equity_curve(
        self,
        equity_curve: pd.DataFrame,
        title: str = "Equity Curve",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot equity curve over time

        Args:
            equity_curve: DataFrame with equity values
            title: Chart title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot equity
        ax.plot(equity_curve.index, equity_curve['equity'],
                linewidth=2, label='Portfolio Value')

        # Format
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity ($)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved equity curve to {save_path}")

        return fig

    def plot_drawdown(
        self,
        equity_curve: pd.DataFrame,
        title: str = "Drawdown Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot drawdown over time

        Args:
            equity_curve: DataFrame with equity values
            title: Chart title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        # Calculate drawdown
        equity = equity_curve['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100  # Convert to percentage

        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot drawdown
        ax.fill_between(drawdown.index, drawdown, 0,
                        alpha=0.3, color='red', label='Drawdown')
        ax.plot(drawdown.index, drawdown,
                linewidth=1.5, color='darkred')

        # Format
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Mark max drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_val = drawdown.min()
        ax.annotate(f'Max DD: {max_dd_val:.1f}%',
                   xy=(max_dd_idx, max_dd_val),
                   xytext=(10, 10),
                   textcoords='offset points',
                   fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved drawdown chart to {save_path}")

        return fig

    def plot_monthly_returns(
        self,
        equity_curve: pd.DataFrame,
        title: str = "Monthly Returns Heatmap",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot monthly returns as heatmap

        Args:
            equity_curve: DataFrame with equity values
            title: Chart title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        # Calculate daily returns
        returns = equity_curve['equity'].pct_change()

        # Resample to monthly
        monthly_returns = (1 + returns).resample('M').prod() - 1

        # Create pivot table (year x month)
        monthly_returns_pct = monthly_returns * 100
        monthly_returns_pct.index = pd.to_datetime(monthly_returns_pct.index)

        pivot_table = pd.DataFrame({
            'Year': monthly_returns_pct.index.year,
            'Month': monthly_returns_pct.index.month,
            'Return': monthly_returns_pct.values
        })

        # Pivot
        heatmap_data = pivot_table.pivot(index='Year', columns='Month', values='Return')

        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Return (%)'},
            linewidths=0.5,
            ax=ax
        )

        # Format
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)

        # Month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_labels, rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved monthly returns heatmap to {save_path}")

        return fig

    def plot_trade_distribution(
        self,
        trades: List[Any],
        title: str = "Trade P&L Distribution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution of trade P&L

        Args:
            trades: List of Trade objects
            title: Chart title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if not trades:
            logger.warning("No trades to plot")
            return None

        # Extract P&L
        pnl_values = [t.net_pnl for t in trades]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Histogram
        ax1.hist(pnl_values, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax1.set_title('P&L Histogram', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Net P&L ($)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot([pnl_values], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax2.set_title('P&L Box Plot', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Net P&L ($)', fontsize=12)
        ax2.set_xticklabels(['All Trades'])
        ax2.grid(True, alpha=0.3, axis='y')

        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved trade distribution to {save_path}")

        return fig

    def plot_rolling_sharpe(
        self,
        equity_curve: pd.DataFrame,
        window: int = 252,
        title: str = "Rolling Sharpe Ratio (1-Year Window)",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot rolling Sharpe ratio

        Args:
            equity_curve: DataFrame with equity values
            window: Rolling window size (default: 252 trading days)
            title: Chart title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        # Calculate returns
        returns = equity_curve['equity'].pct_change()

        # Calculate rolling Sharpe
        rolling_sharpe = (
            returns.rolling(window).mean() /
            returns.rolling(window).std() * np.sqrt(252)
        )

        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot rolling Sharpe
        ax.plot(rolling_sharpe.index, rolling_sharpe,
                linewidth=2, label=f'{window}-day Rolling Sharpe')

        # Add reference lines
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, label='Sharpe = 1.0 (Good)')
        ax.axhline(y=2.0, color='blue', linestyle='--', linewidth=1, label='Sharpe = 2.0 (Excellent)')
        ax.axhline(y=0, color='red', linestyle='-', linewidth=1, label='Sharpe = 0')

        # Format
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sharpe Ratio', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved rolling Sharpe to {save_path}")

        return fig

    def plot_walk_forward_results(
        self,
        wf_results: Any,
        title: str = "Walk-Forward Analysis Results",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot walk-forward analysis results

        Args:
            wf_results: WalkForwardResults object
            title: Chart title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        # Extract metrics per window
        window_ids = [w.window_id for w in wf_results.windows]
        sharpe_ratios = [w.metrics.sharpe_ratio for w in wf_results.windows]
        returns = [w.metrics.total_return * 100 for w in wf_results.windows]  # Convert to %
        win_rates = [w.metrics.win_rate * 100 for w in wf_results.windows]  # Convert to %

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # Plot 1: Sharpe Ratio per window
        axes[0].bar(window_ids, sharpe_ratios, alpha=0.7, color='steelblue')
        axes[0].axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Target: 1.0')
        axes[0].axhline(y=wf_results.avg_sharpe, color='red', linestyle='-', linewidth=2,
                       label=f'Average: {wf_results.avg_sharpe:.2f}')
        axes[0].set_title('Sharpe Ratio by Window', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Window', fontsize=12)
        axes[0].set_ylabel('Sharpe Ratio', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # Plot 2: Returns per window
        colors = ['green' if r > 0 else 'red' for r in returns]
        axes[1].bar(window_ids, returns, alpha=0.7, color=colors)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1].axhline(y=wf_results.avg_return * 100, color='blue', linestyle='--', linewidth=2,
                       label=f'Average: {wf_results.avg_return:.2%}')
        axes[1].set_title('Returns by Window', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Window', fontsize=12)
        axes[1].set_ylabel('Return (%)', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        # Plot 3: Win Rate per window
        axes[2].bar(window_ids, win_rates, alpha=0.7, color='orange')
        axes[2].axhline(y=60, color='green', linestyle='--', linewidth=2, label='Target: 60%')
        axes[2].axhline(y=wf_results.avg_win_rate * 100, color='red', linestyle='-', linewidth=2,
                       label=f'Average: {wf_results.avg_win_rate:.2%}')
        axes[2].set_title('Win Rate by Window', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Window', fontsize=12)
        axes[2].set_ylabel('Win Rate (%)', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, axis='y')

        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved walk-forward results to {save_path}")

        return fig

    def generate_html_report(
        self,
        results: Dict[str, Any],
        metrics: Any,
        symbol: str,
        strategy_name: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive HTML report

        Args:
            results: Backtest results dictionary
            metrics: PerformanceMetrics object
            symbol: Symbol traded
            strategy_name: Name of strategy
            output_path: Optional path to save HTML

        Returns:
            HTML string
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report - {symbol}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric-value {{
            font-weight: bold;
            color: #4CAF50;
        }}
        .metric-value.negative {{
            color: #f44336;
        }}
        .status-passed {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .status-failed {{
            color: #f44336;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Backtest Report: {symbol}</h1>
        <p><strong>Strategy:</strong> {strategy_name}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Performance Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Return</td>
                <td class="metric-value {'negative' if metrics.total_return < 0 else ''}">{metrics.total_return:.2%}</td>
            </tr>
            <tr>
                <td>CAGR</td>
                <td class="metric-value">{metrics.cagr:.2%}</td>
            </tr>
            <tr>
                <td>Sharpe Ratio</td>
                <td class="metric-value">{metrics.sharpe_ratio:.2f}</td>
            </tr>
            <tr>
                <td>Sortino Ratio</td>
                <td class="metric-value">{metrics.sortino_ratio:.2f}</td>
            </tr>
            <tr>
                <td>Max Drawdown</td>
                <td class="metric-value negative">{metrics.max_drawdown:.2%}</td>
            </tr>
            <tr>
                <td>Win Rate</td>
                <td class="metric-value">{metrics.win_rate:.2%}</td>
            </tr>
            <tr>
                <td>Profit Factor</td>
                <td class="metric-value">{metrics.profit_factor:.2f}</td>
            </tr>
        </table>

        <h2>Trade Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Trades</td>
                <td>{metrics.total_trades}</td>
            </tr>
            <tr>
                <td>Winning Trades</td>
                <td>{metrics.winning_trades}</td>
            </tr>
            <tr>
                <td>Losing Trades</td>
                <td>{metrics.losing_trades}</td>
            </tr>
            <tr>
                <td>Average Win</td>
                <td class="metric-value">${metrics.avg_win:,.2f}</td>
            </tr>
            <tr>
                <td>Average Loss</td>
                <td class="metric-value negative">${metrics.avg_loss:,.2f}</td>
            </tr>
            <tr>
                <td>Expectancy</td>
                <td class="metric-value">${metrics.expectancy:,.2f}</td>
            </tr>
        </table>

        <h2>Statistical Validation</h2>
        <table>
            <tr>
                <th>Test</th>
                <th>Value</th>
                <th>Status</th>
            </tr>
            <tr>
                <td>p-value (returns ≠ 0)</td>
                <td>{metrics.p_value:.4f}</td>
                <td class="{'status-passed' if metrics.p_value < 0.05 else 'status-failed'}">
                    {'PASSED' if metrics.p_value < 0.05 else 'FAILED'}
                </td>
            </tr>
            <tr>
                <td>t-statistic</td>
                <td>{metrics.t_statistic:.2f}</td>
                <td>-</td>
            </tr>
            <tr>
                <td>95% Confidence Interval</td>
                <td>[{metrics.confidence_interval_95[0]:.4f}, {metrics.confidence_interval_95[1]:.4f}]</td>
                <td>-</td>
            </tr>
        </table>

        <h2>Validation Criteria</h2>
        <table>
            <tr>
                <th>Criterion</th>
                <th>Target</th>
                <th>Actual</th>
                <th>Status</th>
            </tr>
            <tr>
                <td>Sharpe Ratio</td>
                <td>&gt; 1.5</td>
                <td>{metrics.sharpe_ratio:.2f}</td>
                <td class="{'status-passed' if metrics.sharpe_ratio > 1.5 else 'status-failed'}">
                    {'PASSED' if metrics.sharpe_ratio > 1.5 else 'FAILED'}
                </td>
            </tr>
            <tr>
                <td>Win Rate</td>
                <td>&gt; 60%</td>
                <td>{metrics.win_rate:.2%}</td>
                <td class="{'status-passed' if metrics.win_rate > 0.6 else 'status-failed'}">
                    {'PASSED' if metrics.win_rate > 0.6 else 'FAILED'}
                </td>
            </tr>
            <tr>
                <td>Max Drawdown</td>
                <td>&lt; 20%</td>
                <td>{metrics.max_drawdown:.2%}</td>
                <td class="{'status-passed' if abs(metrics.max_drawdown) < 0.2 else 'status-failed'}">
                    {'PASSED' if abs(metrics.max_drawdown) < 0.2 else 'FAILED'}
                </td>
            </tr>
            <tr>
                <td>p-value</td>
                <td>&lt; 0.05</td>
                <td>{metrics.p_value:.4f}</td>
                <td class="{'status-passed' if metrics.p_value < 0.05 else 'status-failed'}">
                    {'PASSED' if metrics.p_value < 0.05 else 'FAILED'}
                </td>
            </tr>
        </table>
    </div>
</body>
</html>
        """

        if output_path:
            output_file = Path(output_path)
            output_file.write_text(html)
            logger.info(f"Saved HTML report to {output_path}")

        return html
