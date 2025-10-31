"""
Backtesting Engine - Historical Trading Simulation

This module implements a comprehensive backtesting framework for validating
trading strategies on historical data with realistic transaction costs.

Features:
- Event-driven architecture for realistic simulation
- Transaction cost modeling (commission + slippage)
- Position tracking and P&L calculation
- Support for multiple strategies
- Detailed trade logging
- Performance metrics calculation

Author: Trading System Team
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enum"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enum"""
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    """Order status enum"""
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class Order:
    """Represents a trading order"""
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    timestamp: datetime
    price: Optional[float] = None  # For limit orders
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: int = 0
    commission: float = 0.0
    slippage: float = 0.0
    order_id: Optional[str] = None

    def __post_init__(self):
        if self.order_id is None:
            self.order_id = f"{self.symbol}_{self.timestamp.strftime('%Y%m%d%H%M%S')}"


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def update_price(self, current_price: float):
        """Update current price and unrealized P&L"""
        self.current_price = current_price
        self.unrealized_pnl = (current_price - self.entry_price) * self.quantity


@dataclass
class Trade:
    """Represents a completed trade (entry + exit)"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: OrderSide
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    holding_period: timedelta

    @property
    def net_pnl(self) -> float:
        """Net P&L after costs"""
        return self.pnl - self.commission - self.slippage


@dataclass
class BacktestConfig:
    """Configuration for backtest execution"""
    initial_capital: float = 100000.0
    commission_rate: float = 0.005  # $0.005 per share
    slippage_rate: float = 0.001  # 0.1% slippage
    position_size_pct: float = 0.10  # 10% of capital per position
    max_positions: int = 1  # Maximum concurrent positions
    stop_loss_pct: Optional[float] = 0.05  # 5% stop loss
    take_profit_pct: Optional[float] = None  # No take profit by default
    allow_shorting: bool = False
    risk_free_rate: float = 0.02  # 2% annual risk-free rate

    def __post_init__(self):
        """Validate configuration"""
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if self.commission_rate < 0:
            raise ValueError("Commission rate cannot be negative")
        if self.slippage_rate < 0:
            raise ValueError("Slippage rate cannot be negative")
        if not 0 < self.position_size_pct <= 1:
            raise ValueError("Position size percentage must be between 0 and 1")
        if self.max_positions < 1:
            raise ValueError("Max positions must be at least 1")


@dataclass
class BacktestState:
    """Current state of the backtest"""
    cash: float
    equity: float
    positions: Dict[str, Position] = field(default_factory=dict)
    orders: List[Order] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    drawdowns: List[float] = field(default_factory=list)

    @property
    def num_positions(self) -> int:
        """Number of open positions"""
        return len(self.positions)

    @property
    def position_value(self) -> float:
        """Total value of all positions"""
        return sum(pos.current_price * pos.quantity for pos in self.positions.values())

    @property
    def total_equity(self) -> float:
        """Total equity (cash + position value)"""
        return self.cash + self.position_value


class BacktestEngine:
    """
    Core backtesting engine for strategy validation

    This engine simulates trading on historical data with realistic
    transaction costs and position management.

    Example:
        >>> config = BacktestConfig(initial_capital=100000)
        >>> engine = BacktestEngine(config)
        >>> strategy = MeanReversionStrategy()
        >>> results = engine.run_backtest(data, strategy)
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtesting engine

        Args:
            config: Backtest configuration
        """
        self.config = config
        self.state = BacktestState(
            cash=config.initial_capital,
            equity=config.initial_capital
        )
        self.peak_equity = config.initial_capital
        logger.info(f"Initialized BacktestEngine with ${config.initial_capital:,.2f}")

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy: Any,
        symbol: str = "STOCK"
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data

        Args:
            data: Historical OHLCV data with indicators
            strategy: Trading strategy object with generate_signal method
            symbol: Symbol being traded

        Returns:
            Dictionary containing backtest results and metrics
        """
        logger.info(f"Starting backtest for {symbol} from {data.index[0]} to {data.index[-1]}")
        logger.info(f"Data points: {len(data)}, Initial capital: ${self.config.initial_capital:,.2f}")

        # Reset state
        self.state = BacktestState(
            cash=self.config.initial_capital,
            equity=self.config.initial_capital
        )
        self.peak_equity = self.config.initial_capital

        # Main event loop
        for timestamp, bar in data.iterrows():
            # Update current prices for all positions
            if symbol in self.state.positions:
                self.state.positions[symbol].update_price(bar['close'])

            # Check stop loss and take profit
            self._check_exit_conditions(symbol, bar, timestamp)

            # Generate trading signal
            signal = strategy.generate_signal(bar, self.state)

            # Execute signal
            if signal:
                self._execute_signal(signal, bar, timestamp, symbol)

            # Update equity curve
            current_equity = self.state.total_equity
            self.state.equity_curve.append((timestamp, current_equity))

            # Track drawdown
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            self.state.drawdowns.append(drawdown)

        # Close any remaining positions at end
        if symbol in self.state.positions:
            last_bar = data.iloc[-1]
            self._close_position(symbol, last_bar['close'], data.index[-1])

        # Calculate and return results
        results = self._calculate_results(data)
        logger.info(f"Backtest complete. Final equity: ${results['final_equity']:,.2f}")
        logger.info(f"Total return: {results['total_return']:.2%}")
        logger.info(f"Total trades: {results['total_trades']}")

        return results

    def _execute_signal(
        self,
        signal: Dict[str, Any],
        bar: pd.Series,
        timestamp: datetime,
        symbol: str
    ):
        """Execute a trading signal"""
        action = signal.get('action')

        if action == 'buy' and self.state.num_positions < self.config.max_positions:
            # Calculate position size
            position_value = self.state.cash * self.config.position_size_pct
            price = bar['close']
            quantity = int(position_value / price)

            if quantity > 0 and self._has_sufficient_capital(quantity, price):
                self._open_position(symbol, quantity, price, timestamp)

        elif action == 'sell' and symbol in self.state.positions:
            # Close existing position
            price = bar['close']
            self._close_position(symbol, price, timestamp)

    def _open_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        timestamp: datetime
    ):
        """Open a new position"""
        # Calculate costs
        commission = self._calculate_commission(quantity, price)
        slippage = self._calculate_slippage(quantity, price, OrderSide.BUY)
        total_cost = (price * quantity) + commission + slippage

        # Check if we have enough cash
        if total_cost > self.state.cash:
            logger.warning(f"Insufficient cash for position. Required: ${total_cost:,.2f}, Available: ${self.state.cash:,.2f}")
            return

        # Create position
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_time=timestamp,
            current_price=price
        )

        # Update state
        self.state.positions[symbol] = position
        self.state.cash -= total_cost

        logger.debug(f"Opened {symbol} position: {quantity} shares @ ${price:.2f} (cost: ${total_cost:,.2f})")

    def _close_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime
    ):
        """Close an existing position"""
        if symbol not in self.state.positions:
            return

        position = self.state.positions[symbol]
        quantity = position.quantity

        # Calculate costs
        commission = self._calculate_commission(quantity, price)
        slippage = self._calculate_slippage(quantity, price, OrderSide.SELL)
        proceeds = (price * quantity) - commission - slippage

        # Calculate P&L
        pnl = (price - position.entry_price) * quantity
        pnl_pct = (price - position.entry_price) / position.entry_price

        # Create trade record
        trade = Trade(
            symbol=symbol,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price=position.entry_price,
            exit_price=price,
            quantity=quantity,
            side=OrderSide.BUY,  # We're tracking long positions
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission,
            slippage=slippage,
            holding_period=timestamp - position.entry_time
        )

        # Update state
        self.state.cash += proceeds
        self.state.trades.append(trade)
        del self.state.positions[symbol]

        logger.debug(f"Closed {symbol} position: {quantity} shares @ ${price:.2f} (P&L: ${trade.net_pnl:,.2f})")

    def _check_exit_conditions(
        self,
        symbol: str,
        bar: pd.Series,
        timestamp: datetime
    ):
        """Check if stop loss or take profit conditions are met"""
        if symbol not in self.state.positions:
            return

        position = self.state.positions[symbol]
        current_price = bar['close']
        pnl_pct = (current_price - position.entry_price) / position.entry_price

        # Check stop loss
        if self.config.stop_loss_pct and pnl_pct <= -self.config.stop_loss_pct:
            logger.info(f"Stop loss triggered for {symbol} at {pnl_pct:.2%}")
            self._close_position(symbol, current_price, timestamp)
            return

        # Check take profit
        if self.config.take_profit_pct and pnl_pct >= self.config.take_profit_pct:
            logger.info(f"Take profit triggered for {symbol} at {pnl_pct:.2%}")
            self._close_position(symbol, current_price, timestamp)
            return

    def _has_sufficient_capital(self, quantity: int, price: float) -> bool:
        """Check if there's sufficient capital for a trade"""
        commission = self._calculate_commission(quantity, price)
        slippage = self._calculate_slippage(quantity, price, OrderSide.BUY)
        total_cost = (price * quantity) + commission + slippage
        return total_cost <= self.state.cash

    def _calculate_commission(self, quantity: int, price: float) -> float:
        """Calculate commission cost"""
        return quantity * self.config.commission_rate

    def _calculate_slippage(self, quantity: int, price: float, side: OrderSide) -> float:
        """Calculate slippage cost"""
        slippage_cost = price * quantity * self.config.slippage_rate
        return slippage_cost

    def _calculate_results(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive backtest results"""
        equity_curve = pd.DataFrame(
            self.state.equity_curve,
            columns=['timestamp', 'equity']
        ).set_index('timestamp')

        # Basic metrics
        final_equity = equity_curve['equity'].iloc[-1]
        total_return = (final_equity - self.config.initial_capital) / self.config.initial_capital

        # Trade statistics
        trades = self.state.trades
        winning_trades = [t for t in trades if t.net_pnl > 0]
        losing_trades = [t for t in trades if t.net_pnl <= 0]

        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        avg_win = np.mean([t.net_pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.net_pnl for t in losing_trades]) if losing_trades else 0

        total_wins = sum(t.net_pnl for t in winning_trades)
        total_losses = abs(sum(t.net_pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Drawdown metrics
        max_drawdown = max(self.state.drawdowns) if self.state.drawdowns else 0

        return {
            'final_equity': final_equity,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_curve,
            'trades': trades,
            'config': self.config
        }
