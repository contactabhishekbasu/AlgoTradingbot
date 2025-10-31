"""
Mean Reversion Trading Strategy

This module implements a mean reversion trading strategy using technical indicators
like RSI and Williams %R to identify oversold/overbought conditions.

Strategy Logic:
- Entry: Buy when RSI < 30 or Williams %R < -80 (oversold)
- Exit: Sell when RSI > 70 or Williams %R > -20 (overbought)
- Can also integrate ML predictions for confirmation

Author: Trading System Team
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MeanReversionConfig:
    """Configuration for mean reversion strategy"""
    rsi_oversold: float = 30.0  # RSI threshold for oversold
    rsi_overbought: float = 70.0  # RSI threshold for overbought
    williams_oversold: float = -80.0  # Williams %R oversold threshold
    williams_overbought: float = -20.0  # Williams %R overbought threshold
    use_ml_confirmation: bool = True  # Use ML predictions to confirm signals
    min_ml_confidence: float = 0.6  # Minimum ML confidence for entry
    hold_period_min: int = 1  # Minimum holding period (days)
    hold_period_max: int = 10  # Maximum holding period (days)


class MeanReversionStrategy:
    """
    Mean Reversion Trading Strategy

    This strategy identifies oversold conditions as buying opportunities
    and overbought conditions as selling opportunities.

    Example:
        >>> config = MeanReversionConfig()
        >>> strategy = MeanReversionStrategy(config)
        >>> signal = strategy.generate_signal(bar, state)
    """

    def __init__(self, config: MeanReversionConfig = None, ml_predictor: Any = None):
        """
        Initialize mean reversion strategy

        Args:
            config: Strategy configuration
            ml_predictor: Optional ML predictor for signal confirmation
        """
        self.config = config or MeanReversionConfig()
        self.ml_predictor = ml_predictor
        self.entry_time = None
        self.signals_generated = 0
        self.signals_with_ml = 0

        logger.info(f"Initialized MeanReversionStrategy")
        logger.info(f"RSI thresholds: {self.config.rsi_oversold}/{self.config.rsi_overbought}")
        logger.info(f"Williams %R thresholds: {self.config.williams_oversold}/{self.config.williams_overbought}")

    def generate_signal(self, bar: pd.Series, state: Any) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal based on current market data

        Args:
            bar: Current bar with OHLCV and indicators
            state: Current backtest state

        Returns:
            Signal dictionary with action and metadata, or None
        """
        # Ensure we have required indicators
        required_indicators = ['rsi', 'williams_r']
        if not all(ind in bar.index for ind in required_indicators):
            return None

        # Skip if indicators are NaN
        if pd.isna(bar['rsi']) or pd.isna(bar['williams_r']):
            return None

        # Get current indicators
        rsi = bar['rsi']
        williams_r = bar['williams_r']

        # Check if we have a position
        has_position = state.num_positions > 0

        # Generate exit signal if we have a position
        if has_position:
            return self._generate_exit_signal(bar, rsi, williams_r, state)

        # Generate entry signal if we don't have a position
        return self._generate_entry_signal(bar, rsi, williams_r, state)

    def _generate_entry_signal(
        self,
        bar: pd.Series,
        rsi: float,
        williams_r: float,
        state: Any
    ) -> Optional[Dict[str, Any]]:
        """Generate entry (buy) signal"""
        # Check for oversold conditions
        rsi_oversold = rsi < self.config.rsi_oversold
        williams_oversold = williams_r < self.config.williams_oversold

        # Entry signal if either indicator shows oversold
        if not (rsi_oversold or williams_oversold):
            return None

        # ML confirmation if enabled
        ml_prediction = None
        ml_confidence = None

        if self.config.use_ml_confirmation and self.ml_predictor is not None:
            ml_prediction, ml_confidence = self._get_ml_prediction(bar)

            # Require ML to predict UP with sufficient confidence
            if ml_prediction != 'up' or ml_confidence < self.config.min_ml_confidence:
                logger.debug(f"ML rejected entry: prediction={ml_prediction}, confidence={ml_confidence:.2f}")
                return None

            self.signals_with_ml += 1

        self.signals_generated += 1
        self.entry_time = bar.name  # Store entry time

        signal = {
            'action': 'buy',
            'reason': self._get_entry_reason(rsi_oversold, williams_oversold),
            'rsi': rsi,
            'williams_r': williams_r,
            'ml_prediction': ml_prediction,
            'ml_confidence': ml_confidence,
            'timestamp': bar.name
        }

        logger.debug(f"Entry signal: {signal['reason']} (RSI={rsi:.1f}, WR={williams_r:.1f})")
        return signal

    def _generate_exit_signal(
        self,
        bar: pd.Series,
        rsi: float,
        williams_r: float,
        state: Any
    ) -> Optional[Dict[str, Any]]:
        """Generate exit (sell) signal"""
        # Check holding period
        if self.entry_time is not None:
            holding_period = (bar.name - self.entry_time).days
            if holding_period < self.config.hold_period_min:
                return None  # Don't exit too early
            if holding_period >= self.config.hold_period_max:
                # Force exit if max holding period reached
                return {
                    'action': 'sell',
                    'reason': 'max_holding_period',
                    'rsi': rsi,
                    'williams_r': williams_r,
                    'holding_period': holding_period,
                    'timestamp': bar.name
                }

        # Check for overbought conditions
        rsi_overbought = rsi > self.config.rsi_overbought
        williams_overbought = williams_r > self.config.williams_overbought

        # Exit signal if either indicator shows overbought
        if not (rsi_overbought or williams_overbought):
            return None

        # ML confirmation if enabled
        if self.config.use_ml_confirmation and self.ml_predictor is not None:
            ml_prediction, ml_confidence = self._get_ml_prediction(bar)

            # Exit if ML predicts DOWN with high confidence
            if ml_prediction == 'down' and ml_confidence >= self.config.min_ml_confidence:
                signal = {
                    'action': 'sell',
                    'reason': 'ml_prediction_down',
                    'rsi': rsi,
                    'williams_r': williams_r,
                    'ml_prediction': ml_prediction,
                    'ml_confidence': ml_confidence,
                    'timestamp': bar.name
                }
                logger.debug(f"Exit signal: ML predicts down (confidence={ml_confidence:.2f})")
                self.entry_time = None
                return signal

        # Exit on overbought
        signal = {
            'action': 'sell',
            'reason': self._get_exit_reason(rsi_overbought, williams_overbought),
            'rsi': rsi,
            'williams_r': williams_r,
            'timestamp': bar.name
        }

        logger.debug(f"Exit signal: {signal['reason']} (RSI={rsi:.1f}, WR={williams_r:.1f})")
        self.entry_time = None
        return signal

    def _get_ml_prediction(self, bar: pd.Series) -> tuple[str, float]:
        """
        Get ML prediction and confidence

        Returns:
            Tuple of (prediction, confidence)
        """
        # If ML features are in the bar, use them directly
        if 'ml_prediction' in bar.index and 'ml_confidence' in bar.index:
            prediction_code = int(bar['ml_prediction'])
            confidence = float(bar['ml_confidence'])

            # Map prediction codes to labels
            prediction_map = {0: 'down', 1: 'neutral', 2: 'up'}
            prediction = prediction_map.get(prediction_code, 'neutral')

            return prediction, confidence

        # Otherwise, we'd call the ML predictor here
        # For now, return neutral with low confidence
        return 'neutral', 0.5

    def _get_entry_reason(self, rsi_oversold: bool, williams_oversold: bool) -> str:
        """Get human-readable entry reason"""
        if rsi_oversold and williams_oversold:
            return 'rsi_and_williams_oversold'
        elif rsi_oversold:
            return 'rsi_oversold'
        else:
            return 'williams_oversold'

    def _get_exit_reason(self, rsi_overbought: bool, williams_overbought: bool) -> str:
        """Get human-readable exit reason"""
        if rsi_overbought and williams_overbought:
            return 'rsi_and_williams_overbought'
        elif rsi_overbought:
            return 'rsi_overbought'
        else:
            return 'williams_overbought'

    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics"""
        ml_usage_pct = (self.signals_with_ml / self.signals_generated * 100
                        if self.signals_generated > 0 else 0)

        return {
            'signals_generated': self.signals_generated,
            'signals_with_ml_confirmation': self.signals_with_ml,
            'ml_usage_percentage': ml_usage_pct,
            'config': self.config
        }


class MLEnhancedMeanReversion(MeanReversionStrategy):
    """
    Mean reversion strategy with integrated ML predictions

    This version requires ML predictions to be present in the data
    and uses them more aggressively for both entry and exit decisions.
    """

    def __init__(self, config: MeanReversionConfig = None):
        """Initialize ML-enhanced strategy"""
        if config is None:
            config = MeanReversionConfig()
        config.use_ml_confirmation = True
        config.min_ml_confidence = 0.65  # Higher confidence threshold

        super().__init__(config)
        logger.info("Initialized MLEnhancedMeanReversion strategy")

    def _generate_entry_signal(
        self,
        bar: pd.Series,
        rsi: float,
        williams_r: float,
        state: Any
    ) -> Optional[Dict[str, Any]]:
        """Generate entry signal with strict ML requirements"""
        # Must have ML prediction
        if 'ml_prediction' not in bar.index or 'ml_confidence' not in bar.index:
            return None

        ml_prediction, ml_confidence = self._get_ml_prediction(bar)

        # Require ML to predict UP
        if ml_prediction != 'up' or ml_confidence < self.config.min_ml_confidence:
            return None

        # Also require at least one oversold indicator
        rsi_oversold = rsi < self.config.rsi_oversold
        williams_oversold = williams_r < self.config.williams_oversold

        if not (rsi_oversold or williams_oversold):
            return None

        self.signals_generated += 1
        self.signals_with_ml += 1
        self.entry_time = bar.name

        return {
            'action': 'buy',
            'reason': 'ml_confirmed_oversold',
            'rsi': rsi,
            'williams_r': williams_r,
            'ml_prediction': ml_prediction,
            'ml_confidence': ml_confidence,
            'timestamp': bar.name
        }
