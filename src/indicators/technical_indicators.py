"""Technical indicators implementation with NumPy vectorization."""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logger import logger


class TechnicalIndicators:
    """
    Vectorized technical indicators for trading strategies.

    Based on research:
    - Mean reversion indicators (QuantifiedStrategies, 2024)
    - 1068 technical patterns (Leci37, 2023)

    Optimized for performance with NumPy vectorization.
    """

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss

        Args:
            data: Price series (typically Close prices)
            period: RSI period (default: 14)

        Returns:
            Series with RSI values (0-100)

        Reference:
            - Win rate: 70-80% in mean reversion
            - Entry: RSI < 30 (oversold)
            - Exit: RSI > 70 (overbought)
        """
        logger.debug("calculating_rsi", period=period, data_points=len(data))

        # Calculate price changes
        delta = data.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)

        # Calculate average gains and losses using exponential moving average
        avg_gains = gains.ewm(span=period, adjust=False).mean()
        avg_losses = losses.ewm(span=period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100.0 - (100.0 / (1.0 + rs))

        logger.debug(
            "rsi_calculated",
            min_value=rsi.min(),
            max_value=rsi.max(),
            current_value=rsi.iloc[-1] if len(rsi) > 0 else None,
        )

        return rsi

    @staticmethod
    def williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Calculate Williams %R indicator.

        Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Lookback period (default: 14)

        Returns:
            Series with Williams %R values (-100 to 0)

        Reference:
            - Entry signal: Williams %R < -80 (oversold)
            - Exit signal: Williams %R > -20 (overbought)
            - Holding period: 1-10 days
        """
        logger.debug("calculating_williams_r", period=period, data_points=len(close))

        # Calculate highest high and lowest low over period
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        # Calculate Williams %R
        williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100

        logger.debug(
            "williams_r_calculated",
            min_value=williams_r.min(),
            max_value=williams_r.max(),
            current_value=williams_r.iloc[-1] if len(williams_r) > 0 else None,
        )

        return williams_r

    @staticmethod
    def bollinger_bands(
        data: pd.Series,
        period: int = 20,
        num_std: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            data: Price series (typically Close prices)
            period: Moving average period (default: 20)
            num_std: Number of standard deviations (default: 2.0)

        Returns:
            Tuple of (upper_band, middle_band, lower_band)

        Reference:
            - Middle band: 20-period SMA
            - Upper/Lower: ±2 standard deviations
            - Band squeeze signals volatility contraction
        """
        logger.debug(
            "calculating_bollinger_bands",
            period=period,
            std_dev=num_std,
            data_points=len(data),
        )

        # Calculate middle band (SMA)
        middle_band = data.rolling(window=period).mean()

        # Calculate standard deviation
        std = data.rolling(window=period).std()

        # Calculate upper and lower bands
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)

        logger.debug(
            "bollinger_bands_calculated",
            current_upper=upper_band.iloc[-1] if len(upper_band) > 0 else None,
            current_middle=middle_band.iloc[-1] if len(middle_band) > 0 else None,
            current_lower=lower_band.iloc[-1] if len(lower_band) > 0 else None,
        )

        return upper_band, middle_band, lower_band

    @staticmethod
    def macd(
        data: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            data: Price series (typically Close prices)
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)

        Returns:
            Tuple of (macd_line, signal_line, histogram)

        Reference:
            - MACD line: 12-period EMA - 26-period EMA
            - Signal line: 9-period EMA of MACD line
            - Histogram: MACD line - Signal line
            - Crossover signals trend changes
        """
        logger.debug(
            "calculating_macd",
            fast=fast_period,
            slow=slow_period,
            signal=signal_period,
            data_points=len(data),
        )

        # Calculate fast and slow EMAs
        fast_ema = data.ewm(span=fast_period, adjust=False).mean()
        slow_ema = data.ewm(span=slow_period, adjust=False).mean()

        # Calculate MACD line
        macd_line = fast_ema - slow_ema

        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Calculate histogram
        histogram = macd_line - signal_line

        logger.debug(
            "macd_calculated",
            current_macd=macd_line.iloc[-1] if len(macd_line) > 0 else None,
            current_signal=signal_line.iloc[-1] if len(signal_line) > 0 else None,
            current_histogram=histogram.iloc[-1] if len(histogram) > 0 else None,
        )

        return macd_line, signal_line, histogram

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average.

        Args:
            data: Price series
            period: SMA period

        Returns:
            Series with SMA values
        """
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.

        Args:
            data: Price series
            period: EMA period

        Returns:
            Series with EMA values
        """
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period (default: 14)

        Returns:
            Series with ATR values
        """
        # Calculate True Range
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calculate ATR (EMA of True Range)
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Lookback period (default: 14)
            smooth_k: %K smoothing period (default: 3)
            smooth_d: %D smoothing period (default: 3)

        Returns:
            Tuple of (%K, %D)
        """
        # Calculate %K
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()

        k_percent = ((close - lowest_low) / (highest_high - lowest_low)) * 100

        # Smooth %K
        k_percent = k_percent.rolling(window=smooth_k).mean()

        # Calculate %D (SMA of %K)
        d_percent = k_percent.rolling(window=smooth_d).mean()

        return k_percent, d_percent

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).

        Args:
            close: Close prices
            volume: Volume data

        Returns:
            Series with OBV values
        """
        # Calculate price changes
        price_change = close.diff()

        # Assign volume based on price direction
        obv_values = np.where(
            price_change > 0,
            volume,
            np.where(price_change < 0, -volume, 0),
        )

        # Cumulative sum
        obv = pd.Series(obv_values, index=close.index).cumsum()

        return obv

    def calculate_all(
        self,
        data: pd.DataFrame,
        indicators: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Calculate all requested technical indicators.

        Args:
            data: DataFrame with OHLCV data
            indicators: List of indicator names to calculate (default: all)

        Returns:
            DataFrame with original data and calculated indicators

        Available indicators:
            - rsi: Relative Strength Index
            - williams_r: Williams %R
            - bbands: Bollinger Bands
            - macd: MACD
            - sma_20, sma_50, sma_200: Simple Moving Averages
            - ema_12, ema_26: Exponential Moving Averages
            - atr: Average True Range
            - stochastic: Stochastic Oscillator
            - obv: On-Balance Volume
        """
        result = data.copy()

        # Default to core indicators if not specified
        if indicators is None:
            indicators = ["rsi", "williams_r", "bbands", "macd"]

        logger.info(
            "calculating_indicators",
            indicators=indicators,
            data_points=len(data),
        )

        # RSI
        if "rsi" in indicators:
            result["RSI"] = self.rsi(data["Close"])

        # Williams %R
        if "williams_r" in indicators:
            result["Williams_R"] = self.williams_r(
                data["High"], data["Low"], data["Close"]
            )

        # Bollinger Bands
        if "bbands" in indicators:
            upper, middle, lower = self.bollinger_bands(data["Close"])
            result["BB_Upper"] = upper
            result["BB_Middle"] = middle
            result["BB_Lower"] = lower
            result["BB_Width"] = (upper - lower) / middle  # Normalized width

        # MACD
        if "macd" in indicators:
            macd_line, signal_line, histogram = self.macd(data["Close"])
            result["MACD"] = macd_line
            result["MACD_Signal"] = signal_line
            result["MACD_Histogram"] = histogram

        # Simple Moving Averages
        for period in [20, 50, 200]:
            if f"sma_{period}" in indicators:
                result[f"SMA_{period}"] = self.sma(data["Close"], period)

        # Exponential Moving Averages
        for period in [12, 26]:
            if f"ema_{period}" in indicators:
                result[f"EMA_{period}"] = self.ema(data["Close"], period)

        # ATR
        if "atr" in indicators:
            result["ATR"] = self.atr(data["High"], data["Low"], data["Close"])

        # Stochastic
        if "stochastic" in indicators:
            k, d = self.stochastic(data["High"], data["Low"], data["Close"])
            result["Stochastic_K"] = k
            result["Stochastic_D"] = d

        # OBV
        if "obv" in indicators:
            result["OBV"] = self.obv(data["Close"], data["Volume"])

        logger.info(
            "indicators_calculated",
            total_columns=len(result.columns),
            indicators_added=len(result.columns) - len(data.columns),
        )

        return result
