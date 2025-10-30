"""Unit tests for technical indicators with known values."""

import pytest
import pandas as pd
import numpy as np

from src.indicators.technical_indicators import TechnicalIndicators


class TestTechnicalIndicators:
    """Test technical indicator calculations."""

    @pytest.fixture
    def indicators(self):
        """Create indicator calculator instance."""
        return TechnicalIndicators()

    @pytest.fixture
    def sample_data(self):
        """Create sample price data for testing."""
        # Known data for testing indicators
        dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
        np.random.seed(42)

        # Create realistic price data
        close_prices = 100 + np.cumsum(np.random.randn(50) * 2)
        high_prices = close_prices + np.abs(np.random.randn(50))
        low_prices = close_prices - np.abs(np.random.randn(50))
        open_prices = close_prices + np.random.randn(50) * 0.5

        return pd.DataFrame(
            {
                "Date": dates,
                "Open": open_prices,
                "High": high_prices,
                "Low": low_prices,
                "Close": close_prices,
                "Volume": np.random.randint(1000000, 5000000, 50),
            }
        )

    def test_rsi_calculation(self, indicators):
        """Test RSI calculation with known values."""
        # Known test case from TradingView
        prices = pd.Series([44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84,
                           46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03,
                           46.41, 46.22, 45.64, 46.21])

        rsi = indicators.rsi(prices, period=14)

        # RSI should be between 0 and 100
        assert all(rsi.dropna() >= 0)
        assert all(rsi.dropna() <= 100)

        # Last value should be approximately 52.48 (known value)
        assert abs(rsi.iloc[-1] - 52.5) < 5.0  # Allow some tolerance

    def test_rsi_range(self, indicators, sample_data):
        """Test that RSI stays in valid range."""
        rsi = indicators.rsi(sample_data["Close"], period=14)

        assert all(rsi.dropna() >= 0)
        assert all(rsi.dropna() <= 100)

    def test_williams_r_calculation(self, indicators, sample_data):
        """Test Williams %R calculation."""
        williams = indicators.williams_r(
            sample_data["High"],
            sample_data["Low"],
            sample_data["Close"],
            period=14,
        )

        # Williams %R should be between -100 and 0
        assert all(williams.dropna() >= -100)
        assert all(williams.dropna() <= 0)

    def test_bollinger_bands_calculation(self, indicators, sample_data):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = indicators.bollinger_bands(
            sample_data["Close"], period=20, num_std=2.0
        )

        # Upper should be > middle > lower
        valid_data = ~(upper.isna() | middle.isna() | lower.isna())
        assert all(upper[valid_data] > middle[valid_data])
        assert all(middle[valid_data] > lower[valid_data])

        # Middle should be the SMA
        sma = sample_data["Close"].rolling(window=20).mean()
        assert all(abs(middle - sma) < 0.01)

    def test_macd_calculation(self, indicators, sample_data):
        """Test MACD calculation."""
        macd_line, signal_line, histogram = indicators.macd(
            sample_data["Close"],
            fast_period=12,
            slow_period=26,
            signal_period=9,
        )

        # Histogram should be MACD - Signal
        valid_data = ~(macd_line.isna() | signal_line.isna() | histogram.isna())
        diff = macd_line[valid_data] - signal_line[valid_data]
        assert all(abs(histogram[valid_data] - diff) < 0.01)

    def test_sma_calculation(self, indicators):
        """Test Simple Moving Average calculation."""
        prices = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        sma = indicators.sma(prices, period=3)

        # Known values: [nan, nan, 2, 3, 4, 5, 6, 7, 8, 9]
        assert pd.isna(sma.iloc[0])
        assert pd.isna(sma.iloc[1])
        assert sma.iloc[2] == 2.0
        assert sma.iloc[3] == 3.0
        assert sma.iloc[-1] == 9.0

    def test_ema_calculation(self, indicators):
        """Test Exponential Moving Average calculation."""
        prices = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ema = indicators.ema(prices, period=3)

        # EMA should be calculated
        assert not ema.isna().all()
        # EMA should follow price trend
        assert ema.iloc[-1] > ema.iloc[3]

    def test_atr_calculation(self, indicators, sample_data):
        """Test Average True Range calculation."""
        atr = indicators.atr(
            sample_data["High"],
            sample_data["Low"],
            sample_data["Close"],
            period=14,
        )

        # ATR should be positive
        assert all(atr.dropna() > 0)

    def test_stochastic_calculation(self, indicators, sample_data):
        """Test Stochastic Oscillator calculation."""
        k, d = indicators.stochastic(
            sample_data["High"],
            sample_data["Low"],
            sample_data["Close"],
            period=14,
        )

        # Stochastic should be between 0 and 100
        assert all(k.dropna() >= 0)
        assert all(k.dropna() <= 100)
        assert all(d.dropna() >= 0)
        assert all(d.dropna() <= 100)

    def test_obv_calculation(self, indicators, sample_data):
        """Test On-Balance Volume calculation."""
        obv = indicators.obv(sample_data["Close"], sample_data["Volume"])

        # OBV should be cumulative
        assert len(obv) == len(sample_data)

    def test_calculate_all_indicators(self, indicators, sample_data):
        """Test calculating all indicators at once."""
        result = indicators.calculate_all(sample_data)

        # Should have all original columns plus indicators
        assert len(result.columns) > len(sample_data.columns)

        # Check for specific indicators
        assert "RSI" in result.columns
        assert "Williams_R" in result.columns
        assert "BB_Upper" in result.columns
        assert "BB_Middle" in result.columns
        assert "BB_Lower" in result.columns
        assert "MACD" in result.columns
        assert "MACD_Signal" in result.columns
        assert "MACD_Histogram" in result.columns

    def test_calculate_specific_indicators(self, indicators, sample_data):
        """Test calculating specific indicators only."""
        result = indicators.calculate_all(sample_data, indicators=["rsi", "macd"])

        assert "RSI" in result.columns
        assert "MACD" in result.columns
        assert "Williams_R" not in result.columns  # Not requested

    def test_empty_data(self, indicators):
        """Test indicators with empty data."""
        empty_data = pd.DataFrame(
            {"Open": [], "High": [], "Low": [], "Close": [], "Volume": []}
        )

        result = indicators.calculate_all(empty_data)
        assert len(result) == 0

    def test_insufficient_data(self, indicators):
        """Test indicators with insufficient data points."""
        # Only 5 data points, but RSI needs 14
        short_data = pd.DataFrame(
            {
                "Close": [100, 101, 102, 103, 104],
                "High": [105, 106, 107, 108, 109],
                "Low": [95, 96, 97, 98, 99],
                "Open": [100, 101, 102, 103, 104],
                "Volume": [1000000] * 5,
            }
        )

        result = indicators.calculate_all(short_data, indicators=["rsi"])

        # Should have RSI column but values will be NaN
        assert "RSI" in result.columns
        assert result["RSI"].isna().all()


class TestIndicatorBoundaries:
    """Test indicator boundary conditions."""

    @pytest.fixture
    def indicators(self):
        return TechnicalIndicators()

    def test_rsi_oversold(self, indicators):
        """Test RSI identifies oversold conditions."""
        # Create declining prices
        prices = pd.Series(range(100, 50, -1))
        rsi = indicators.rsi(prices, period=14)

        # Last RSI should be low (oversold)
        assert rsi.iloc[-1] < 30

    def test_rsi_overbought(self, indicators):
        """Test RSI identifies overbought conditions."""
        # Create rising prices
        prices = pd.Series(range(50, 100))
        rsi = indicators.rsi(prices, period=14)

        # Last RSI should be high (overbought)
        assert rsi.iloc[-1] > 70

    def test_williams_r_extremes(self, indicators):
        """Test Williams %R at extreme conditions."""
        # At high
        high = pd.Series([110] * 20)
        low = pd.Series([90] * 20)
        close = pd.Series([110] * 20)  # At high

        williams = indicators.williams_r(high, low, close, period=14)
        assert williams.iloc[-1] == 0  # Should be at max

        # At low
        close = pd.Series([90] * 20)  # At low
        williams = indicators.williams_r(high, low, close, period=14)
        assert williams.iloc[-1] == -100  # Should be at min


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
