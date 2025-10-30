"""Unit tests for YFinance client."""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.data.yfinance_client import YFinanceClient, RateLimiter


class TestRateLimiter:
    """Test rate limiting functionality."""

    def test_rate_limiter_init(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(max_requests=100, period=60)
        assert limiter.max_requests == 100
        assert limiter.period == 60
        assert limiter.tokens == 100

    def test_rate_limiter_acquire(self):
        """Test acquiring tokens."""
        limiter = RateLimiter(max_requests=10, period=1)
        assert limiter.acquire() is True
        assert limiter.tokens < 10

    def test_rate_limiter_exhaustion(self):
        """Test rate limit exhaustion."""
        limiter = RateLimiter(max_requests=2, period=1)
        limiter.acquire()
        limiter.acquire()
        # Third request should wait
        import time
        start = time.time()
        limiter.acquire()
        duration = time.time() - start
        assert duration > 0  # Should have waited


class TestYFinanceClient:
    """Test YFinance client functionality."""

    @pytest.fixture
    def client(self):
        """Create YFinance client instance."""
        return YFinanceClient(rate_limit_requests=1000, rate_limit_period=3600)

    def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.rate_limiter is not None
        assert client.rate_limiter.max_requests == 1000

    def test_validate_symbol_valid(self, client):
        """Test valid symbol validation."""
        assert client._validate_symbol("AAPL") == "AAPL"
        assert client._validate_symbol("msft") == "MSFT"
        assert client._validate_symbol("  GOOGL  ") == "GOOGL"

    def test_validate_symbol_invalid(self, client):
        """Test invalid symbol validation."""
        with pytest.raises(ValueError):
            client._validate_symbol("")

        with pytest.raises(ValueError):
            client._validate_symbol("12345")

        with pytest.raises(ValueError):
            client._validate_symbol("TOOLONGSYMBOL")

    @patch("src.data.yfinance_client.yf.Ticker")
    def test_get_current_price_success(self, mock_ticker, client):
        """Test successful price fetch."""
        # Mock ticker data
        mock_info = {
            "currentPrice": 150.50,
            "volume": 1000000,
            "marketState": "REGULAR",
        }
        mock_ticker.return_value.info = mock_info

        result = client.get_current_price("AAPL")

        assert result["symbol"] == "AAPL"
        assert result["price"] == 150.50
        assert result["volume"] == 1000000
        assert result["market_state"] == "REGULAR"

    @patch("src.data.yfinance_client.yf.Ticker")
    def test_get_current_price_with_fast_info(self, mock_ticker, client):
        """Test price fetch using fast_info fallback."""
        # Mock ticker with empty info but valid fast_info
        mock_ticker.return_value.info = {}
        mock_ticker.return_value.fast_info = {"lastPrice": 145.25}

        result = client.get_current_price("AAPL")

        assert result["symbol"] == "AAPL"
        assert result["price"] == 145.25

    @patch("src.data.yfinance_client.yf.Ticker")
    def test_get_historical_data_success(self, mock_ticker, client):
        """Test successful historical data fetch."""
        # Create sample data
        dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
        sample_data = pd.DataFrame(
            {
                "Open": [100.0] * len(dates),
                "High": [105.0] * len(dates),
                "Low": [95.0] * len(dates),
                "Close": [102.0] * len(dates),
                "Volume": [1000000] * len(dates),
            },
            index=dates,
        )

        mock_ticker.return_value.history.return_value = sample_data

        result = client.get_historical_data("AAPL", period="1mo")

        assert not result.empty
        assert "Symbol" in result.columns
        assert result["Symbol"].iloc[0] == "AAPL"
        assert len(result) == len(sample_data)

    @patch("src.data.yfinance_client.yf.Ticker")
    def test_get_historical_data_empty(self, mock_ticker, client):
        """Test historical data fetch with no data."""
        mock_ticker.return_value.history.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="No data available"):
            client.get_historical_data("INVALID")

    def test_get_multiple_symbols(self, client):
        """Test fetching multiple symbols."""
        with patch.object(client, "get_historical_data") as mock_fetch:
            # Mock successful fetch for first symbol
            mock_fetch.side_effect = [
                pd.DataFrame({"Close": [100, 101, 102]}),
                Exception("Network error"),  # Second symbol fails
            ]

            results = client.get_multiple_symbols(["AAPL", "INVALID"], period="1mo")

            assert "AAPL" in results
            assert "INVALID" not in results

    @patch("src.data.yfinance_client.yf.Ticker")
    def test_get_market_status(self, mock_ticker, client):
        """Test market status check."""
        mock_ticker.return_value.info = {"marketState": "REGULAR"}

        result = client.get_market_status()

        assert result["is_open"] is True
        assert result["market_state"] == "REGULAR"

    @patch("src.data.yfinance_client.yf.Ticker")
    def test_get_company_info(self, mock_ticker, client):
        """Test company information fetch."""
        mock_info = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "marketCap": 3000000000000,
            "trailingPE": 25.5,
        }
        mock_ticker.return_value.info = mock_info

        result = client.get_company_info("AAPL")

        assert result["symbol"] == "AAPL"
        assert result["name"] == "Apple Inc."
        assert result["sector"] == "Technology"
        assert result["market_cap"] == 3000000000000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
