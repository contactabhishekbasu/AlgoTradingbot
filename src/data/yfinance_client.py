"""YFinance API client with rate limiting and retry logic."""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..utils.config import settings
from ..utils.logger import logger


class RateLimiter:
    """Token bucket rate limiter for API requests."""

    def __init__(
        self, max_requests: int = 2000, period: int = 3600, burst: int = 10
    ):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed per period
            period: Time period in seconds (default: 1 hour)
            burst: Maximum burst requests allowed
        """
        self.max_requests = max_requests
        self.period = period
        self.burst = burst
        self.tokens = max_requests
        self.last_update = time.time()
        self.lock_until = 0

    def acquire(self) -> bool:
        """
        Acquire a token for making a request.

        Returns:
            True if token acquired, False if rate limited

        Raises:
            Exception if rate limited
        """
        current_time = time.time()

        # Check if we're in a lock period
        if current_time < self.lock_until:
            wait_time = self.lock_until - current_time
            logger.warning(
                "rate_limit_locked",
                wait_time=wait_time,
                reason="Rate limit exceeded",
            )
            time.sleep(wait_time)
            return self.acquire()

        # Refill tokens based on elapsed time
        elapsed = current_time - self.last_update
        refill = (elapsed / self.period) * self.max_requests
        self.tokens = min(self.max_requests, self.tokens + refill)
        self.last_update = current_time

        # Try to acquire token
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        else:
            # Wait for token refill
            wait_time = (1 - self.tokens) * (self.period / self.max_requests)
            logger.info("rate_limit_waiting", wait_time=wait_time)
            time.sleep(wait_time)
            return self.acquire()


class YFinanceClient:
    """
    Wrapper for Yahoo Finance API with advanced features.

    Features:
    - Rate limiting (2000 requests/hour)
    - Retry logic with exponential backoff
    - Error handling and validation
    - Batch fetching support
    """

    def __init__(
        self,
        rate_limit_requests: int = None,
        rate_limit_period: int = None,
    ):
        """
        Initialize YFinance client.

        Args:
            rate_limit_requests: Max requests per period (default from settings)
            rate_limit_period: Period in seconds (default from settings)
        """
        self.rate_limiter = RateLimiter(
            max_requests=rate_limit_requests or settings.rate_limit_requests,
            period=rate_limit_period or settings.rate_limit_period,
        )

        logger.info(
            "yfinance_client_initialized",
            rate_limit=rate_limit_requests or settings.rate_limit_requests,
            period=rate_limit_period or settings.rate_limit_period,
        )

    def _validate_symbol(self, symbol: str) -> str:
        """
        Validate stock symbol format.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Validated and normalized symbol

        Raises:
            ValueError: If symbol format is invalid
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"Invalid symbol: {symbol}")

        symbol = symbol.strip().upper()

        # Basic validation (1-5 uppercase letters)
        if not symbol.replace(".", "").replace("-", "").isalpha():
            raise ValueError(f"Invalid symbol format: {symbol}")

        if len(symbol) > 10:
            raise ValueError(f"Symbol too long: {symbol}")

        return symbol

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=16),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True,
    )
    def get_current_price(self, symbol: str) -> Dict[str, Union[float, str]]:
        """
        Get current price for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with price data:
            {
                'symbol': str,
                'price': float,
                'volume': int,
                'timestamp': str (ISO format),
                'market_state': str
            }

        Raises:
            ValueError: If symbol is invalid
            Exception: If API request fails after retries
        """
        symbol = self._validate_symbol(symbol)
        self.rate_limiter.acquire()

        logger.debug("fetching_current_price", symbol=symbol)

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or "currentPrice" not in info:
                # Fallback to fast_info
                price = ticker.fast_info.get("lastPrice")
                if price is None:
                    raise ValueError(f"Unable to fetch price for {symbol}")
            else:
                price = info.get("currentPrice") or info.get("regularMarketPrice")

            result = {
                "symbol": symbol,
                "price": float(price),
                "volume": info.get("volume", 0),
                "timestamp": datetime.now().isoformat(),
                "market_state": info.get("marketState", "UNKNOWN"),
            }

            logger.info("current_price_fetched", **result)
            return result

        except Exception as e:
            logger.error(
                "failed_to_fetch_price",
                symbol=symbol,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=16),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True,
    )
    def get_historical_data(
        self,
        symbol: str,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.

        Args:
            symbol: Stock ticker symbol
            start: Start date (YYYY-MM-DD or datetime)
            end: End date (YYYY-MM-DD or datetime)
            period: Period to download (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close

        Raises:
            ValueError: If symbol is invalid or no data available
        """
        symbol = self._validate_symbol(symbol)
        self.rate_limiter.acquire()

        logger.debug(
            "fetching_historical_data",
            symbol=symbol,
            start=start,
            end=end,
            period=period,
            interval=interval,
        )

        try:
            ticker = yf.Ticker(symbol)

            # Download historical data
            if start and end:
                data = ticker.history(start=start, end=end, interval=interval)
            else:
                data = ticker.history(period=period, interval=interval)

            if data.empty:
                raise ValueError(f"No data available for {symbol}")

            # Add symbol column
            data["Symbol"] = symbol

            # Reset index to make date a column
            data.reset_index(inplace=True)

            logger.info(
                "historical_data_fetched",
                symbol=symbol,
                rows=len(data),
                start_date=data["Date"].min(),
                end_date=data["Date"].max(),
            )

            return data

        except Exception as e:
            logger.error(
                "failed_to_fetch_historical_data",
                symbol=symbol,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def get_multiple_symbols(
        self,
        symbols: List[str],
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        period: str = "1y",
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols.

        Args:
            symbols: List of stock ticker symbols
            start: Start date
            end: End date
            period: Period to download
            interval: Data interval

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        failed = []

        for symbol in symbols:
            try:
                data = self.get_historical_data(
                    symbol=symbol,
                    start=start,
                    end=end,
                    period=period,
                    interval=interval,
                )
                results[symbol] = data
            except Exception as e:
                logger.warning(
                    "failed_to_fetch_symbol",
                    symbol=symbol,
                    error=str(e),
                )
                failed.append(symbol)

        if failed:
            logger.warning(
                "some_symbols_failed",
                failed_count=len(failed),
                failed_symbols=failed,
                successful_count=len(results),
            )

        return results

    def get_market_status(self) -> Dict[str, Union[bool, str]]:
        """
        Check if US markets are currently open.

        Returns:
            Dictionary with market status information
        """
        try:
            # Use SPY as proxy for market status
            ticker = yf.Ticker("SPY")
            info = ticker.info

            market_state = info.get("marketState", "UNKNOWN")
            is_open = market_state == "REGULAR"

            result = {
                "is_open": is_open,
                "market_state": market_state,
                "timestamp": datetime.now().isoformat(),
            }

            logger.debug("market_status_checked", **result)
            return result

        except Exception as e:
            logger.error("failed_to_check_market_status", error=str(e))
            return {
                "is_open": False,
                "market_state": "UNKNOWN",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }

    def get_company_info(self, symbol: str) -> Dict[str, any]:
        """
        Get company information and fundamentals.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with company information
        """
        symbol = self._validate_symbol(symbol)
        self.rate_limiter.acquire()

        logger.debug("fetching_company_info", symbol=symbol)

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Extract relevant information
            result = {
                "symbol": symbol,
                "name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                "52_week_low": info.get("fiftyTwoWeekLow", 0),
                "avg_volume": info.get("averageVolume", 0),
                "description": info.get("longBusinessSummary", ""),
            }

            logger.info("company_info_fetched", symbol=symbol)
            return result

        except Exception as e:
            logger.error(
                "failed_to_fetch_company_info",
                symbol=symbol,
                error=str(e),
            )
            raise
