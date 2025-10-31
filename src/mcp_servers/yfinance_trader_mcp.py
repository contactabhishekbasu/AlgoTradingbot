"""YFinance Trader MCP Server - Market data and technical analysis."""

import asyncio
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from cache.redis_client import CacheKeys, RedisCache
from data.database import DatabaseClient
from data.validators import DataValidator
from data.yfinance_client import YFinanceClient
from indicators.technical_indicators import TechnicalIndicators
from utils.config import settings
from utils.logger import logger


class YFinanceTraderMCP:
    """
    YFinance Trader MCP Server.

    Provides 8 tool endpoints:
    1. get_current_price - Real-time price data
    2. get_historical_data - Historical OHLCV data
    3. calculate_indicators - Technical indicators
    4. get_market_status - Market open/close status
    5. bulk_fetch - Multiple symbols at once
    6. get_rsi - Specific RSI calculation
    7. get_williams_r - Williams %R calculation
    8. get_bollinger_bands - Bollinger Bands calculation
    """

    def __init__(self):
        """Initialize MCP server components."""
        self.yfinance_client = YFinanceClient()
        self.cache = RedisCache()
        self.db = DatabaseClient()
        self.validator = DataValidator()
        self.indicators = TechnicalIndicators()

        logger.info("yfinance_trader_mcp_initialized")

    async def get_current_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get current price for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with price data
        """
        logger.info("get_current_price_called", symbol=symbol)

        # Check cache first
        cache_key = CacheKeys.market_price(symbol)
        cached = self.cache.get(cache_key)

        if cached:
            logger.debug("price_from_cache", symbol=symbol)
            return cached

        # Fetch from API
        price_data = self.yfinance_client.get_current_price(symbol)

        # Cache result
        self.cache.set(cache_key, price_data, ttl=60)  # 1 minute TTL

        return price_data

    async def get_historical_data(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "1y",
        interval: str = "1d",
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Get historical OHLCV data for a symbol.

        Args:
            symbol: Stock ticker symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            period: Period to download
            interval: Data interval
            validate: Run data validation

        Returns:
            Dictionary with historical data and metadata
        """
        logger.info(
            "get_historical_data_called",
            symbol=symbol,
            start=start,
            end=end,
            period=period,
            interval=interval,
        )

        # Check cache first
        cache_key = CacheKeys.historical_data(symbol, start or period, end or "", interval)
        cached = self.cache.get(cache_key, serializer="pickle")

        if cached:
            logger.debug("historical_data_from_cache", symbol=symbol)
            return cached

        # Try database first for daily data
        if interval == "1d" and start and end:
            try:
                db_data = self.db.fetch_market_data(symbol, start, end)
                if not db_data.empty:
                    logger.debug("historical_data_from_database", symbol=symbol, rows=len(db_data))

                    result = {
                        "symbol": symbol,
                        "data": db_data.to_dict("records"),
                        "rows": len(db_data),
                        "source": "database",
                        "start_date": db_data["Date"].min(),
                        "end_date": db_data["Date"].max(),
                    }

                    # Cache for 1 hour
                    self.cache.set(cache_key, result, ttl=3600, serializer="pickle")
                    return result
            except Exception as e:
                logger.warning("database_fetch_failed", error=str(e))

        # Fetch from YFinance API
        data = self.yfinance_client.get_historical_data(
            symbol=symbol,
            start=start,
            end=end,
            period=period,
            interval=interval,
        )

        # Validate data quality
        quality_report = None
        if validate:
            data, quality_report = self.validator.validate(data, symbol)

            if quality_report.has_errors():
                logger.warning(
                    "data_quality_issues",
                    symbol=symbol,
                    issues=quality_report.summary(),
                )

        # Store in database for daily data
        if interval == "1d":
            try:
                self.db.store_market_data(data, symbol)
            except Exception as e:
                logger.error("failed_to_store_market_data", symbol=symbol, error=str(e))

        result = {
            "symbol": symbol,
            "data": data.to_dict("records"),
            "rows": len(data),
            "source": "yfinance",
            "start_date": str(data["Date"].min()),
            "end_date": str(data["Date"].max()),
        }

        if quality_report:
            result["quality_report"] = quality_report.to_dict()

        # Cache for 1 hour (or 1 day for historical data)
        cache_ttl = 86400 if interval == "1d" else 3600
        self.cache.set(cache_key, result, ttl=cache_ttl, serializer="pickle")

        return result

    async def calculate_indicators(
        self,
        symbol: str,
        indicators: Optional[List[str]] = None,
        period: str = "1y",
    ) -> Dict[str, Any]:
        """
        Calculate technical indicators for a symbol.

        Args:
            symbol: Stock ticker symbol
            indicators: List of indicators to calculate
            period: Historical period for calculation

        Returns:
            Dictionary with calculated indicators
        """
        logger.info(
            "calculate_indicators_called",
            symbol=symbol,
            indicators=indicators,
        )

        # Default indicators
        if indicators is None:
            indicators = ["rsi", "williams_r", "bbands", "macd"]

        # Check cache
        cache_key = CacheKeys.indicator(symbol, ",".join(sorted(indicators)), period)
        cached = self.cache.get(cache_key, serializer="pickle")

        if cached:
            logger.debug("indicators_from_cache", symbol=symbol)
            return cached

        # Get historical data
        hist_data = await self.get_historical_data(symbol, period=period)
        import pandas as pd
        df = pd.DataFrame(hist_data["data"])

        # Calculate indicators
        df_with_indicators = self.indicators.calculate_all(df, indicators)

        # Get latest values
        latest = df_with_indicators.iloc[-1].to_dict()

        result = {
            "symbol": symbol,
            "timestamp": str(datetime.now()),
            "indicators": {
                col: latest[col]
                for col in df_with_indicators.columns
                if col not in ["Date", "Open", "High", "Low", "Close", "Volume", "Symbol"]
            },
            "price": latest["Close"],
        }

        # Cache for 1 minute
        self.cache.set(cache_key, result, ttl=60, serializer="pickle")

        return result

    async def get_market_status(self) -> Dict[str, Any]:
        """
        Check if markets are open.

        Returns:
            Dictionary with market status
        """
        logger.info("get_market_status_called")

        # Check cache
        cache_key = "market:status"
        cached = self.cache.get(cache_key)

        if cached:
            return cached

        # Get market status
        status = self.yfinance_client.get_market_status()

        # Cache for 1 minute
        self.cache.set(cache_key, status, ttl=60)

        return status

    async def bulk_fetch(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
    ) -> Dict[str, Any]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of stock symbols
            period: Historical period
            interval: Data interval

        Returns:
            Dictionary mapping symbols to data
        """
        logger.info("bulk_fetch_called", symbols=symbols, count=len(symbols))

        results = {}
        errors = []

        for symbol in symbols:
            try:
                data = await self.get_historical_data(
                    symbol=symbol,
                    period=period,
                    interval=interval,
                )
                results[symbol] = data
            except Exception as e:
                logger.error("bulk_fetch_failed", symbol=symbol, error=str(e))
                errors.append({"symbol": symbol, "error": str(e)})

        return {
            "results": results,
            "successful": len(results),
            "failed": len(errors),
            "errors": errors,
        }

    async def get_rsi(
        self,
        symbol: str,
        period: int = 14,
    ) -> Dict[str, Any]:
        """
        Calculate RSI for a symbol.

        Args:
            symbol: Stock ticker symbol
            period: RSI period

        Returns:
            Dictionary with RSI value and interpretation
        """
        logger.info("get_rsi_called", symbol=symbol, period=period)

        # Get historical data
        hist_data = await self.get_historical_data(symbol, period="3mo")
        import pandas as pd
        df = pd.DataFrame(hist_data["data"])

        # Calculate RSI
        rsi = self.indicators.rsi(df["Close"], period)
        current_rsi = float(rsi.iloc[-1])

        # Interpretation
        if current_rsi < 30:
            signal = "OVERSOLD"
            recommendation = "POTENTIAL_BUY"
        elif current_rsi > 70:
            signal = "OVERBOUGHT"
            recommendation = "POTENTIAL_SELL"
        else:
            signal = "NEUTRAL"
            recommendation = "HOLD"

        return {
            "symbol": symbol,
            "rsi": current_rsi,
            "period": period,
            "signal": signal,
            "recommendation": recommendation,
            "timestamp": str(datetime.now()),
        }

    async def get_williams_r(
        self,
        symbol: str,
        period: int = 14,
    ) -> Dict[str, Any]:
        """
        Calculate Williams %R for a symbol.

        Args:
            symbol: Stock ticker symbol
            period: Williams %R period

        Returns:
            Dictionary with Williams %R value and interpretation
        """
        logger.info("get_williams_r_called", symbol=symbol, period=period)

        # Get historical data
        hist_data = await self.get_historical_data(symbol, period="3mo")
        import pandas as pd
        df = pd.DataFrame(hist_data["data"])

        # Calculate Williams %R
        williams = self.indicators.williams_r(
            df["High"], df["Low"], df["Close"], period
        )
        current_williams = float(williams.iloc[-1])

        # Interpretation
        if current_williams < -80:
            signal = "OVERSOLD"
            recommendation = "POTENTIAL_BUY"
        elif current_williams > -20:
            signal = "OVERBOUGHT"
            recommendation = "POTENTIAL_SELL"
        else:
            signal = "NEUTRAL"
            recommendation = "HOLD"

        return {
            "symbol": symbol,
            "williams_r": current_williams,
            "period": period,
            "signal": signal,
            "recommendation": recommendation,
            "timestamp": str(datetime.now()),
        }

    async def get_bollinger_bands(
        self,
        symbol: str,
        period: int = 20,
        num_std: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Calculate Bollinger Bands for a symbol.

        Args:
            symbol: Stock ticker symbol
            period: Period for calculation
            num_std: Number of standard deviations

        Returns:
            Dictionary with Bollinger Bands values
        """
        logger.info("get_bollinger_bands_called", symbol=symbol, period=period)

        # Get historical data
        hist_data = await self.get_historical_data(symbol, period="3mo")
        import pandas as pd
        df = pd.DataFrame(hist_data["data"])

        # Calculate Bollinger Bands
        upper, middle, lower = self.indicators.bollinger_bands(
            df["Close"], period, num_std
        )

        current_price = float(df["Close"].iloc[-1])
        current_upper = float(upper.iloc[-1])
        current_middle = float(middle.iloc[-1])
        current_lower = float(lower.iloc[-1])

        # Calculate position within bands
        band_width = current_upper - current_lower
        price_position = (current_price - current_lower) / band_width * 100

        # Interpretation
        if current_price > current_upper:
            signal = "ABOVE_UPPER"
            recommendation = "POTENTIAL_SELL"
        elif current_price < current_lower:
            signal = "BELOW_LOWER"
            recommendation = "POTENTIAL_BUY"
        else:
            signal = "WITHIN_BANDS"
            recommendation = "HOLD"

        return {
            "symbol": symbol,
            "price": current_price,
            "upper_band": current_upper,
            "middle_band": current_middle,
            "lower_band": current_lower,
            "band_width": band_width,
            "price_position_pct": price_position,
            "signal": signal,
            "recommendation": recommendation,
            "timestamp": str(datetime.now()),
        }


async def main():
    """Main entry point for MCP server."""
    logger.info("starting_yfinance_trader_mcp_server")

    # Initialize server
    server = Server("yfinance-trader")
    mcp = YFinanceTraderMCP()

    # Register tools
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available tools."""
        return [
            Tool(
                name="get_current_price",
                description="Get current real-time price for a stock symbol",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g., AAPL, MSFT)",
                        }
                    },
                    "required": ["symbol"],
                },
            ),
            Tool(
                name="get_historical_data",
                description="Get historical OHLCV data for a stock symbol",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol",
                        },
                        "start": {
                            "type": "string",
                            "description": "Start date (YYYY-MM-DD)",
                        },
                        "end": {
                            "type": "string",
                            "description": "End date (YYYY-MM-DD)",
                        },
                        "period": {
                            "type": "string",
                            "description": "Period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max)",
                            "default": "1y",
                        },
                        "interval": {
                            "type": "string",
                            "description": "Data interval (1d, 1h, 5m, etc.)",
                            "default": "1d",
                        },
                    },
                    "required": ["symbol"],
                },
            ),
            Tool(
                name="calculate_indicators",
                description="Calculate technical indicators for a stock",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol",
                        },
                        "indicators": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of indicators (rsi, williams_r, bbands, macd)",
                        },
                        "period": {
                            "type": "string",
                            "description": "Historical period for calculation",
                            "default": "1y",
                        },
                    },
                    "required": ["symbol"],
                },
            ),
            Tool(
                name="get_market_status",
                description="Check if US stock markets are currently open",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="bulk_fetch",
                description="Fetch historical data for multiple symbols",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of stock symbols",
                        },
                        "period": {
                            "type": "string",
                            "description": "Period to fetch",
                            "default": "1y",
                        },
                    },
                    "required": ["symbols"],
                },
            ),
            Tool(
                name="get_rsi",
                description="Calculate RSI (Relative Strength Index) for a stock",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol",
                        },
                        "period": {
                            "type": "integer",
                            "description": "RSI period",
                            "default": 14,
                        },
                    },
                    "required": ["symbol"],
                },
            ),
            Tool(
                name="get_williams_r",
                description="Calculate Williams %R indicator for a stock",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol",
                        },
                        "period": {
                            "type": "integer",
                            "description": "Williams %R period",
                            "default": 14,
                        },
                    },
                    "required": ["symbol"],
                },
            ),
            Tool(
                name="get_bollinger_bands",
                description="Calculate Bollinger Bands for a stock",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol",
                        },
                        "period": {
                            "type": "integer",
                            "description": "Period for calculation",
                            "default": 20,
                        },
                        "num_std": {
                            "type": "number",
                            "description": "Number of standard deviations",
                            "default": 2.0,
                        },
                    },
                    "required": ["symbol"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Any) -> List[TextContent]:
        """Handle tool calls."""
        try:
            if name == "get_current_price":
                result = await mcp.get_current_price(**arguments)
            elif name == "get_historical_data":
                result = await mcp.get_historical_data(**arguments)
            elif name == "calculate_indicators":
                result = await mcp.calculate_indicators(**arguments)
            elif name == "get_market_status":
                result = await mcp.get_market_status()
            elif name == "bulk_fetch":
                result = await mcp.bulk_fetch(**arguments)
            elif name == "get_rsi":
                result = await mcp.get_rsi(**arguments)
            elif name == "get_williams_r":
                result = await mcp.get_williams_r(**arguments)
            elif name == "get_bollinger_bands":
                result = await mcp.get_bollinger_bands(**arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

            import json
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        except Exception as e:
            logger.error("tool_call_failed", tool=name, error=str(e))
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": str(e),
                            "tool": name,
                            "arguments": arguments,
                        },
                        indent=2,
                    ),
                )
            ]

    # Run server
    async with stdio_server() as (read_stream, write_stream):
        logger.info("yfinance_trader_mcp_server_running")
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
