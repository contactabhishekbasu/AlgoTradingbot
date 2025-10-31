"""
Phase 2: Data Layer Validation
"""

import sys
import os
import asyncio
from pathlib import Path
from typing import Tuple, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from .base import BaseValidator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Phase2DataValidator(BaseValidator):
    """Validator for data layer checks"""

    def __init__(self):
        super().__init__(2, "Data Layer")

    def validate(self):
        """Run all Phase 2 validations"""
        self.start()

        print(f"\n🔍 Phase 2: {self.phase.phase_name}")
        print("=" * 70)

        # Run async checks
        asyncio.run(self._run_async_checks())

        self.end()
        return self.phase

    async def _run_async_checks(self):
        """Run all async validation checks"""
        await self._check_api_connectivity()
        await self._check_historical_data()
        await self._check_data_validation()
        await self._check_technical_indicators()
        await self._check_database_connection()
        await self._check_database_storage()
        await self._check_redis_cache()

    async def _check_api_connectivity(self):
        """Check API connectivity to Yahoo Finance"""
        async def check():
            try:
                from data.yfinance_client import YFinanceClient

                client = YFinanceClient()
                price = await client.get_current_price('AAPL')
                await client.close()

                if price and 50 < price < 300:  # Reasonable range for AAPL
                    return True, f"Connected, AAPL price: ${price:.2f}", {'price': price}
                else:
                    return False, f"Unexpected price: ${price:.2f}", {'price': price}

            except Exception as e:
                return False, f"Failed to connect: {str(e)}", {}

        result = await self.run_async_check("API Connectivity", check)
        print(f"  {result.status.value} API Connectivity: {result.message}")

    async def _check_historical_data(self):
        """Check historical data fetching"""
        async def check():
            try:
                from data.yfinance_client import YFinanceClient

                client = YFinanceClient()
                data = await client.get_historical_data('AAPL', period='5d')
                await client.close()

                if len(data) >= 5:
                    return True, f"Fetched {len(data)} days of data", {
                        'rows': len(data),
                        'columns': list(data.columns)
                    }
                else:
                    return False, f"Only fetched {len(data)} days (expected 5)", {
                        'rows': len(data)
                    }

            except Exception as e:
                return False, f"Failed: {str(e)}", {}

        result = await self.run_async_check("Historical Data Fetching", check)
        print(f"  {result.status.value} Historical Data Fetching: {result.message}")

    async def _check_data_validation(self):
        """Check data validation"""
        async def check():
            try:
                from data.yfinance_client import YFinanceClient
                from data.validators import DataValidator

                client = YFinanceClient()
                data = await client.get_historical_data('AAPL', period='1mo')

                validator = DataValidator()
                is_valid, errors = validator.validate_ohlcv(data)

                await client.close()

                if is_valid:
                    return True, f"Validated {len(data)} rows successfully", {
                        'rows': len(data),
                        'errors': []
                    }
                else:
                    return False, f"Validation failed: {', '.join(errors)}", {
                        'rows': len(data),
                        'errors': errors
                    }

            except Exception as e:
                return False, f"Failed: {str(e)}", {}

        result = await self.run_async_check("Data Validation", check)
        print(f"  {result.status.value} Data Validation: {result.message}")

    async def _check_technical_indicators(self):
        """Check technical indicators calculation"""
        async def check():
            try:
                from indicators.technical_indicators import TechnicalIndicators
                from data.yfinance_client import YFinanceClient

                client = YFinanceClient()
                data = await client.get_historical_data('AAPL', period='3mo')

                ti = TechnicalIndicators()
                rsi = ti.calculate_rsi(data['Close'])

                await client.close()

                last_rsi = rsi.iloc[-1]

                if 0 <= last_rsi <= 100:
                    return True, f"RSI calculated: {last_rsi:.2f}", {
                        'rsi': last_rsi,
                        'valid_range': True
                    }
                else:
                    return False, f"RSI out of range: {last_rsi:.2f}", {
                        'rsi': last_rsi,
                        'valid_range': False
                    }

            except Exception as e:
                return False, f"Failed: {str(e)}", {}

        result = await self.run_async_check("Technical Indicators", check)
        print(f"  {result.status.value} Technical Indicators: {result.message}")

    async def _check_database_connection(self):
        """Check PostgreSQL database connection"""
        async def check():
            try:
                from data.database import DatabaseClient

                db = DatabaseClient()
                await db.connect()

                # Test query
                result = await db.execute_query("SELECT version();")

                await db.close()

                return True, "Connected successfully", {'connected': True}

            except Exception as e:
                return False, f"Connection failed: {str(e)}", {}

        result = await self.run_async_check("Database Connection", check)
        print(f"  {result.status.value} Database Connection: {result.message}")

    async def _check_database_storage(self):
        """Check database storage operations"""
        async def check():
            try:
                from data.database import DatabaseManager
                from data.yfinance_client import YFinanceClient

                client = YFinanceClient()
                data = await client.get_historical_data('AAPL', period='5d')

                db = DatabaseManager()
                await db.connect()

                # Store data
                rows_stored = await db.store_market_data('AAPL', data)

                # Retrieve data
                retrieved = await db.get_market_data('AAPL', limit=5)

                await db.close()
                await client.close()

                if rows_stored >= 5 and len(retrieved) >= 5:
                    return True, f"Stored {rows_stored} rows, retrieved {len(retrieved)}", {
                        'stored': rows_stored,
                        'retrieved': len(retrieved)
                    }
                else:
                    return False, f"Storage mismatch: stored {rows_stored}, retrieved {len(retrieved)}", {
                        'stored': rows_stored,
                        'retrieved': len(retrieved)
                    }

            except Exception as e:
                return False, f"Failed: {str(e)}", {}

        result = await self.run_async_check("Database Storage", check)
        print(f"  {result.status.value} Database Storage: {result.message}")

    async def _check_redis_cache(self):
        """Check Redis cache operations"""
        async def check():
            try:
                from cache.redis_client import RedisCache

                cache = RedisCache()
                await cache.connect()

                # Test set
                test_key = 'validation_test'
                test_value = {'price': 123.45, 'timestamp': '2025-10-31'}
                await cache.set(test_key, test_value, ttl=60)

                # Test get
                retrieved = await cache.get(test_key)

                # Test delete
                await cache.delete(test_key)

                await cache.close()

                if retrieved and retrieved.get('price') == 123.45:
                    return True, "Cache set/get/delete working", {'retrieved': retrieved}
                else:
                    return False, "Cache operations failed", {'retrieved': retrieved}

            except Exception as e:
                return False, f"Failed: {str(e)}", {}

        result = await self.run_async_check("Redis Cache", check)
        print(f"  {result.status.value} Redis Cache: {result.message}")
