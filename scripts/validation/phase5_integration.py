"""
Phase 5: Integration Validation
"""

import sys
import asyncio
import time
from pathlib import Path
from typing import Tuple, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from .base import BaseValidator


class Phase5IntegrationValidator(BaseValidator):
    """Validator for integration checks"""

    def __init__(self):
        super().__init__(5, "Integration & Performance")
        self.project_root = Path(__file__).parent.parent.parent

    def validate(self):
        """Run all Phase 5 validations"""
        self.start()

        print(f"\n🔍 Phase 5: {self.phase.phase_name}")
        print("=" * 70)

        # Run async checks
        asyncio.run(self._run_async_checks())

        self.end()
        return self.phase

    async def _run_async_checks(self):
        """Run all async validation checks"""
        await self._check_database_performance()
        await self._check_cache_performance()
        await self._check_end_to_end_prediction()
        await self._check_backtesting_engine()
        await self._check_concurrent_operations()

    async def _check_database_performance(self):
        """Check database performance"""
        async def check():
            try:
                from data.database import DatabaseManager
                from data.yfinance_client import YFinanceClient

                client = YFinanceClient()
                data = await client.get_historical_data('AAPL', period='1mo')

                db = DatabaseManager()
                await db.connect()

                # Test write performance
                start = time.time()
                await db.store_market_data('AAPL', data)
                write_time = (time.time() - start) * 1000

                # Test read performance
                start = time.time()
                retrieved = await db.get_market_data('AAPL', limit=100)
                read_time = (time.time() - start) * 1000

                await db.close()
                await client.close()

                if write_time < 200 and read_time < 100:
                    return True, f"Write: {write_time:.1f}ms, Read: {read_time:.1f}ms", {
                        'write_time_ms': write_time,
                        'read_time_ms': read_time,
                        'rows_written': len(data),
                        'rows_read': len(retrieved)
                    }
                elif write_time >= 200:
                    return False, f"Slow write: {write_time:.1f}ms (target < 200ms)", {
                        'write_time_ms': write_time
                    }
                else:
                    return False, f"Slow read: {read_time:.1f}ms (target < 100ms)", {
                        'read_time_ms': read_time
                    }

            except Exception as e:
                return False, f"Failed: {str(e)}", {}

        result = await self.run_async_check("Database Performance", check)
        print(f"  {result.status.value} Database Performance: {result.message}")

    async def _check_cache_performance(self):
        """Check Redis cache performance"""
        async def check():
            try:
                from cache.redis_client import CacheManager

                cache = CacheManager()
                await cache.connect()

                test_data = {'price': 123.45, 'volume': 1000000, 'indicators': list(range(50))}

                # Test write performance (100 operations)
                start = time.time()
                for i in range(100):
                    await cache.set(f'perf_test_{i}', test_data, ttl=60)
                write_time = (time.time() - start) * 1000
                avg_write = write_time / 100

                # Test read performance (100 operations)
                start = time.time()
                for i in range(100):
                    await cache.get(f'perf_test_{i}')
                read_time = (time.time() - start) * 1000
                avg_read = read_time / 100

                # Test hit rate
                hits = 0
                for i in range(100):
                    if await cache.get(f'perf_test_{i}'):
                        hits += 1

                # Cleanup
                for i in range(100):
                    await cache.delete(f'perf_test_{i}')

                await cache.close()

                if avg_write < 5 and avg_read < 3 and hits == 100:
                    return True, f"Write: {avg_write:.2f}ms, Read: {avg_read:.2f}ms, Hit rate: 100%", {
                        'avg_write_ms': avg_write,
                        'avg_read_ms': avg_read,
                        'hit_rate': 100
                    }
                elif avg_write >= 5:
                    return False, f"Slow write: {avg_write:.2f}ms avg (target < 5ms)", {
                        'avg_write_ms': avg_write
                    }
                elif avg_read >= 3:
                    return False, f"Slow read: {avg_read:.2f}ms avg (target < 3ms)", {
                        'avg_read_ms': avg_read
                    }
                else:
                    return False, f"Low hit rate: {hits}%", {
                        'hit_rate': hits
                    }

            except Exception as e:
                return False, f"Failed: {str(e)}", {}

        result = await self.run_async_check("Cache Performance", check)
        print(f"  {result.status.value} Cache Performance: {result.message}")

    async def _check_end_to_end_prediction(self):
        """Check end-to-end trading signal generation"""
        async def check():
            try:
                from ml.ensemble import EnsembleModel
                from ml.feature_engineering import FeatureEngineer
                from data.yfinance_client import YFinanceClient

                print("    Testing end-to-end prediction pipeline...")

                # Try to load pre-trained model
                ensemble = EnsembleModel()
                model_path = self.project_root / 'models' / 'ensemble_model'

                if model_path.exists():
                    await ensemble.load(str(model_path))
                    print("    Using pre-trained model")
                else:
                    # Train a quick model for testing
                    print("    Training model for test...")
                    client = YFinanceClient()
                    data = await client.get_historical_data('AAPL', period='1y')
                    fe = FeatureEngineer()
                    features = fe.create_features(data)
                    await ensemble.train(features, epochs_lstm=3)
                    await client.close()

                # Test predictions for multiple symbols
                symbols = ['AAPL', 'MSFT', 'GOOGL']
                signals = []

                for symbol in symbols:
                    signal = await ensemble.predict(symbol)
                    signals.append(signal)

                # Validate signals
                all_valid = True
                for signal in signals:
                    if not all([
                        signal.get('signal') in ['BUY', 'SELL', 'HOLD'],
                        0 <= signal.get('confidence', 0) <= 1,
                        signal.get('price', 0) > 0
                    ]):
                        all_valid = False

                if all_valid:
                    return True, f"Generated signals for {len(symbols)} symbols", {
                        'symbols': symbols,
                        'signals': [s['signal'] for s in signals],
                        'avg_confidence': sum(s['confidence'] for s in signals) / len(signals)
                    }
                else:
                    return False, "Invalid signals generated", {
                        'signals': signals
                    }

            except Exception as e:
                return False, f"Failed: {str(e)}", {}

        result = await self.run_async_check("End-to-End Prediction", check)
        print(f"  {result.status.value} End-to-End Prediction: {result.message}")

    async def _check_backtesting_engine(self):
        """Check backtesting engine"""
        async def check():
            try:
                from backtesting.engine import BacktestingEngine
                from backtesting.strategies.mean_reversion import MeanReversionStrategy
                from data.yfinance_client import YFinanceClient

                print("    Running backtest (this may take a minute)...")

                client = YFinanceClient()
                data = await client.get_historical_data('AAPL', period='6mo')
                await client.close()

                # Initialize strategy and engine
                strategy = MeanReversionStrategy()
                engine = BacktestingEngine(
                    initial_capital=100000,
                    commission=0.001,
                    slippage=0.001
                )

                # Run backtest
                results = engine.run(data, strategy)

                # Validate results
                has_required_metrics = all([
                    'total_return' in results,
                    'sharpe_ratio' in results,
                    'max_drawdown' in results,
                    'num_trades' in results
                ])

                if has_required_metrics and results['num_trades'] > 0:
                    return True, f"Completed with {results['num_trades']} trades, return: {results['total_return']:.2%}", {
                        'total_return': results['total_return'],
                        'sharpe_ratio': results.get('sharpe_ratio', 0),
                        'max_drawdown': results['max_drawdown'],
                        'num_trades': results['num_trades']
                    }
                elif not has_required_metrics:
                    return False, "Missing required metrics in results", {
                        'results': results
                    }
                else:
                    return False, "No trades executed", {
                        'num_trades': results.get('num_trades', 0)
                    }

            except Exception as e:
                return False, f"Failed: {str(e)}", {}

        result = await self.run_async_check("Backtesting Engine", check)
        print(f"  {result.status.value} Backtesting Engine: {result.message}")

    async def _check_concurrent_operations(self):
        """Check concurrent data fetching"""
        async def check():
            try:
                from data.yfinance_client import YFinanceClient

                print("    Testing concurrent operations...")

                symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
                client = YFinanceClient()

                # Fetch data concurrently
                start = time.time()
                tasks = [client.get_current_price(symbol) for symbol in symbols]
                prices = await asyncio.gather(*tasks, return_exceptions=True)
                duration = (time.time() - start) * 1000

                await client.close()

                # Check for errors
                errors = [p for p in prices if isinstance(p, Exception)]
                valid_prices = [p for p in prices if not isinstance(p, Exception) and p > 0]

                if len(errors) == 0 and len(valid_prices) == len(symbols):
                    avg_time = duration / len(symbols)
                    return True, f"Fetched {len(symbols)} prices in {duration:.0f}ms ({avg_time:.0f}ms avg)", {
                        'total_time_ms': duration,
                        'avg_time_ms': avg_time,
                        'symbols': len(symbols),
                        'errors': 0
                    }
                else:
                    return False, f"{len(errors)} errors, {len(valid_prices)}/{len(symbols)} successful", {
                        'errors': len(errors),
                        'successful': len(valid_prices),
                        'total': len(symbols)
                    }

            except Exception as e:
                return False, f"Failed: {str(e)}", {}

        result = await self.run_async_check("Concurrent Operations", check)
        print(f"  {result.status.value} Concurrent Operations: {result.message}")
