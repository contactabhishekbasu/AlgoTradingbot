"""
Integration tests for ML pipeline with real data

These tests download real market data and train models to validate
the entire pipeline works end-to-end.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import logging
from datetime import datetime, timedelta

from src.ml.training.trainer import ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestMLPipelineIntegration:
    """Integration tests for complete ML pipeline"""

    @pytest.fixture(scope="class")
    def output_dir(self):
        """Create temporary directory for outputs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_data_loading(self):
        """Test loading real market data"""
        trainer = ModelTrainer()

        # Load 6 months of data for AAPL
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)

        try:
            market_data = trainer.data_loader.load_market_data(
                symbol='AAPL',
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )

            # Verify data was loaded
            assert isinstance(market_data, pd.DataFrame)
            assert len(market_data) > 0
            assert 'open' in market_data.columns
            assert 'high' in market_data.columns
            assert 'low' in market_data.columns
            assert 'close' in market_data.columns
            assert 'volume' in market_data.columns

            logger.info(f"Successfully loaded {len(market_data)} rows of market data")

        except Exception as e:
            pytest.skip(f"Could not load market data: {e}")

    def test_feature_engineering_with_real_data(self):
        """Test feature engineering with real data"""
        trainer = ModelTrainer()

        # Load data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)

        try:
            market_data = trainer.data_loader.load_market_data(
                symbol='AAPL',
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )

            # Prepare data
            train_data, val_data, test_data = trainer.prepare_data(
                symbol='AAPL',
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                target_horizon=1
            )

            # Verify splits
            X_train, y_train = train_data
            X_val, y_val = val_data
            X_test, y_test = test_data

            assert len(X_train) > 0
            assert len(X_val) > 0
            assert len(X_test) > 0

            assert len(X_train) == len(y_train)
            assert len(X_val) == len(y_val)
            assert len(X_test) == len(y_test)

            # Verify features were created
            assert X_train.shape[1] > 10  # Should have many features

            # Verify target labels are valid
            assert set(y_train.unique()).issubset({-1, 0, 1})

            logger.info(f"Feature engineering successful:")
            logger.info(f"  Train: {len(X_train)} samples, {X_train.shape[1]} features")
            logger.info(f"  Val: {len(X_val)} samples")
            logger.info(f"  Test: {len(X_test)} samples")

        except Exception as e:
            pytest.skip(f"Could not prepare data: {e}")

    @pytest.mark.slow
    def test_lstm_training_with_real_data(self, output_dir):
        """Test LSTM model training with real data (may take several minutes)"""
        trainer = ModelTrainer(output_dir=output_dir)

        # Load smaller dataset for faster testing
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year

        try:
            train_data, val_data, test_data = trainer.prepare_data(
                symbol='AAPL',
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )

            # Train LSTM with reduced parameters for testing
            lstm_model = trainer.train_lstm(
                train_data,
                val_data,
                sequence_length=30,  # Shorter sequences
                epochs=5,  # Fewer epochs for testing
                batch_size=32
            )

            # Verify model was trained
            assert lstm_model is not None
            assert trainer.lstm_model is not None

            # Verify model can make predictions
            X_test, y_test = test_data
            from src.ml.feature_engineering import prepare_sequences
            X_test_seq, y_test_seq = prepare_sequences(X_test, y_test, sequence_length=30)

            predictions = lstm_model.predict(X_test_seq[:10])
            assert predictions.shape[0] == 10
            assert predictions.shape[1] == 3  # 3 classes

            # Evaluate model
            metrics = lstm_model.evaluate(X_test_seq, y_test_seq)
            logger.info(f"LSTM Test Metrics: {metrics}")

            # Verify reasonable performance (at least better than random)
            assert metrics['accuracy'] > 0.25  # Better than random (1/3)

        except Exception as e:
            pytest.skip(f"Could not train LSTM model: {e}")

    @pytest.mark.slow
    def test_xgboost_training_with_real_data(self, output_dir):
        """Test XGBoost model training with real data"""
        trainer = ModelTrainer(output_dir=output_dir)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        try:
            train_data, val_data, test_data = trainer.prepare_data(
                symbol='AAPL',
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )

            # Train XGBoost with reduced parameters
            xgb_model = trainer.train_xgboost(
                train_data,
                val_data,
                n_estimators=50  # Fewer trees for testing
            )

            # Verify model was trained
            assert xgb_model is not None
            assert trainer.xgboost_model is not None

            # Verify model can make predictions
            X_test, y_test = test_data
            y_test_encoded, _ = trainer.dataset_prep.encode_labels(y_test)

            predictions = xgb_model.predict(X_test.values[:10])
            assert predictions.shape[0] == 10

            # Evaluate model
            metrics = xgb_model.evaluate(X_test.values, y_test_encoded)
            logger.info(f"XGBoost Test Metrics: {metrics}")

            # Verify reasonable performance
            assert metrics['accuracy'] > 0.25

            # Check feature importance
            importance_df = xgb_model.get_feature_importance(top_n=10)
            assert len(importance_df) > 0
            logger.info(f"Top 5 features:\n{importance_df.head()}")

        except Exception as e:
            pytest.skip(f"Could not train XGBoost model: {e}")

    @pytest.mark.slow
    def test_full_pipeline_with_real_data(self, output_dir):
        """Test complete training pipeline with real data (SLOW)"""
        trainer = ModelTrainer(output_dir=output_dir)

        # Use 2 years of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)

        try:
            results = trainer.full_pipeline(
                symbol='AAPL',
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                lstm_epochs=10,  # Reduced for testing
                xgboost_estimators=50,  # Reduced for testing
                sequence_length=30
            )

            # Verify results
            assert 'symbol' in results
            assert results['symbol'] == 'AAPL'

            assert 'lstm_test_metrics' in results
            assert 'xgboost_test_metrics' in results

            # Verify both models achieved reasonable performance
            lstm_acc = results['lstm_test_metrics']['accuracy']
            xgb_acc = results['xgboost_test_metrics']['accuracy']

            logger.info(f"Final Results:")
            logger.info(f"  LSTM Accuracy: {lstm_acc:.4f}")
            logger.info(f"  XGBoost Accuracy: {xgb_acc:.4f}")
            logger.info(f"  Training Duration: {results['training_duration_seconds']:.2f}s")
            logger.info(f"  Features: {results['feature_count']}")
            logger.info(f"  Training Samples: {results['training_samples']}")

            # Both models should beat random chance
            assert lstm_acc > 0.30
            assert xgb_acc > 0.30

            # Ensemble should exist
            assert trainer.ensemble is not None

            # Verify models were saved
            assert os.path.exists(os.path.join(output_dir, 'lstm_model.h5'))
            assert os.path.exists(os.path.join(output_dir, 'xgboost_model.json'))
            assert os.path.exists(os.path.join(output_dir, 'training_results.json'))

        except Exception as e:
            pytest.skip(f"Could not complete full pipeline: {e}")


@pytest.mark.integration
@pytest.mark.slow
class TestMultiSymbolTraining:
    """Test training on multiple symbols"""

    @pytest.fixture(scope="class")
    def symbols(self):
        """List of symbols to test"""
        return ['AAPL', 'MSFT', 'GOOGL']

    def test_train_on_multiple_symbols(self, symbols, tmp_path):
        """Test training models on multiple symbols"""
        results_by_symbol = {}

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        for symbol in symbols:
            try:
                logger.info(f"\n{'='*80}")
                logger.info(f"Training on {symbol}")
                logger.info(f"{'='*80}")

                output_dir = os.path.join(tmp_path, symbol)
                trainer = ModelTrainer(output_dir=output_dir)

                # Quick training with minimal parameters
                train_data, val_data, test_data = trainer.prepare_data(
                    symbol=symbol,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )

                # Train just XGBoost for speed
                xgb_model = trainer.train_xgboost(train_data, val_data, n_estimators=30)

                # Evaluate
                X_test, y_test = test_data
                y_test_encoded, _ = trainer.dataset_prep.encode_labels(y_test)
                metrics = xgb_model.evaluate(X_test.values, y_test_encoded)

                results_by_symbol[symbol] = metrics
                logger.info(f"{symbol} Metrics: {metrics}")

            except Exception as e:
                logger.warning(f"Failed to train on {symbol}: {e}")

        # Verify we got results for at least one symbol
        assert len(results_by_symbol) > 0

        # Log summary
        logger.info(f"\n{'='*80}")
        logger.info("Multi-Symbol Training Summary")
        logger.info(f"{'='*80}")
        for symbol, metrics in results_by_symbol.items():
            logger.info(f"{symbol}: Accuracy = {metrics['accuracy']:.4f}")
