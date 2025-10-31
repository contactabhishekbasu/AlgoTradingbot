"""
Phase 3: ML Model Validation
"""

import sys
import asyncio
import os
from pathlib import Path
from typing import Tuple, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from .base import BaseValidator


class Phase3MLValidator(BaseValidator):
    """Validator for ML model checks"""

    def __init__(self, quick_mode=True):
        super().__init__(3, "ML Models")
        self.quick_mode = quick_mode  # Use fewer epochs for faster validation
        self.project_root = Path(__file__).parent.parent.parent

    def validate(self):
        """Run all Phase 3 validations"""
        self.start()

        print(f"\n🔍 Phase 3: {self.phase.phase_name}")
        if self.quick_mode:
            print("  (Quick mode: Using reduced epochs for faster validation)")
        print("=" * 70)

        # Run async checks
        asyncio.run(self._run_async_checks())

        self.end()
        return self.phase

    async def _run_async_checks(self):
        """Run all async validation checks"""
        await self._check_feature_engineering()
        await self._check_lstm_model()
        await self._check_xgboost_model()
        await self._check_ensemble_model()
        await self._check_model_persistence()

    async def _check_feature_engineering(self):
        """Check feature engineering"""
        async def check():
            try:
                from ml.feature_engineering import FeatureEngineer
                from data.yfinance_client import YFinanceClient

                client = YFinanceClient()
                data = await client.get_historical_data('AAPL', period='6mo')

                fe = FeatureEngineer()
                features = fe.create_features(data)

                await client.close()

                original_cols = len(data.columns)
                feature_cols = len(features.columns)
                missing_values = features.isnull().sum().sum()

                if feature_cols >= 50 and missing_values == 0:
                    return True, f"Created {feature_cols} features from {original_cols} columns", {
                        'original_columns': original_cols,
                        'feature_columns': feature_cols,
                        'rows': len(features),
                        'missing_values': missing_values
                    }
                elif missing_values > 0:
                    return False, f"Has {missing_values} missing values", {
                        'missing_values': missing_values
                    }
                else:
                    return False, f"Only {feature_cols} features created (expected 50+)", {
                        'feature_columns': feature_cols
                    }

            except Exception as e:
                return False, f"Failed: {str(e)}", {}

        result = await self.run_async_check("Feature Engineering", check)
        print(f"  {result.status.value} Feature Engineering: {result.message}")

    async def _check_lstm_model(self):
        """Check LSTM model training"""
        async def check():
            try:
                from ml.models.lstm_attention import LSTMAttentionModel
                from ml.feature_engineering import FeatureEngineer
                from data.yfinance_client import YFinanceClient

                print("    Training LSTM model (this may take a few minutes)...")

                client = YFinanceClient()
                data = await client.get_historical_data('AAPL', period='1y')

                fe = FeatureEngineer()
                features = fe.create_features(data)
                X, y = fe.create_sequences(features, sequence_length=60)

                epochs = 5 if self.quick_mode else 10
                model = LSTMAttentionModel(input_shape=(60, X.shape[2]))
                history = model.train(X, y, epochs=epochs, batch_size=32, validation_split=0.2)

                train_accuracy = history.history['accuracy'][-1]
                val_accuracy = history.history['val_accuracy'][-1]

                await client.close()

                # Lower threshold for quick mode
                min_accuracy = 0.50 if self.quick_mode else 0.55

                if train_accuracy >= min_accuracy:
                    return True, f"Training acc: {train_accuracy:.4f}, Val acc: {val_accuracy:.4f}", {
                        'train_accuracy': train_accuracy,
                        'val_accuracy': val_accuracy,
                        'epochs': epochs,
                        'sequences': len(X)
                    }
                else:
                    return False, f"Low accuracy: train {train_accuracy:.4f}, val {val_accuracy:.4f}", {
                        'train_accuracy': train_accuracy,
                        'val_accuracy': val_accuracy
                    }

            except Exception as e:
                return False, f"Failed: {str(e)}", {}

        result = await self.run_async_check("LSTM Model Training", check)
        print(f"  {result.status.value} LSTM Model Training: {result.message}")

    async def _check_xgboost_model(self):
        """Check XGBoost model training"""
        async def check():
            try:
                from ml.models.xgboost_model import XGBoostModel
                from ml.feature_engineering import FeatureEngineer
                from data.yfinance_client import YFinanceClient

                print("    Training XGBoost model...")

                client = YFinanceClient()
                data = await client.get_historical_data('AAPL', period='1y')

                fe = FeatureEngineer()
                features = fe.create_features(data)
                X, y = fe.prepare_supervised_data(features)

                model = XGBoostModel()
                metrics = model.train(X, y, validation_split=0.2)

                train_accuracy = metrics['train_accuracy']
                val_accuracy = metrics['val_accuracy']
                train_time = metrics['train_time']

                await client.close()

                if val_accuracy >= 0.50 and train_time < 10:
                    return True, f"Val acc: {val_accuracy:.4f}, Time: {train_time:.2f}s", {
                        'train_accuracy': train_accuracy,
                        'val_accuracy': val_accuracy,
                        'train_time': train_time,
                        'samples': len(X)
                    }
                elif train_time >= 10:
                    return False, f"Training took too long: {train_time:.2f}s", {
                        'train_time': train_time
                    }
                else:
                    return False, f"Low accuracy: {val_accuracy:.4f}", {
                        'val_accuracy': val_accuracy
                    }

            except Exception as e:
                return False, f"Failed: {str(e)}", {}

        result = await self.run_async_check("XGBoost Model Training", check)
        print(f"  {result.status.value} XGBoost Model Training: {result.message}")

    async def _check_ensemble_model(self):
        """Check ensemble model"""
        async def check():
            try:
                from ml.ensemble import EnsembleModel
                from ml.feature_engineering import FeatureEngineer
                from data.yfinance_client import YFinanceClient

                print("    Training Ensemble model...")

                client = YFinanceClient()
                data = await client.get_historical_data('AAPL', period='1y')

                fe = FeatureEngineer()
                features = fe.create_features(data)

                epochs = 5 if self.quick_mode else 10
                ensemble = EnsembleModel()
                metrics = await ensemble.train(features, epochs_lstm=epochs)

                lstm_acc = metrics['lstm_accuracy']
                xgb_acc = metrics['xgboost_accuracy']
                ens_acc = metrics['ensemble_accuracy']

                await client.close()

                # Ensemble should be at least as good as best individual model
                best_individual = max(lstm_acc, xgb_acc)
                ensemble_improvement = ens_acc >= (best_individual - 0.02)  # Allow 2% margin

                if ensemble_improvement:
                    return True, f"Ensemble: {ens_acc:.4f} (LSTM: {lstm_acc:.4f}, XGB: {xgb_acc:.4f})", {
                        'lstm_accuracy': lstm_acc,
                        'xgboost_accuracy': xgb_acc,
                        'ensemble_accuracy': ens_acc,
                        'improvement': ens_acc - best_individual
                    }
                else:
                    return False, f"Ensemble worse than best model: {ens_acc:.4f} vs {best_individual:.4f}", {
                        'ensemble_accuracy': ens_acc,
                        'best_individual': best_individual
                    }

            except Exception as e:
                return False, f"Failed: {str(e)}", {}

        result = await self.run_async_check("Ensemble Model", check)
        print(f"  {result.status.value} Ensemble Model: {result.message}")

    async def _check_model_persistence(self):
        """Check model save/load functionality"""
        async def check():
            try:
                from ml.models.xgboost_model import XGBoostModel
                from ml.feature_engineering import FeatureEngineer
                from data.yfinance_client import YFinanceClient
                import numpy as np

                client = YFinanceClient()
                data = await client.get_historical_data('AAPL', period='6mo')

                fe = FeatureEngineer()
                features = fe.create_features(data)
                X, y = fe.prepare_supervised_data(features)

                # Train and save
                model = XGBoostModel()
                model.train(X, y)

                model_path = self.project_root / 'models' / 'validation_test_model.pkl'
                model.save(str(model_path))

                # Load
                loaded_model = XGBoostModel()
                loaded_model.load(str(model_path))

                # Compare predictions
                orig_pred = model.predict(X[:5])
                loaded_pred = loaded_model.predict(X[:5])

                predictions_match = np.array_equal(orig_pred, loaded_pred)

                # Cleanup
                if model_path.exists():
                    model_path.unlink()

                await client.close()

                if predictions_match:
                    return True, "Model save/load working, predictions match", {
                        'predictions_match': True
                    }
                else:
                    return False, "Predictions don't match after load", {
                        'predictions_match': False
                    }

            except Exception as e:
                return False, f"Failed: {str(e)}", {}

        result = await self.run_async_check("Model Persistence", check)
        print(f"  {result.status.value} Model Persistence: {result.message}")
