"""
Model Training Pipeline

Orchestrates the complete training workflow:
- Data loading and preprocessing
- Feature engineering
- Model training (LSTM, XGBoost)
- Ensemble creation
- Model evaluation and persistence
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import logging
import os
import json
from datetime import datetime

from ml.feature_engineering import FeatureEngineer, prepare_sequences
from ml.dataset import DatasetPreparator, DataLoader
from ml.models.lstm_attention import LSTMWithAttention, LSTMModelCheckpoint
from ml.models.xgboost_model import XGBoostModel
from ml.ensemble import EnsemblePredictor

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Complete ML model training pipeline

    Features:
    - End-to-end training workflow
    - Multiple model support
    - Ensemble creation
    - Model versioning and persistence
    - Comprehensive evaluation
    """

    def __init__(self,
                 output_dir: str = './models',
                 cache_dir: str = './data/cache',
                 random_state: int = 42):
        """
        Initialize model trainer

        Args:
            output_dir: Directory to save trained models
            cache_dir: Directory for caching processed data
            random_state: Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.random_state = random_state

        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

        # Components
        self.feature_engineer = None
        self.dataset_prep = DatasetPreparator()
        self.data_loader = DataLoader(cache_dir=cache_dir)

        # Models
        self.lstm_model = None
        self.xgboost_model = None
        self.ensemble = None

        # Training results
        self.training_history = {}

        # Set random seeds
        np.random.seed(random_state)

    def prepare_data(self,
                    symbol: str,
                    start_date: str,
                    end_date: str,
                    target_horizon: int = 1) -> Tuple:
        """
        Prepare data for training

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            target_horizon: Prediction horizon in days

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        logger.info(f"Preparing data for {symbol} from {start_date} to {end_date}")

        # Load market data
        market_data = self.data_loader.load_market_data(symbol, start_date, end_date)

        logger.info(f"Loaded {len(market_data)} rows of market data")

        # Feature engineering
        self.feature_engineer = FeatureEngineer(scaler_type='standard')
        features, target = self.feature_engineer.fit_transform(
            market_data,
            target_horizon=target_horizon
        )

        logger.info(f"Created {len(features.columns)} features")

        # Check data quality
        quality_report = self.dataset_prep.check_data_quality(features, target)
        logger.info(f"Data quality report: {quality_report}")

        # Split data
        train_data, val_data, test_data = self.dataset_prep.time_series_split(features, target)

        return train_data, val_data, test_data

    def train_lstm(self,
                  train_data: Tuple,
                  val_data: Tuple,
                  sequence_length: int = 60,
                  epochs: int = 100,
                  batch_size: int = 32) -> LSTMWithAttention:
        """
        Train LSTM model

        Args:
            train_data: (X_train, y_train)
            val_data: (X_val, y_val)
            sequence_length: Number of time steps
            epochs: Training epochs
            batch_size: Batch size

        Returns:
            Trained LSTM model
        """
        logger.info("Training LSTM model...")

        X_train, y_train = train_data
        X_val, y_val = val_data

        # Prepare sequences for LSTM
        X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, sequence_length)
        X_val_seq, y_val_seq = prepare_sequences(X_val, y_val, sequence_length)

        logger.info(f"Sequence shapes - Train: {X_train_seq.shape}, Val: {X_val_seq.shape}")

        # Create LSTM model
        input_shape = (sequence_length, X_train.shape[1])
        num_classes = len(np.unique(y_train))

        self.lstm_model = LSTMWithAttention(
            input_shape=input_shape,
            num_classes=num_classes,
            lstm_units=[128, 128, 64],
            attention_heads=8,
            dropout_rate=0.2,
            learning_rate=0.001
        )

        # Print model summary
        logger.info(f"\nLSTM Model Architecture:")
        self.lstm_model.summary()

        # Setup callbacks
        checkpoint_dir = os.path.join(self.output_dir, 'lstm_checkpoints')
        callbacks = [
            LSTMModelCheckpoint(checkpoint_dir, monitor='val_loss', save_best_only=True)
        ]

        # Train model
        history = self.lstm_model.train(
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.training_history['lstm'] = history

        # Evaluate
        metrics = self.lstm_model.evaluate(X_val_seq, y_val_seq)
        logger.info(f"LSTM Validation Metrics: {metrics}")

        # Save model
        model_path = os.path.join(self.output_dir, 'lstm_model.h5')
        self.lstm_model.save(model_path)

        return self.lstm_model

    def train_xgboost(self,
                     train_data: Tuple,
                     val_data: Tuple,
                     n_estimators: int = 100) -> XGBoostModel:
        """
        Train XGBoost model

        Args:
            train_data: (X_train, y_train)
            val_data: (X_val, y_val)
            n_estimators: Number of trees

        Returns:
            Trained XGBoost model
        """
        logger.info("Training XGBoost model...")

        X_train, y_train = train_data
        X_val, y_val = val_data

        # Encode labels (XGBoost expects 0, 1, 2 instead of -1, 0, 1)
        y_train_encoded, label_map = self.dataset_prep.encode_labels(y_train)
        y_val_encoded, _ = self.dataset_prep.encode_labels(y_val)

        num_classes = len(np.unique(y_train))

        # Create XGBoost model
        self.xgboost_model = XGBoostModel(
            num_classes=num_classes,
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state
        )

        # Train model
        history = self.xgboost_model.train(
            X_train.values, y_train_encoded,
            X_val.values, y_val_encoded,
            feature_names=X_train.columns.tolist(),
            early_stopping_rounds=10,
            verbose=50
        )

        self.training_history['xgboost'] = history

        # Evaluate
        metrics = self.xgboost_model.evaluate(X_val.values, y_val_encoded)
        logger.info(f"XGBoost Validation Metrics: {metrics}")

        # Feature importance
        importance_df = self.xgboost_model.get_feature_importance(top_n=20)
        logger.info(f"\nTop 20 Important Features:\n{importance_df}")

        # Save model
        model_path = os.path.join(self.output_dir, 'xgboost_model.json')
        self.xgboost_model.save(model_path)

        return self.xgboost_model

    def create_ensemble(self,
                       train_data: Tuple,
                       val_data: Tuple,
                       sequence_length: int = 60) -> EnsemblePredictor:
        """
        Create ensemble from trained models

        Args:
            train_data: Training data
            val_data: Validation data
            sequence_length: Sequence length for LSTM

        Returns:
            Ensemble predictor
        """
        logger.info("Creating ensemble...")

        if self.lstm_model is None or self.xgboost_model is None:
            raise ValueError("Both LSTM and XGBoost models must be trained first")

        X_val, y_val = val_data

        # Prepare sequences for LSTM
        X_val_seq, y_val_seq = prepare_sequences(X_val, y_val, sequence_length)

        # Encode labels for XGBoost
        y_val_encoded, _ = self.dataset_prep.encode_labels(y_val_seq)

        # Evaluate individual models on same data
        lstm_metrics = self.lstm_model.evaluate(X_val_seq, y_val_seq)
        xgboost_metrics = self.xgboost_model.evaluate(X_val.values[-len(y_val_seq):], y_val_encoded)

        logger.info(f"LSTM accuracy: {lstm_metrics['accuracy']:.4f}")
        logger.info(f"XGBoost accuracy: {xgboost_metrics['accuracy']:.4f}")

        # Set initial weights based on validation accuracy
        total_accuracy = lstm_metrics['accuracy'] + xgboost_metrics['accuracy']
        initial_weights = {
            'lstm': lstm_metrics['accuracy'] / total_accuracy,
            'xgboost': xgboost_metrics['accuracy'] / total_accuracy
        }

        logger.info(f"Initial ensemble weights: {initial_weights}")

        # Create wrapper for consistent interface
        class LSTMWrapper:
            def __init__(self, model, seq_length, feature_engineer):
                self.model = model
                self.seq_length = seq_length
                self.feature_engineer = feature_engineer

            def predict_proba(self, X):
                # Need to prepare sequences
                # For simplicity, assume X is already in correct shape
                # In production, would need to handle sequence preparation
                return self.model.predict(X)

        class XGBoostWrapper:
            def __init__(self, model):
                self.model = model

            def predict_proba(self, X):
                return self.model.predict_proba(X)

        models = {
            'lstm': LSTMWrapper(self.lstm_model, sequence_length, self.feature_engineer),
            'xgboost': XGBoostWrapper(self.xgboost_model)
        }

        self.ensemble = EnsemblePredictor(
            models=models,
            initial_weights=initial_weights,
            adaptive=True
        )

        # Save ensemble config
        config_path = os.path.join(self.output_dir, 'ensemble_config.json')
        self.ensemble.save_weights(config_path)

        return self.ensemble

    def full_pipeline(self,
                     symbol: str,
                     start_date: str,
                     end_date: str,
                     lstm_epochs: int = 100,
                     xgboost_estimators: int = 100,
                     sequence_length: int = 60) -> Dict:
        """
        Run complete training pipeline

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            lstm_epochs: LSTM training epochs
            xgboost_estimators: XGBoost n_estimators
            sequence_length: Sequence length for LSTM

        Returns:
            Dictionary with training results
        """
        start_time = datetime.now()

        logger.info("="*80)
        logger.info(f"Starting full training pipeline for {symbol}")
        logger.info("="*80)

        # 1. Prepare data
        train_data, val_data, test_data = self.prepare_data(symbol, start_date, end_date)

        # 2. Train LSTM
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: Training LSTM Model")
        logger.info("="*80)
        self.train_lstm(train_data, val_data, sequence_length, lstm_epochs)

        # 3. Train XGBoost
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: Training XGBoost Model")
        logger.info("="*80)
        self.train_xgboost(train_data, val_data, xgboost_estimators)

        # 4. Create ensemble
        logger.info("\n" + "="*80)
        logger.info("PHASE 3: Creating Ensemble")
        logger.info("="*80)
        self.create_ensemble(train_data, val_data, sequence_length)

        # 5. Evaluate on test set
        logger.info("\n" + "="*80)
        logger.info("PHASE 4: Final Evaluation on Test Set")
        logger.info("="*80)

        X_test, y_test = test_data

        # Test LSTM
        X_test_seq, y_test_seq = prepare_sequences(X_test, y_test, sequence_length)
        lstm_test_metrics = self.lstm_model.evaluate(X_test_seq, y_test_seq)
        logger.info(f"LSTM Test Metrics: {lstm_test_metrics}")

        # Test XGBoost
        y_test_encoded, _ = self.dataset_prep.encode_labels(y_test)
        xgboost_test_metrics = self.xgboost_model.evaluate(
            X_test.values, y_test_encoded
        )
        logger.info(f"XGBoost Test Metrics: {xgboost_test_metrics}")

        # Training summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        results = {
            'symbol': symbol,
            'date_range': (start_date, end_date),
            'training_duration_seconds': duration,
            'lstm_test_metrics': lstm_test_metrics,
            'xgboost_test_metrics': xgboost_test_metrics,
            'ensemble_config': self.ensemble.get_config(),
            'feature_count': len(self.feature_engineer.feature_names),
            'training_samples': len(train_data[0]),
            'validation_samples': len(val_data[0]),
            'test_samples': len(test_data[0])
        }

        # Save results
        results_path = os.path.join(self.output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("\n" + "="*80)
        logger.info("Training Pipeline Complete!")
        logger.info("="*80)
        logger.info(f"Total duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info(f"Results saved to: {results_path}")

        return results
