"""
XGBoost Model for Price Prediction

Features:
- Gradient boosting classification
- Feature importance analysis
- Hyperparameter optimization support
- Model persistence
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import joblib
import logging
import json

logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    XGBoost classifier for price direction prediction

    Features:
    - Multi-class classification
    - Feature importance tracking
    - Cross-validation support
    - Model serialization
    """

    def __init__(self,
                 num_classes: int = 3,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.01,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 min_child_weight: int = 1,
                 gamma: float = 0,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 random_state: int = 42):
        """
        Initialize XGBoost model

        Args:
            num_classes: Number of output classes
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio of training data
            colsample_bytree: Subsample ratio of columns
            min_child_weight: Minimum sum of instance weight in child
            gamma: Minimum loss reduction for split
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            random_state: Random seed
        """
        self.num_classes = num_classes
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state

        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.eval_results = {}

    def _get_params(self) -> Dict:
        """Get model parameters"""
        params = {
            'objective': 'multi:softmax' if self.num_classes > 2 else 'binary:logistic',
            'num_class': self.num_classes if self.num_classes > 2 else None,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'tree_method': 'hist',  # Faster than 'auto'
            'eval_metric': ['mlogloss', 'merror'] if self.num_classes > 2 else ['logloss', 'error']
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        return params

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              feature_names: Optional[list] = None,
              early_stopping_rounds: int = 10,
              verbose: int = 50) -> Dict:
        """
        Train XGBoost model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: List of feature names
            early_stopping_rounds: Early stopping patience
            verbose: Print evaluation every N rounds

        Returns:
            Training results dictionary
        """
        logger.info(f"Training XGBoost model...")
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val) if X_val is not None else 0}")

        # Store feature names
        self.feature_names = feature_names

        # Get parameters
        params = self._get_params()

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)

        # Evaluation list
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
            evals.append((dval, 'val'))

        # Train model
        self.eval_results = {}
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds if X_val is not None else None,
            evals_result=self.eval_results,
            verbose_eval=verbose
        )

        # Get feature importance
        self.feature_importance = self.model.get_score(importance_type='gain')

        logger.info(f"Training complete! Best iteration: {self.model.best_iteration if X_val is not None else self.n_estimators}")

        return self.eval_results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions (class labels)

        Args:
            X: Input features

        Returns:
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        predictions = self.model.predict(dmatrix)

        return predictions.astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X: Input features

        Returns:
            Predicted probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Temporarily change objective to softprob
        params = self._get_params()
        params['objective'] = 'multi:softprob' if self.num_classes > 2 else 'binary:logistic'

        # Need to retrain with softprob objective for probabilities
        # For now, use a workaround with raw predictions
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        raw_predictions = self.model.predict(dmatrix, output_margin=True)

        # Apply softmax for multi-class
        if self.num_classes > 2:
            # raw_predictions shape: (n_samples, n_classes)
            exp_preds = np.exp(raw_predictions - np.max(raw_predictions, axis=1, keepdims=True))
            probabilities = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        else:
            # Binary classification: apply sigmoid
            probabilities = 1 / (1 + np.exp(-raw_predictions))
            probabilities = np.column_stack([1 - probabilities, probabilities])

        return probabilities

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            X: Input features
            y: True labels

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        predictions = self.predict(X)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(y, predictions, average='weighted', zero_division=0)
        }

        # Confusion matrix
        cm = confusion_matrix(y, predictions)
        logger.info(f"Confusion Matrix:\n{cm}")

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance scores

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet")

        # Convert to DataFrame
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in self.feature_importance.items()
        ])

        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)

        # Return top N
        return importance_df.head(top_n)

    def save(self, filepath: str):
        """
        Save model to file

        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Save model
        self.model.save_model(filepath)

        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'num_classes': self.num_classes,
            'config': self._get_params()
        }

        metadata_path = filepath.replace('.json', '_metadata.json').replace('.model', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """
        Load model from file

        Args:
            filepath: Path to load model from
        """
        # Load model
        self.model = xgb.Booster()
        self.model.load_model(filepath)

        # Load metadata
        metadata_path = filepath.replace('.json', '_metadata.json').replace('.model', '_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.feature_names = metadata.get('feature_names')
        self.feature_importance = metadata.get('feature_importance')
        self.num_classes = metadata.get('num_classes', 3)

        logger.info(f"Model loaded from {filepath}")

    def get_config(self) -> Dict:
        """Get model configuration"""
        return {
            'num_classes': self.num_classes,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'best_iteration': self.model.best_iteration if self.model and hasattr(self.model, 'best_iteration') else None
        }


def optimize_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            n_trials: int = 50) -> Dict:
    """
    Optimize XGBoost hyperparameters using Optuna

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of optimization trials

    Returns:
        Best hyperparameters
    """
    try:
        import optuna
    except ImportError:
        logger.error("Optuna not installed. Install with: pip install optuna")
        return {}

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2)
        }

        model = XGBoostModel(**params)
        model.train(X_train, y_train, X_val, y_val, verbose=0)

        metrics = model.evaluate(X_val, y_val)
        return metrics['accuracy']

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Best hyperparameters: {study.best_params}")
    logger.info(f"Best accuracy: {study.best_value:.4f}")

    return study.best_params
