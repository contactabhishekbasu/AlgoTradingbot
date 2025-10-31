"""
Ensemble Model for Combining LSTM and XGBoost Predictions

Features:
- Weighted ensemble prediction
- Adaptive weight adjustment based on performance
- Confidence scoring
- Model performance tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from collections import deque
import json

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple models

    Features:
    - Weighted average of model predictions
    - Dynamic weight adjustment
    - Confidence scoring
    - Performance-based model selection
    """

    def __init__(self,
                 models: Dict[str, any],
                 initial_weights: Optional[Dict[str, float]] = None,
                 adaptive: bool = True,
                 performance_window: int = 100):
        """
        Initialize ensemble predictor

        Args:
            models: Dictionary of {model_name: model_object}
            initial_weights: Initial weights for each model (default: equal)
            adaptive: Whether to adapt weights based on performance
            performance_window: Window size for performance tracking
        """
        self.models = models
        self.adaptive = adaptive
        self.performance_window = performance_window

        # Initialize weights
        if initial_weights is None:
            # Equal weights
            n_models = len(models)
            self.weights = {name: 1.0 / n_models for name in models.keys()}
        else:
            # Normalize provided weights
            total = sum(initial_weights.values())
            self.weights = {k: v / total for k, v in initial_weights.items()}

        # Performance tracking
        self.performance_history = {
            name: deque(maxlen=performance_window)
            for name in models.keys()
        }

        self.prediction_count = 0

        logger.info(f"Initialized ensemble with {len(models)} models: {list(models.keys())}")
        logger.info(f"Initial weights: {self.weights}")

    def predict(self, X: np.ndarray, return_individual: bool = False) -> np.ndarray:
        """
        Make ensemble predictions

        Args:
            X: Input features
            return_individual: Whether to return individual model predictions

        Returns:
            Ensemble predictions (and optionally individual predictions)
        """
        individual_predictions = {}

        # Get predictions from each model
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                # Get probabilities
                proba = model.predict_proba(X)
            elif hasattr(model, 'predict'):
                # XGBoost or other models
                proba = model.predict_proba(X)
            else:
                raise ValueError(f"Model {name} doesn't have predict or predict_proba method")

            individual_predictions[name] = proba

        # Weighted average
        ensemble_proba = np.zeros_like(list(individual_predictions.values())[0])

        for name, proba in individual_predictions.items():
            ensemble_proba += self.weights[name] * proba

        self.prediction_count += len(X)

        if return_individual:
            return ensemble_proba, individual_predictions
        else:
            return ensemble_proba

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels

        Args:
            X: Input features

        Returns:
            Predicted class labels
        """
        probabilities = self.predict(X)
        return np.argmax(probabilities, axis=1)

    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence scores

        Args:
            X: Input features

        Returns:
            Tuple of (predictions, confidence_scores)
        """
        probabilities = self.predict(X)
        predictions = np.argmax(probabilities, axis=1)

        # Confidence is the maximum probability
        confidence = np.max(probabilities, axis=1)

        return predictions, confidence

    def update_performance(self, predictions: np.ndarray, actuals: np.ndarray,
                          model_predictions: Dict[str, np.ndarray]):
        """
        Update performance history for each model

        Args:
            predictions: Ensemble predictions
            actuals: True labels
            model_predictions: Individual model predictions
        """
        # Calculate accuracy for each model
        for name, preds in model_predictions.items():
            pred_classes = np.argmax(preds, axis=1)
            accuracy = np.mean(pred_classes == actuals)
            self.performance_history[name].append(accuracy)

        # Update weights if adaptive
        if self.adaptive:
            self._update_weights()

    def _update_weights(self):
        """Update model weights based on recent performance"""
        if all(len(history) > 0 for history in self.performance_history.values()):
            # Calculate average performance over recent window
            avg_performance = {
                name: np.mean(list(history))
                for name, history in self.performance_history.items()
            }

            # Convert to weights using softmax
            performances = np.array(list(avg_performance.values()))

            # Apply softmax with temperature
            temperature = 2.0  # Higher temperature = more uniform weights
            exp_perf = np.exp(performances / temperature)
            new_weights = exp_perf / np.sum(exp_perf)

            # Update weights
            for i, name in enumerate(avg_performance.keys()):
                self.weights[name] = new_weights[i]

            logger.info(f"Updated weights: {self.weights}")
            logger.info(f"Based on performance: {avg_performance}")

    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance statistics for each model

        Returns:
            Dictionary of performance metrics per model
        """
        performance = {}

        for name, history in self.performance_history.items():
            if len(history) > 0:
                performance[name] = {
                    'mean_accuracy': np.mean(list(history)),
                    'std_accuracy': np.std(list(history)),
                    'current_weight': self.weights[name],
                    'samples': len(history)
                }
            else:
                performance[name] = {
                    'mean_accuracy': 0.0,
                    'std_accuracy': 0.0,
                    'current_weight': self.weights[name],
                    'samples': 0
                }

        return performance

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate ensemble performance

        Args:
            X: Input features
            y: True labels

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        # Get predictions
        predictions = self.predict_classes(X)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(y, predictions, average='weighted', zero_division=0)
        }

        # Get individual model performance
        ensemble_proba, individual_preds = self.predict(X, return_individual=True)

        individual_metrics = {}
        for name, proba in individual_preds.items():
            pred_classes = np.argmax(proba, axis=1)
            individual_metrics[f'{name}_accuracy'] = accuracy_score(y, pred_classes)

        metrics.update(individual_metrics)

        return metrics

    def save_weights(self, filepath: str):
        """
        Save current weights and performance history

        Args:
            filepath: Path to save weights
        """
        data = {
            'weights': self.weights,
            'performance_history': {
                name: list(history)
                for name, history in self.performance_history.items()
            },
            'prediction_count': self.prediction_count,
            'adaptive': self.adaptive
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved ensemble weights to {filepath}")

    def load_weights(self, filepath: str):
        """
        Load weights and performance history

        Args:
            filepath: Path to load weights from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.weights = data['weights']
        self.prediction_count = data['prediction_count']
        self.adaptive = data['adaptive']

        # Load performance history
        for name, history in data['performance_history'].items():
            self.performance_history[name] = deque(history, maxlen=self.performance_window)

        logger.info(f"Loaded ensemble weights from {filepath}")
        logger.info(f"Loaded weights: {self.weights}")

    def get_config(self) -> Dict:
        """Get ensemble configuration"""
        return {
            'models': list(self.models.keys()),
            'weights': self.weights,
            'adaptive': self.adaptive,
            'performance_window': self.performance_window,
            'prediction_count': self.prediction_count
        }


class ModelSelector:
    """
    Dynamically select best model based on recent performance
    """

    def __init__(self, models: Dict[str, any], window_size: int = 50):
        """
        Initialize model selector

        Args:
            models: Dictionary of models
            window_size: Window for performance evaluation
        """
        self.models = models
        self.window_size = window_size
        self.performance_history = {
            name: deque(maxlen=window_size)
            for name in models.keys()
        }
        self.active_model = list(models.keys())[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using currently active model

        Args:
            X: Input features

        Returns:
            Predictions
        """
        model = self.models[self.active_model]

        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            return model.predict(X)

    def update_performance(self, predictions: np.ndarray, actuals: np.ndarray):
        """
        Update performance and potentially switch model

        Args:
            predictions: Model predictions
            actuals: True labels
        """
        # Calculate accuracy
        pred_classes = np.argmax(predictions, axis=1) if predictions.ndim > 1 else predictions
        accuracy = np.mean(pred_classes == actuals)

        self.performance_history[self.active_model].append(accuracy)

        # Check if should switch model
        if len(self.performance_history[self.active_model]) >= self.window_size:
            self._select_best_model()

    def _select_best_model(self):
        """Select model with best recent performance"""
        avg_performance = {
            name: np.mean(list(history)) if len(history) > 0 else 0
            for name, history in self.performance_history.items()
        }

        best_model = max(avg_performance, key=avg_performance.get)

        if best_model != self.active_model:
            logger.info(f"Switching from {self.active_model} to {best_model}")
            logger.info(f"Performance: {avg_performance}")
            self.active_model = best_model

    def get_active_model(self) -> str:
        """Get currently active model name"""
        return self.active_model
