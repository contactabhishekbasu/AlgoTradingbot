"""
LSTM Model with Attention Mechanism

Architecture:
- 3 LSTM layers with 128, 128, 64 units
- Multi-head attention layer
- Dense layers for classification
- Dropout for regularization
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional, Dict
import logging
import os

logger = logging.getLogger(__name__)


class LSTMWithAttention:
    """
    LSTM model with attention mechanism for price prediction

    Features:
    - Multi-layer LSTM with dropout
    - Multi-head attention mechanism
    - Classification head (3-class or binary)
    - Apple Silicon (MPS) optimization
    - Model checkpointing and versioning
    """

    def __init__(self,
                 input_shape: Tuple[int, int],
                 num_classes: int = 3,
                 lstm_units: list = [128, 128, 64],
                 attention_heads: int = 8,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Initialize LSTM model

        Args:
            input_shape: (sequence_length, num_features)
            num_classes: Number of output classes (2 or 3)
            lstm_units: List of units for each LSTM layer
            attention_heads: Number of attention heads
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.model = None
        self.history = None

        # Build model
        self._build_model()

    def _build_model(self):
        """Build LSTM model with attention"""

        # Input layer
        inputs = keras.Input(shape=self.input_shape, name='input')

        # First LSTM layer
        x = layers.LSTM(
            self.lstm_units[0],
            return_sequences=True,
            name='lstm_1'
        )(inputs)
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)

        # Second LSTM layer
        x = layers.LSTM(
            self.lstm_units[1],
            return_sequences=True,
            name='lstm_2'
        )(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_2')(x)

        # Third LSTM layer
        x = layers.LSTM(
            self.lstm_units[2],
            return_sequences=True,
            name='lstm_3'
        )(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_3')(x)

        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=self.attention_heads,
            key_dim=self.lstm_units[2] // self.attention_heads,
            name='multi_head_attention'
        )(x, x)

        # Add & Norm
        x = layers.Add(name='add_attention')([x, attention_output])
        x = layers.LayerNormalization(name='layer_norm')(x)

        # Global average pooling
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)

        # Dense layers
        x = layers.Dense(32, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate + 0.1, name='dropout_dense')(x)

        x = layers.Dense(16, activation='relu', name='dense_2')(x)

        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        )(x)

        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='lstm_attention')

        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy',
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )

        logger.info(f"Built LSTM model with {self.model.count_params():,} parameters")

    def summary(self):
        """Print model summary"""
        return self.model.summary()

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100,
              batch_size: int = 32,
              callbacks: Optional[list] = None,
              verbose: int = 1) -> Dict:
        """
        Train the model

        Args:
            X_train: Training features (samples, sequence_length, features)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: List of Keras callbacks
            verbose: Verbosity level

        Returns:
            Training history dictionary
        """
        logger.info(f"Training LSTM model for {epochs} epochs...")
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val) if X_val is not None else 0}")

        # Default callbacks
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if X_val is not None else 'loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]

        # Train model
        validation_data = (X_val, y_val) if X_val is not None else None

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        logger.info("Training complete!")

        return self.history.history

    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input features (samples, sequence_length, features)
            batch_size: Batch size for prediction

        Returns:
            Predicted probabilities
        """
        return self.model.predict(X, batch_size=batch_size, verbose=0)

    def predict_classes(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Predict class labels

        Args:
            X: Input features
            batch_size: Batch size for prediction

        Returns:
            Predicted class labels
        """
        probabilities = self.predict(X, batch_size)
        return np.argmax(probabilities, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            X: Input features
            y: True labels
            batch_size: Batch size

        Returns:
            Dictionary of metrics
        """
        results = self.model.evaluate(X, y, batch_size=batch_size, verbose=0)

        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'precision': results[2],
            'recall': results[3]
        }

        # Calculate F1 score
        precision = metrics['precision']
        recall = metrics['recall']
        if precision + recall > 0:
            metrics['f1_score'] = 2 * (precision * recall) / (precision + recall)
        else:
            metrics['f1_score'] = 0.0

        return metrics

    def save(self, filepath: str):
        """
        Save model to file

        Args:
            filepath: Path to save model
        """
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """
        Load model from file

        Args:
            filepath: Path to load model from
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")

    def get_config(self) -> Dict:
        """Get model configuration"""
        return {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'lstm_units': self.lstm_units,
            'attention_heads': self.attention_heads,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'total_parameters': self.model.count_params() if self.model else 0
        }


class LSTMModelCheckpoint(keras.callbacks.Callback):
    """
    Custom callback for model checkpointing with versioning
    """

    def __init__(self, checkpoint_dir: str, monitor: str = 'val_loss',
                 save_best_only: bool = True):
        """
        Initialize checkpoint callback

        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor
            save_best_only: Save only when metric improves
        """
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_value = np.inf if 'loss' in monitor else -np.inf

        os.makedirs(checkpoint_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """Save checkpoint at end of epoch"""
        logs = logs or {}
        current_value = logs.get(self.monitor)

        if current_value is None:
            logger.warning(f"Can't save checkpoint: {self.monitor} not available")
            return

        # Check if this is the best model
        is_best = False
        if 'loss' in self.monitor:
            is_best = current_value < self.best_value
        else:
            is_best = current_value > self.best_value

        if is_best or not self.save_best_only:
            if is_best:
                self.best_value = current_value

            # Save checkpoint
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"lstm_epoch_{epoch+1:03d}_{self.monitor}_{current_value:.4f}.h5"
            )
            self.model.save(checkpoint_path)

            if is_best:
                # Also save as "best" model
                best_path = os.path.join(self.checkpoint_dir, "lstm_best.h5")
                self.model.save(best_path)
                logger.info(f"Saved best model to {best_path} "
                          f"(epoch {epoch+1}, {self.monitor}={current_value:.4f})")
