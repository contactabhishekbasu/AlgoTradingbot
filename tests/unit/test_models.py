"""
Unit tests for ML models
"""

import pytest
import numpy as np
import tempfile
import os

from src.ml.models.lstm_attention import LSTMWithAttention
from src.ml.models.xgboost_model import XGBoostModel
from src.ml.ensemble import EnsemblePredictor


@pytest.fixture
def sample_lstm_data():
    """Create sample data for LSTM testing"""
    np.random.seed(42)
    X_train = np.random.randn(100, 60, 10)  # 100 samples, 60 timesteps, 10 features
    y_train = np.random.randint(0, 3, 100)
    X_val = np.random.randn(20, 60, 10)
    y_val = np.random.randint(0, 3, 20)
    return X_train, y_train, X_val, y_val


@pytest.fixture
def sample_xgboost_data():
    """Create sample data for XGBoost testing"""
    np.random.seed(42)
    X_train = np.random.randn(100, 50)
    y_train = np.random.randint(0, 3, 100)
    X_val = np.random.randn(20, 50)
    y_val = np.random.randint(0, 3, 20)
    return X_train, y_train, X_val, y_val


class TestLSTMWithAttention:
    """Test LSTM model"""

    def test_initialization(self):
        """Test LSTM model initialization"""
        model = LSTMWithAttention(
            input_shape=(60, 10),
            num_classes=3
        )

        assert model.input_shape == (60, 10)
        assert model.num_classes == 3
        assert model.model is not None

    def test_model_architecture(self):
        """Test model has expected layers"""
        model = LSTMWithAttention(input_shape=(60, 10), num_classes=3)

        # Check layer names
        layer_names = [layer.name for layer in model.model.layers]
        assert 'lstm_1' in layer_names
        assert 'lstm_2' in layer_names
        assert 'lstm_3' in layer_names
        assert 'multi_head_attention' in layer_names
        assert 'output' in layer_names

    def test_training(self, sample_lstm_data):
        """Test model training"""
        X_train, y_train, X_val, y_val = sample_lstm_data

        model = LSTMWithAttention(input_shape=(60, 10), num_classes=3)

        # Train for just 2 epochs
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=2,
            batch_size=16,
            verbose=0
        )

        # Check training history exists
        assert 'loss' in history
        assert 'accuracy' in history
        assert len(history['loss']) <= 2

    def test_prediction(self, sample_lstm_data):
        """Test model prediction"""
        X_train, y_train, X_val, y_val = sample_lstm_data

        model = LSTMWithAttention(input_shape=(60, 10), num_classes=3)
        model.train(X_train, y_train, epochs=1, verbose=0)

        # Test prediction
        predictions = model.predict(X_val[:5])
        assert predictions.shape == (5, 3)
        assert np.allclose(predictions.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_predict_classes(self, sample_lstm_data):
        """Test class prediction"""
        X_train, y_train, X_val, y_val = sample_lstm_data

        model = LSTMWithAttention(input_shape=(60, 10), num_classes=3)
        model.train(X_train, y_train, epochs=1, verbose=0)

        # Test class prediction
        classes = model.predict_classes(X_val[:5])
        assert classes.shape == (5,)
        assert set(classes).issubset({0, 1, 2})

    def test_evaluate(self, sample_lstm_data):
        """Test model evaluation"""
        X_train, y_train, X_val, y_val = sample_lstm_data

        model = LSTMWithAttention(input_shape=(60, 10), num_classes=3)
        model.train(X_train, y_train, epochs=1, verbose=0)

        # Evaluate
        metrics = model.evaluate(X_val, y_val)
        assert 'accuracy' in metrics
        assert 'loss' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics

    def test_save_and_load(self, sample_lstm_data):
        """Test model save and load"""
        X_train, y_train, _, _ = sample_lstm_data

        model = LSTMWithAttention(input_shape=(60, 10), num_classes=3)
        model.train(X_train, y_train, epochs=1, verbose=0)

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.h5')
            model.save(model_path)

            # Load model
            loaded_model = LSTMWithAttention(input_shape=(60, 10), num_classes=3)
            loaded_model.load(model_path)

            # Test predictions are same
            X_test = X_train[:5]
            original_pred = model.predict(X_test)
            loaded_pred = loaded_model.predict(X_test)

            np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=5)


class TestXGBoostModel:
    """Test XGBoost model"""

    def test_initialization(self):
        """Test XGBoost model initialization"""
        model = XGBoostModel(
            num_classes=3,
            n_estimators=10,
            max_depth=3
        )

        assert model.num_classes == 3
        assert model.n_estimators == 10
        assert model.max_depth == 3

    def test_training(self, sample_xgboost_data):
        """Test model training"""
        X_train, y_train, X_val, y_val = sample_xgboost_data

        model = XGBoostModel(num_classes=3, n_estimators=10)

        results = model.train(
            X_train, y_train,
            X_val, y_val,
            verbose=0
        )

        assert 'train' in results
        assert model.model is not None

    def test_prediction(self, sample_xgboost_data):
        """Test model prediction"""
        X_train, y_train, X_val, y_val = sample_xgboost_data

        model = XGBoostModel(num_classes=3, n_estimators=10)
        model.train(X_train, y_train, verbose=0)

        # Test prediction
        predictions = model.predict(X_val[:5])
        assert predictions.shape == (5,)
        assert set(predictions).issubset({0, 1, 2})

    def test_predict_proba(self, sample_xgboost_data):
        """Test probability prediction"""
        X_train, y_train, X_val, y_val = sample_xgboost_data

        model = XGBoostModel(num_classes=3, n_estimators=10)
        model.train(X_train, y_train, verbose=0)

        # Test probability prediction
        probabilities = model.predict_proba(X_val[:5])
        assert probabilities.shape == (5, 3)
        # Probabilities should sum to approximately 1
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=0.01)

    def test_evaluate(self, sample_xgboost_data):
        """Test model evaluation"""
        X_train, y_train, X_val, y_val = sample_xgboost_data

        model = XGBoostModel(num_classes=3, n_estimators=10)
        model.train(X_train, y_train, verbose=0)

        metrics = model.evaluate(X_val, y_val)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics

    def test_feature_importance(self, sample_xgboost_data):
        """Test feature importance extraction"""
        X_train, y_train, _, _ = sample_xgboost_data

        feature_names = [f'feature_{i}' for i in range(50)]

        model = XGBoostModel(num_classes=3, n_estimators=10)
        model.train(X_train, y_train, feature_names=feature_names, verbose=0)

        importance_df = model.get_feature_importance(top_n=10)
        assert len(importance_df) <= 10
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns

    def test_save_and_load(self, sample_xgboost_data):
        """Test model save and load"""
        X_train, y_train, _, _ = sample_xgboost_data

        model = XGBoostModel(num_classes=3, n_estimators=10)
        model.train(X_train, y_train, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.json')
            model.save(model_path)

            # Load model
            loaded_model = XGBoostModel(num_classes=3)
            loaded_model.load(model_path)

            # Test predictions are same
            X_test = X_train[:5]
            original_pred = model.predict(X_test)
            loaded_pred = loaded_model.predict(X_test)

            np.testing.assert_array_equal(original_pred, loaded_pred)


class TestEnsemblePredictor:
    """Test ensemble predictor"""

    @pytest.fixture
    def trained_models(self, sample_xgboost_data):
        """Create trained models for ensemble testing"""
        X_train, y_train, _, _ = sample_xgboost_data

        # Create two simple models
        model1 = XGBoostModel(num_classes=3, n_estimators=5, random_state=42)
        model1.train(X_train, y_train, verbose=0)

        model2 = XGBoostModel(num_classes=3, n_estimators=5, random_state=43)
        model2.train(X_train, y_train, verbose=0)

        return {'model1': model1, 'model2': model2}

    def test_initialization(self, trained_models):
        """Test ensemble initialization"""
        ensemble = EnsemblePredictor(trained_models)

        assert len(ensemble.models) == 2
        assert 'model1' in ensemble.weights
        assert 'model2' in ensemble.weights
        # Weights should sum to 1
        assert np.isclose(sum(ensemble.weights.values()), 1.0)

    def test_prediction(self, trained_models, sample_xgboost_data):
        """Test ensemble prediction"""
        _, _, X_val, _ = sample_xgboost_data

        ensemble = EnsemblePredictor(trained_models)
        predictions = ensemble.predict(X_val[:5])

        assert predictions.shape == (5, 3)
        assert np.allclose(predictions.sum(axis=1), 1.0)

    def test_predict_classes(self, trained_models, sample_xgboost_data):
        """Test ensemble class prediction"""
        _, _, X_val, _ = sample_xgboost_data

        ensemble = EnsemblePredictor(trained_models)
        classes = ensemble.predict_classes(X_val[:5])

        assert classes.shape == (5,)
        assert set(classes).issubset({0, 1, 2})

    def test_predict_with_confidence(self, trained_models, sample_xgboost_data):
        """Test prediction with confidence scores"""
        _, _, X_val, _ = sample_xgboost_data

        ensemble = EnsemblePredictor(trained_models)
        predictions, confidence = ensemble.predict_with_confidence(X_val[:5])

        assert predictions.shape == (5,)
        assert confidence.shape == (5,)
        assert (confidence >= 0).all() and (confidence <= 1).all()

    def test_evaluate(self, trained_models, sample_xgboost_data):
        """Test ensemble evaluation"""
        _, _, X_val, y_val = sample_xgboost_data

        ensemble = EnsemblePredictor(trained_models)
        metrics = ensemble.evaluate(X_val, y_val)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics

    def test_custom_weights(self, trained_models):
        """Test ensemble with custom weights"""
        custom_weights = {'model1': 0.7, 'model2': 0.3}
        ensemble = EnsemblePredictor(trained_models, initial_weights=custom_weights)

        # Weights should be normalized
        assert np.isclose(ensemble.weights['model1'], 0.7)
        assert np.isclose(ensemble.weights['model2'], 0.3)

    def test_save_and_load_weights(self, trained_models):
        """Test saving and loading ensemble weights"""
        ensemble = EnsemblePredictor(trained_models)

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, 'ensemble_weights.json')
            ensemble.save_weights(weights_path)

            # Create new ensemble and load weights
            new_ensemble = EnsemblePredictor(trained_models)
            new_ensemble.load_weights(weights_path)

            assert ensemble.weights == new_ensemble.weights
