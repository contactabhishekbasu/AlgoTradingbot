"""
Unit tests for feature engineering module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.ml.feature_engineering import FeatureEngineer, prepare_sequences


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    data = pd.DataFrame({
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 102 + np.random.randn(100).cumsum(),
        'low': 98 + np.random.randn(100).cumsum(),
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)

    # Ensure high is highest and low is lowest
    data['high'] = data[['open', 'close']].max(axis=1) + np.abs(np.random.randn(100))
    data['low'] = data[['open', 'close']].min(axis=1) - np.abs(np.random.randn(100))

    return data


class TestFeatureEngineer:
    """Test FeatureEngineer class"""

    def test_initialization(self):
        """Test FeatureEngineer initialization"""
        fe = FeatureEngineer(scaler_type='standard')
        assert fe.scaler_type == 'standard'
        assert not fe.is_fitted

    def test_create_price_features(self, sample_ohlcv_data):
        """Test price feature creation"""
        fe = FeatureEngineer()
        df = fe.create_price_features(sample_ohlcv_data)

        # Check that price features are created
        assert 'returns' in df.columns
        assert 'log_returns' in df.columns
        assert 'price_change' in df.columns
        assert 'hl_spread' in df.columns
        assert 'momentum_5' in df.columns
        assert 'momentum_10' in df.columns
        assert 'momentum_20' in df.columns

        # Check that returns are calculated correctly
        expected_returns = sample_ohlcv_data['close'].pct_change()
        pd.testing.assert_series_equal(df['returns'], expected_returns, check_names=False)

    def test_create_volatility_features(self, sample_ohlcv_data):
        """Test volatility feature creation"""
        fe = FeatureEngineer()
        df = fe.create_price_features(sample_ohlcv_data)  # Need returns first
        df = fe.create_volatility_features(df)

        # Check volatility features exist
        assert 'volatility_5' in df.columns
        assert 'volatility_10' in df.columns
        assert 'atr_14' in df.columns
        assert 'tr' in df.columns

        # Check that volatility is non-negative
        assert (df['volatility_5'].dropna() >= 0).all()
        assert (df['atr_14'].dropna() >= 0).all()

    def test_create_volume_features(self, sample_ohlcv_data):
        """Test volume feature creation"""
        fe = FeatureEngineer()
        df = fe.create_price_features(sample_ohlcv_data)
        df = fe.create_volume_features(df)

        # Check volume features exist
        assert 'volume_change' in df.columns
        assert 'volume_ratio' in df.columns
        assert 'obv' in df.columns
        assert 'vwap' in df.columns

    def test_create_lag_features(self, sample_ohlcv_data):
        """Test lag feature creation"""
        fe = FeatureEngineer()
        df = fe.create_price_features(sample_ohlcv_data)
        df = fe.create_lag_features(df, lags=[1, 2, 5])

        # Check lag features exist
        assert 'returns_lag_1' in df.columns
        assert 'returns_lag_2' in df.columns
        assert 'returns_lag_5' in df.columns

        # Check lag values are correct
        assert df['returns_lag_1'].iloc[1] == df['returns'].iloc[0]
        assert df['returns_lag_2'].iloc[2] == df['returns'].iloc[0]

    def test_create_target_labels(self, sample_ohlcv_data):
        """Test target label creation"""
        fe = FeatureEngineer()
        df = fe.create_target_labels(sample_ohlcv_data, horizon=1, threshold=0.002)

        # Check target columns exist
        assert 'future_returns' in df.columns
        assert 'target' in df.columns
        assert 'target_binary' in df.columns

        # Check target values are in correct range
        assert set(df['target'].dropna().unique()).issubset({-1, 0, 1})
        assert set(df['target_binary'].dropna().unique()).issubset({0, 1})

    def test_fit_transform(self, sample_ohlcv_data):
        """Test complete fit_transform pipeline"""
        fe = FeatureEngineer()
        features, target = fe.fit_transform(sample_ohlcv_data)

        # Check that features and target are returned
        assert isinstance(features, pd.DataFrame)
        assert isinstance(target, pd.Series)

        # Check that NaN values are removed
        assert not features.isnull().any().any()
        assert not target.isnull().any()

        # Check that scaler is fitted
        assert fe.is_fitted

        # Check that feature names are stored
        assert len(fe.feature_names) > 0
        assert len(fe.feature_names) == len(features.columns)

    def test_transform_after_fit(self, sample_ohlcv_data):
        """Test transform on new data after fitting"""
        fe = FeatureEngineer()

        # Fit on first half
        train_data = sample_ohlcv_data.iloc[:50]
        fe.fit_transform(train_data)

        # Transform on second half
        test_data = sample_ohlcv_data.iloc[50:]
        features = fe.transform(test_data)

        # Check that features have same columns
        assert list(features.columns) == fe.feature_names

    def test_transform_without_fit_raises_error(self, sample_ohlcv_data):
        """Test that transform without fit raises error"""
        fe = FeatureEngineer()

        with pytest.raises(ValueError, match="must be fitted"):
            fe.transform(sample_ohlcv_data)


class TestPrepareSequences:
    """Test prepare_sequences function"""

    def test_prepare_sequences_shape(self):
        """Test sequence preparation shapes"""
        # Create sample features and target
        features = pd.DataFrame(np.random.randn(100, 10))
        target = pd.Series(np.random.randint(0, 3, 100))

        X, y = prepare_sequences(features, target, sequence_length=60)

        # Check shapes
        assert X.shape == (40, 60, 10)  # 100 - 60 = 40 sequences
        assert y.shape == (40,)

    def test_prepare_sequences_values(self):
        """Test that sequences contain correct values"""
        features = pd.DataFrame(np.arange(100).reshape(-1, 1))
        target = pd.Series(np.arange(100))

        X, y = prepare_sequences(features, target, sequence_length=10)

        # Check first sequence
        assert np.array_equal(X[0], np.arange(10).reshape(-1, 1))
        assert y[0] == 10

        # Check last sequence
        assert np.array_equal(X[-1], np.arange(90, 100).reshape(-1, 1))
        assert y[-1] == 99

    def test_prepare_sequences_different_lengths(self):
        """Test sequence preparation with different sequence lengths"""
        features = pd.DataFrame(np.random.randn(100, 5))
        target = pd.Series(np.random.randint(0, 3, 100))

        for seq_len in [10, 30, 50]:
            X, y = prepare_sequences(features, target, sequence_length=seq_len)
            assert X.shape == (100 - seq_len, seq_len, 5)
            assert y.shape == (100 - seq_len,)
