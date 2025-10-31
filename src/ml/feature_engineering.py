"""
Feature Engineering Pipeline for ML Models

This module handles:
- Price-based features (returns, log returns)
- Technical indicators integration
- Volatility features
- Volume features
- Lag features
- Feature scaling and normalization
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for trading ML models

    Generates comprehensive features from raw OHLCV data including:
    - Price-based features
    - Technical indicators
    - Volatility metrics
    - Volume features
    - Lag features
    """

    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize feature engineer

        Args:
            scaler_type: Type of scaler ('standard' or 'robust')
        """
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else RobustScaler()
        self.feature_names = []
        self.is_fitted = False

    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added price features
        """
        df = df.copy()

        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Price changes
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open']

        # High-Low spread
        df['hl_spread'] = df['high'] - df['low']
        df['hl_spread_pct'] = (df['high'] - df['low']) / df['low']

        # Close position in range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

        # Gap features
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = df['gap'] / df['close'].shift(1)

        # Price momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1

        return df

    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volatility-based features

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with volatility features
        """
        df = df.copy()

        # Rolling volatility
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()

        # True Range (for ATR calculation)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )

        # Average True Range
        for period in [14, 20]:
            df[f'atr_{period}'] = df['tr'].rolling(period).mean()
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['close']

        # Parkinson volatility (high-low based)
        for period in [10, 20]:
            df[f'parkinson_vol_{period}'] = np.sqrt(
                (np.log(df['high'] / df['low']) ** 2).rolling(period).mean() / (4 * np.log(2))
            )

        return df

    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-based features

        Args:
            df: DataFrame with volume data

        Returns:
            DataFrame with volume features
        """
        df = df.copy()

        # Volume changes
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_change'] = df['obv'].pct_change()

        # Volume-weighted features
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['vwap_diff'] = (df['close'] - df['vwap']) / df['vwap']

        # Money Flow
        df['money_flow'] = df['close'] * df['volume']
        df['money_flow_ratio'] = df['money_flow'] / df['money_flow'].rolling(20).mean()

        return df

    def create_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10, 20]) -> pd.DataFrame:
        """
        Create lagged features

        Args:
            df: DataFrame with features
            lags: List of lag periods

        Returns:
            DataFrame with lag features
        """
        df = df.copy()

        # Key features to lag
        features_to_lag = ['returns', 'volume_change', 'close_position']

        for feature in features_to_lag:
            if feature in df.columns:
                for lag in lags:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)

        return df

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window features

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with rolling features
        """
        df = df.copy()

        # Rolling means
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'sma_{period}_diff'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']

        # Rolling max/min
        for period in [5, 10, 20]:
            df[f'max_{period}'] = df['high'].rolling(period).max()
            df[f'min_{period}'] = df['low'].rolling(period).min()
            df[f'close_to_max_{period}'] = (df['close'] - df[f'max_{period}']) / df[f'max_{period}']
            df[f'close_to_min_{period}'] = (df['close'] - df[f'min_{period}']) / df[f'min_{period}']

        return df

    def add_technical_indicators(self, df: pd.DataFrame, indicators: Dict) -> pd.DataFrame:
        """
        Add pre-calculated technical indicators

        Args:
            df: DataFrame with price data
            indicators: Dictionary of technical indicators

        Returns:
            DataFrame with indicators added
        """
        df = df.copy()

        for name, values in indicators.items():
            if isinstance(values, pd.Series):
                df[name] = values
            elif isinstance(values, dict):
                # Handle multi-value indicators (e.g., Bollinger Bands)
                for sub_name, sub_values in values.items():
                    df[f'{name}_{sub_name}'] = sub_values

        return df

    def create_target_labels(self, df: pd.DataFrame, horizon: int = 1,
                            threshold: float = 0.002) -> pd.DataFrame:
        """
        Create target labels for classification

        Args:
            df: DataFrame with price data
            horizon: Forward-looking period for prediction
            threshold: Threshold for up/down classification (0.2% default)

        Returns:
            DataFrame with target labels
        """
        df = df.copy()

        # Future returns
        df['future_returns'] = df['close'].shift(-horizon).pct_change()

        # Classification labels (3-class: up, down, neutral)
        df['target'] = 0  # neutral
        df.loc[df['future_returns'] > threshold, 'target'] = 1  # up
        df.loc[df['future_returns'] < -threshold, 'target'] = -1  # down

        # Binary classification alternative
        df['target_binary'] = (df['future_returns'] > 0).astype(int)

        return df

    def remove_highly_correlated(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        Remove highly correlated features

        Args:
            df: DataFrame with features
            threshold: Correlation threshold

        Returns:
            DataFrame with reduced features
        """
        # Calculate correlation matrix
        corr_matrix = df.corr().abs()

        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        logger.info(f"Removing {len(to_drop)} highly correlated features: {to_drop}")

        return df.drop(columns=to_drop)

    def fit_transform(self, df: pd.DataFrame, indicators: Optional[Dict] = None,
                     target_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and transform data with full feature pipeline

        Args:
            df: DataFrame with OHLCV data
            indicators: Optional pre-calculated technical indicators
            target_horizon: Horizon for target prediction

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Starting feature engineering pipeline")

        # Create all features
        df = self.create_price_features(df)
        df = self.create_volatility_features(df)
        df = self.create_volume_features(df)
        df = self.create_rolling_features(df)
        df = self.create_lag_features(df)

        # Add technical indicators if provided
        if indicators:
            df = self.add_technical_indicators(df, indicators)

        # Create target labels
        df = self.create_target_labels(df, horizon=target_horizon)

        # Remove rows with NaN values (from rolling windows and lags)
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Removed {initial_rows - len(df)} rows with NaN values")

        # Separate features and target
        target = df['target'].copy()
        feature_columns = [col for col in df.columns
                          if col not in ['target', 'target_binary', 'future_returns']]

        features = df[feature_columns].copy()

        # Remove highly correlated features
        features = self.remove_highly_correlated(features, threshold=0.95)

        # Store feature names
        self.feature_names = features.columns.tolist()

        # Fit and transform with scaler
        feature_values = self.scaler.fit_transform(features)
        features_scaled = pd.DataFrame(
            feature_values,
            index=features.index,
            columns=features.columns
        )

        self.is_fitted = True
        logger.info(f"Feature engineering complete. Created {len(self.feature_names)} features")

        return features_scaled, target

    def transform(self, df: pd.DataFrame, indicators: Optional[Dict] = None) -> pd.DataFrame:
        """
        Transform new data using fitted pipeline

        Args:
            df: DataFrame with OHLCV data
            indicators: Optional pre-calculated technical indicators

        Returns:
            Transformed features DataFrame
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")

        # Create all features (same as fit_transform)
        df = self.create_price_features(df)
        df = self.create_volatility_features(df)
        df = self.create_volume_features(df)
        df = self.create_rolling_features(df)
        df = self.create_lag_features(df)

        if indicators:
            df = self.add_technical_indicators(df, indicators)

        # Remove NaN
        df = df.dropna()

        # Select only fitted features
        features = df[self.feature_names].copy()

        # Scale features
        feature_values = self.scaler.transform(features)
        features_scaled = pd.DataFrame(
            feature_values,
            index=features.index,
            columns=features.columns
        )

        return features_scaled

    def get_feature_importance_names(self) -> List[str]:
        """Get list of feature names for importance analysis"""
        return self.feature_names.copy()


def prepare_sequences(features: pd.DataFrame, target: pd.Series,
                      sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for LSTM model

    Args:
        features: DataFrame with features
        target: Series with targets
        sequence_length: Number of time steps in each sequence

    Returns:
        Tuple of (X sequences, y targets)
    """
    X, y = [], []

    feature_values = features.values
    target_values = target.values

    for i in range(sequence_length, len(feature_values)):
        X.append(feature_values[i-sequence_length:i])
        y.append(target_values[i])

    return np.array(X), np.array(y)
