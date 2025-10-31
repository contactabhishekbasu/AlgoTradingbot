"""
Dataset Preparation Module

Handles:
- Train/validation/test splits
- Time series cross-validation
- Data loading and caching
- Label encoding
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
import logging

logger = logging.getLogger(__name__)


class DatasetPreparator:
    """
    Prepare datasets for ML model training and evaluation

    Features:
    - Time-aware train/val/test splits
    - Cross-validation support
    - Label encoding
    - Data quality checks
    """

    def __init__(self, train_ratio: float = 0.6, val_ratio: float = 0.2,
                 test_ratio: float = 0.2):
        """
        Initialize dataset preparator

        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def time_series_split(self, features: pd.DataFrame, target: pd.Series
                         ) -> Tuple[Tuple[pd.DataFrame, pd.Series],
                                   Tuple[pd.DataFrame, pd.Series],
                                   Tuple[pd.DataFrame, pd.Series]]:
        """
        Split data into train/validation/test sets preserving time order

        Args:
            features: Feature DataFrame with datetime index
            target: Target Series with datetime index

        Returns:
            Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        n_samples = len(features)

        # Calculate split indices
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))

        # Split features
        X_train = features.iloc[:train_end]
        X_val = features.iloc[train_end:val_end]
        X_test = features.iloc[val_end:]

        # Split target
        y_train = target.iloc[:train_end]
        y_val = target.iloc[train_end:val_end]
        y_test = target.iloc[val_end:]

        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Train period: {X_train.index[0]} to {X_train.index[-1]}")
        logger.info(f"Val period: {X_val.index[0]} to {X_val.index[-1]}")
        logger.info(f"Test period: {X_test.index[0]} to {X_test.index[-1]}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def create_cv_splits(self, features: pd.DataFrame, target: pd.Series,
                        n_splits: int = 5) -> TimeSeriesSplit:
        """
        Create time series cross-validation splits

        Args:
            features: Feature DataFrame
            target: Target Series
            n_splits: Number of CV splits

        Returns:
            TimeSeriesSplit object
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)

        logger.info(f"Created {n_splits} time series CV splits")

        return tscv

    def encode_labels(self, target: pd.Series, num_classes: int = 3
                     ) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Encode target labels for classification

        Args:
            target: Target Series with values [-1, 0, 1] or [0, 1]
            num_classes: Number of classes (2 or 3)

        Returns:
            Tuple of (encoded labels, label mapping)
        """
        if num_classes == 3:
            # Map -1, 0, 1 to 0, 1, 2
            label_map = {-1: 0, 0: 1, 1: 2}
            reverse_map = {0: 'down', 1: 'neutral', 2: 'up'}
        else:
            # Binary classification
            label_map = {0: 0, 1: 1}
            reverse_map = {0: 'down', 1: 'up'}

        encoded = target.map(label_map).values

        logger.info(f"Encoded labels for {num_classes}-class classification")
        logger.info(f"Class distribution: {pd.Series(encoded).value_counts().to_dict()}")

        return encoded, reverse_map

    def one_hot_encode(self, target: pd.Series, num_classes: int = 3) -> np.ndarray:
        """
        One-hot encode target labels

        Args:
            target: Target Series
            num_classes: Number of classes

        Returns:
            One-hot encoded array
        """
        encoded, _ = self.encode_labels(target, num_classes)

        # One-hot encode
        one_hot = np.zeros((len(encoded), num_classes))
        one_hot[np.arange(len(encoded)), encoded] = 1

        return one_hot

    def check_data_quality(self, features: pd.DataFrame, target: pd.Series
                          ) -> Dict[str, any]:
        """
        Perform data quality checks

        Args:
            features: Feature DataFrame
            target: Target Series

        Returns:
            Dictionary with quality metrics
        """
        quality_report = {
            'total_samples': len(features),
            'feature_count': len(features.columns),
            'missing_values': features.isnull().sum().sum(),
            'inf_values': np.isinf(features).sum().sum(),
            'duplicate_rows': features.duplicated().sum(),
            'target_distribution': target.value_counts().to_dict(),
            'date_range': (features.index.min(), features.index.max()),
            'feature_dtypes': features.dtypes.value_counts().to_dict()
        }

        # Check for class imbalance
        class_counts = target.value_counts()
        if len(class_counts) > 1:
            imbalance_ratio = class_counts.max() / class_counts.min()
            quality_report['class_imbalance_ratio'] = imbalance_ratio

            if imbalance_ratio > 3:
                logger.warning(f"Severe class imbalance detected: {imbalance_ratio:.2f}")

        # Check for data issues
        if quality_report['missing_values'] > 0:
            logger.warning(f"Found {quality_report['missing_values']} missing values")

        if quality_report['inf_values'] > 0:
            logger.warning(f"Found {quality_report['inf_values']} infinite values")

        logger.info(f"Data quality check complete: {quality_report['total_samples']} samples, "
                   f"{quality_report['feature_count']} features")

        return quality_report

    def balance_classes(self, features: pd.DataFrame, target: pd.Series,
                       method: str = 'undersample') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance classes using sampling

        Args:
            features: Feature DataFrame
            target: Target Series
            method: 'undersample' or 'oversample'

        Returns:
            Tuple of (balanced features, balanced target)
        """
        class_counts = target.value_counts()
        min_class_count = class_counts.min()

        if method == 'undersample':
            # Undersample majority classes
            balanced_indices = []

            for class_label in target.unique():
                class_indices = target[target == class_label].index
                sampled_indices = np.random.choice(
                    class_indices,
                    size=min_class_count,
                    replace=False
                )
                balanced_indices.extend(sampled_indices)

            balanced_indices = sorted(balanced_indices)

            balanced_features = features.loc[balanced_indices]
            balanced_target = target.loc[balanced_indices]

            logger.info(f"Undersampled to {len(balanced_features)} samples per class")

        elif method == 'oversample':
            # Oversample minority classes
            max_class_count = class_counts.max()
            balanced_indices = []

            for class_label in target.unique():
                class_indices = target[target == class_label].index
                sampled_indices = np.random.choice(
                    class_indices,
                    size=max_class_count,
                    replace=True
                )
                balanced_indices.extend(sampled_indices)

            balanced_indices = sorted(balanced_indices)

            balanced_features = features.loc[balanced_indices]
            balanced_target = target.loc[balanced_indices]

            logger.info(f"Oversampled to {len(balanced_features)} samples per class")

        else:
            raise ValueError(f"Unknown balancing method: {method}")

        return balanced_features, balanced_target

    def create_walk_forward_splits(self, features: pd.DataFrame, target: pd.Series,
                                   train_period: int = 252, test_period: int = 21
                                   ) -> list:
        """
        Create walk-forward analysis splits

        Args:
            features: Feature DataFrame
            target: Target Series
            train_period: Number of days for training
            test_period: Number of days for testing

        Returns:
            List of (train_indices, test_indices) tuples
        """
        splits = []
        n_samples = len(features)

        start_idx = 0
        while start_idx + train_period + test_period <= n_samples:
            train_end = start_idx + train_period
            test_end = train_end + test_period

            train_indices = range(start_idx, train_end)
            test_indices = range(train_end, test_end)

            splits.append((train_indices, test_indices))

            # Roll forward by test period (non-overlapping test sets)
            start_idx += test_period

        logger.info(f"Created {len(splits)} walk-forward splits "
                   f"(train: {train_period}, test: {test_period})")

        return splits


class DataLoader:
    """
    Load and cache market data for ML training
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize data loader

        Args:
            cache_dir: Directory for caching processed data
        """
        self.cache_dir = cache_dir

    def load_market_data(self, symbol: str, start_date: str, end_date: str,
                        source: str = 'yfinance') -> pd.DataFrame:
        """
        Load market data from source

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            source: Data source ('yfinance', 'database', etc.)

        Returns:
            DataFrame with OHLCV data
        """
        if source == 'yfinance':
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            # Rename columns to lowercase
            df.columns = df.columns.str.lower()

            # Remove timezone info from index
            df.index = df.index.tz_localize(None)

            logger.info(f"Loaded {len(df)} rows for {symbol} from {start_date} to {end_date}")

            return df

        elif source == 'database':
            # TODO: Implement database loading
            raise NotImplementedError("Database loading not yet implemented")

        else:
            raise ValueError(f"Unknown data source: {source}")

    def cache_processed_data(self, data: pd.DataFrame, cache_key: str):
        """
        Cache processed data for faster loading

        Args:
            data: DataFrame to cache
            cache_key: Unique key for cached data
        """
        if self.cache_dir:
            import os
            os.makedirs(self.cache_dir, exist_ok=True)

            cache_path = os.path.join(self.cache_dir, f"{cache_key}.parquet")
            data.to_parquet(cache_path)

            logger.info(f"Cached data to {cache_path}")

    def load_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Load cached data if available

        Args:
            cache_key: Unique key for cached data

        Returns:
            DataFrame if cached, None otherwise
        """
        if self.cache_dir:
            import os
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.parquet")

            if os.path.exists(cache_path):
                df = pd.read_parquet(cache_path)
                logger.info(f"Loaded cached data from {cache_path}")
                return df

        return None
