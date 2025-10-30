"""Data validation and quality checks for market data."""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logger import logger


class DataQualityReport:
    """Report of data quality issues."""

    def __init__(self):
        self.issues: List[Dict[str, any]] = []
        self.warnings: List[Dict[str, any]] = []
        self.info: List[Dict[str, any]] = []

    def add_issue(
        self, level: str, category: str, message: str, details: Optional[Dict] = None
    ):
        """Add a quality issue to the report."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "category": category,
            "message": message,
            "details": details or {},
        }

        if level == "ERROR":
            self.issues.append(entry)
        elif level == "WARNING":
            self.warnings.append(entry)
        else:
            self.info.append(entry)

    def has_errors(self) -> bool:
        """Check if report contains errors."""
        return len(self.issues) > 0

    def summary(self) -> Dict[str, any]:
        """Get summary of quality report."""
        return {
            "total_issues": len(self.issues),
            "total_warnings": len(self.warnings),
            "total_info": len(self.info),
            "has_errors": self.has_errors(),
        }

    def to_dict(self) -> Dict[str, any]:
        """Convert report to dictionary."""
        return {
            "summary": self.summary(),
            "issues": self.issues,
            "warnings": self.warnings,
            "info": self.info,
        }


class DataValidator:
    """
    Validate and clean market data.

    Checks:
    - Missing data
    - Outliers (statistical and percentage-based)
    - Data integrity (OHLC relationships)
    - Stock splits and corporate actions
    - Trading volume anomalies
    """

    def __init__(
        self,
        outlier_threshold_std: float = 3.0,
        outlier_threshold_pct: float = 0.20,
        max_gap_fill: int = 3,
    ):
        """
        Initialize data validator.

        Args:
            outlier_threshold_std: Standard deviations for outlier detection
            outlier_threshold_pct: Percentage change threshold for outliers
            max_gap_fill: Maximum consecutive missing bars to fill
        """
        self.outlier_threshold_std = outlier_threshold_std
        self.outlier_threshold_pct = outlier_threshold_pct
        self.max_gap_fill = max_gap_fill

    def validate(
        self, data: pd.DataFrame, symbol: str = "UNKNOWN"
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Validate and clean market data.

        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol for logging

        Returns:
            Tuple of (cleaned_data, quality_report)
        """
        report = DataQualityReport()
        cleaned_data = data.copy()

        logger.info(
            "starting_validation",
            symbol=symbol,
            rows=len(data),
            columns=list(data.columns),
        )

        # 1. Check required columns
        cleaned_data, report = self._check_required_columns(
            cleaned_data, report, symbol
        )

        # 2. Check for missing values
        cleaned_data, report = self._check_missing_values(
            cleaned_data, report, symbol
        )

        # 3. Check OHLC relationships
        report = self._check_ohlc_integrity(cleaned_data, report, symbol)

        # 4. Detect outliers
        cleaned_data, report = self._detect_outliers(cleaned_data, report, symbol)

        # 5. Check for stock splits
        report = self._detect_stock_splits(cleaned_data, report, symbol)

        # 6. Validate volume
        report = self._validate_volume(cleaned_data, report, symbol)

        # 7. Check temporal consistency
        report = self._check_temporal_consistency(cleaned_data, report, symbol)

        logger.info(
            "validation_complete",
            symbol=symbol,
            summary=report.summary(),
        )

        return cleaned_data, report

    def _check_required_columns(
        self, data: pd.DataFrame, report: DataQualityReport, symbol: str
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """Check if required columns are present."""
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [col for col in required if col not in data.columns]

        if missing:
            report.add_issue(
                "ERROR",
                "MISSING_COLUMNS",
                f"Required columns missing: {missing}",
                {"symbol": symbol, "missing_columns": missing},
            )

        return data, report

    def _check_missing_values(
        self, data: pd.DataFrame, report: DataQualityReport, symbol: str
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """Check and handle missing values."""
        # Count missing values
        missing_counts = data[["Open", "High", "Low", "Close", "Volume"]].isnull().sum()
        total_missing = missing_counts.sum()

        if total_missing > 0:
            missing_pct = (total_missing / (len(data) * 5)) * 100

            if missing_pct > 5:
                report.add_issue(
                    "ERROR",
                    "EXCESSIVE_MISSING_DATA",
                    f"Too many missing values: {missing_pct:.2f}%",
                    {
                        "symbol": symbol,
                        "missing_percentage": missing_pct,
                        "missing_by_column": missing_counts.to_dict(),
                    },
                )
            else:
                report.add_issue(
                    "WARNING",
                    "MISSING_DATA",
                    f"Some missing values detected: {missing_pct:.2f}%",
                    {
                        "symbol": symbol,
                        "missing_percentage": missing_pct,
                        "missing_by_column": missing_counts.to_dict(),
                    },
                )

                # Forward fill small gaps
                data = data.fillna(method="ffill", limit=self.max_gap_fill)

                report.add_issue(
                    "INFO",
                    "DATA_FILLED",
                    f"Filled small gaps using forward fill (limit={self.max_gap_fill})",
                    {"symbol": symbol},
                )

        return data, report

    def _check_ohlc_integrity(
        self, data: pd.DataFrame, report: DataQualityReport, symbol: str
    ) -> DataQualityReport:
        """Check if OHLC relationships are valid."""
        # High should be >= Low
        invalid_high_low = data[data["High"] < data["Low"]]
        if len(invalid_high_low) > 0:
            report.add_issue(
                "ERROR",
                "INVALID_HIGH_LOW",
                f"Found {len(invalid_high_low)} bars where High < Low",
                {
                    "symbol": symbol,
                    "count": len(invalid_high_low),
                    "dates": invalid_high_low.index.tolist()[:10],
                },
            )

        # High should be >= Open and Close
        invalid_high = data[
            (data["High"] < data["Open"]) | (data["High"] < data["Close"])
        ]
        if len(invalid_high) > 0:
            report.add_issue(
                "ERROR",
                "INVALID_HIGH",
                f"Found {len(invalid_high)} bars where High < Open or Close",
                {
                    "symbol": symbol,
                    "count": len(invalid_high),
                    "dates": invalid_high.index.tolist()[:10],
                },
            )

        # Low should be <= Open and Close
        invalid_low = data[(data["Low"] > data["Open"]) | (data["Low"] > data["Close"])]
        if len(invalid_low) > 0:
            report.add_issue(
                "ERROR",
                "INVALID_LOW",
                f"Found {len(invalid_low)} bars where Low > Open or Close",
                {
                    "symbol": symbol,
                    "count": len(invalid_low),
                    "dates": invalid_low.index.tolist()[:10],
                },
            )

        return report

    def _detect_outliers(
        self, data: pd.DataFrame, report: DataQualityReport, symbol: str
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """Detect outliers using statistical and percentage methods."""
        # Calculate returns
        returns = data["Close"].pct_change()

        # Statistical outliers (Z-score method)
        mean_return = returns.mean()
        std_return = returns.std()
        z_scores = np.abs((returns - mean_return) / std_return)
        statistical_outliers = data[z_scores > self.outlier_threshold_std]

        if len(statistical_outliers) > 0:
            report.add_issue(
                "WARNING",
                "STATISTICAL_OUTLIERS",
                f"Found {len(statistical_outliers)} statistical outliers (>{self.outlier_threshold_std}σ)",
                {
                    "symbol": symbol,
                    "count": len(statistical_outliers),
                    "dates": statistical_outliers.index.tolist()[:10],
                },
            )

        # Percentage-based outliers
        pct_outliers = data[np.abs(returns) > self.outlier_threshold_pct]
        if len(pct_outliers) > 0:
            report.add_issue(
                "WARNING",
                "LARGE_PRICE_CHANGES",
                f"Found {len(pct_outliers)} large price changes (>{self.outlier_threshold_pct*100}%)",
                {
                    "symbol": symbol,
                    "count": len(pct_outliers),
                    "dates": pct_outliers.index.tolist()[:10],
                    "max_change": returns.abs().max(),
                },
            )

        return data, report

    def _detect_stock_splits(
        self, data: pd.DataFrame, report: DataQualityReport, symbol: str
    ) -> DataQualityReport:
        """Detect potential stock splits."""
        # Look for large overnight gaps with volume spikes
        returns = data["Close"].pct_change()
        volume_ratio = data["Volume"] / data["Volume"].rolling(20).mean()

        # Potential split: >30% price drop with volume spike
        potential_splits = data[
            (returns < -0.3) & (returns > -0.6) & (volume_ratio > 1.5)
        ]

        if len(potential_splits) > 0:
            report.add_issue(
                "WARNING",
                "POTENTIAL_STOCK_SPLIT",
                f"Detected {len(potential_splits)} potential stock splits",
                {
                    "symbol": symbol,
                    "count": len(potential_splits),
                    "dates": potential_splits.index.tolist(),
                },
            )

        return report

    def _validate_volume(
        self, data: pd.DataFrame, report: DataQualityReport, symbol: str
    ) -> DataQualityReport:
        """Validate trading volume."""
        # Check for zero volume
        zero_volume = data[data["Volume"] == 0]
        if len(zero_volume) > 0:
            report.add_issue(
                "WARNING",
                "ZERO_VOLUME",
                f"Found {len(zero_volume)} bars with zero volume",
                {
                    "symbol": symbol,
                    "count": len(zero_volume),
                    "dates": zero_volume.index.tolist()[:10],
                },
            )

        # Check for negative volume (should never happen)
        negative_volume = data[data["Volume"] < 0]
        if len(negative_volume) > 0:
            report.add_issue(
                "ERROR",
                "NEGATIVE_VOLUME",
                f"Found {len(negative_volume)} bars with negative volume",
                {
                    "symbol": symbol,
                    "count": len(negative_volume),
                    "dates": negative_volume.index.tolist(),
                },
            )

        return report

    def _check_temporal_consistency(
        self, data: pd.DataFrame, report: DataQualityReport, symbol: str
    ) -> DataQualityReport:
        """Check temporal consistency of data."""
        if "Date" in data.columns:
            # Check for duplicate dates
            duplicates = data[data["Date"].duplicated()]
            if len(duplicates) > 0:
                report.add_issue(
                    "ERROR",
                    "DUPLICATE_DATES",
                    f"Found {len(duplicates)} duplicate dates",
                    {
                        "symbol": symbol,
                        "count": len(duplicates),
                        "dates": duplicates["Date"].tolist(),
                    },
                )

            # Check if data is sorted
            if not data["Date"].is_monotonic_increasing:
                report.add_issue(
                    "WARNING",
                    "UNSORTED_DATA",
                    "Data is not sorted by date",
                    {"symbol": symbol},
                )

        return report
