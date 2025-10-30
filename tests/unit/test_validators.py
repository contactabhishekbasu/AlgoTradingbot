"""Unit tests for data validators."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.validators import DataValidator, DataQualityReport


class TestDataQualityReport:
    """Test data quality reporting."""

    def test_report_initialization(self):
        """Test report initialization."""
        report = DataQualityReport()
        assert len(report.issues) == 0
        assert len(report.warnings) == 0
        assert len(report.info) == 0

    def test_add_issue(self):
        """Test adding issues to report."""
        report = DataQualityReport()

        report.add_issue("ERROR", "TEST", "Test error")
        assert len(report.issues) == 1

        report.add_issue("WARNING", "TEST", "Test warning")
        assert len(report.warnings) == 1

        report.add_issue("INFO", "TEST", "Test info")
        assert len(report.info) == 1

    def test_has_errors(self):
        """Test error detection."""
        report = DataQualityReport()
        assert report.has_errors() is False

        report.add_issue("ERROR", "TEST", "Error")
        assert report.has_errors() is True

    def test_summary(self):
        """Test report summary."""
        report = DataQualityReport()
        report.add_issue("ERROR", "TEST", "Error")
        report.add_issue("WARNING", "TEST", "Warning")

        summary = report.summary()
        assert summary["total_issues"] == 1
        assert summary["total_warnings"] == 1
        assert summary["has_errors"] is True


class TestDataValidator:
    """Test data validation functionality."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return DataValidator()

    @pytest.fixture
    def valid_data(self):
        """Create valid OHLCV data."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
        return pd.DataFrame(
            {
                "Date": dates,
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
                "High": [105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0],
                "Low": [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
                "Close": [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0],
                "Volume": [1000000] * 10,
            }
        )

    def test_validate_valid_data(self, validator, valid_data):
        """Test validation with valid data."""
        cleaned, report = validator.validate(valid_data, "AAPL")

        assert not report.has_errors()
        assert len(cleaned) == len(valid_data)

    def test_missing_columns(self, validator):
        """Test validation with missing columns."""
        invalid_data = pd.DataFrame({"Close": [100, 101, 102]})

        cleaned, report = validator.validate(invalid_data, "AAPL")

        assert report.has_errors()
        assert any("MISSING_COLUMNS" in issue["category"] for issue in report.issues)

    def test_missing_values(self, validator, valid_data):
        """Test validation with missing values."""
        data_with_missing = valid_data.copy()
        data_with_missing.loc[2, "Close"] = np.nan
        data_with_missing.loc[3, "Close"] = np.nan

        cleaned, report = validator.validate(data_with_missing, "AAPL")

        # Should have warnings but not necessarily errors
        assert len(report.warnings) > 0 or len(report.info) > 0

    def test_invalid_ohlc_relationships(self, validator, valid_data):
        """Test validation with invalid OHLC relationships."""
        invalid_data = valid_data.copy()
        # Make High < Low (invalid)
        invalid_data.loc[2, "High"] = 90.0
        invalid_data.loc[2, "Low"] = 95.0

        cleaned, report = validator.validate(invalid_data, "AAPL")

        assert report.has_errors()
        assert any("INVALID_HIGH_LOW" in issue["category"] for issue in report.issues)

    def test_outlier_detection(self, validator):
        """Test outlier detection."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-30", freq="D")
        data = pd.DataFrame(
            {
                "Date": dates,
                "Open": [100.0] * len(dates),
                "High": [105.0] * len(dates),
                "Low": [95.0] * len(dates),
                "Close": [100.0] * (len(dates) - 1) + [200.0],  # Outlier
                "Volume": [1000000] * len(dates),
            }
        )

        cleaned, report = validator.validate(data, "AAPL")

        # Should detect outlier
        assert len(report.warnings) > 0

    def test_zero_volume_detection(self, validator, valid_data):
        """Test detection of zero volume."""
        data_with_zero_volume = valid_data.copy()
        data_with_zero_volume.loc[5, "Volume"] = 0

        cleaned, report = validator.validate(data_with_zero_volume, "AAPL")

        assert any("ZERO_VOLUME" in warning["category"] for warning in report.warnings)

    def test_negative_volume_detection(self, validator, valid_data):
        """Test detection of negative volume."""
        data_with_negative_volume = valid_data.copy()
        data_with_negative_volume.loc[5, "Volume"] = -1000

        cleaned, report = validator.validate(data_with_negative_volume, "AAPL")

        assert report.has_errors()
        assert any("NEGATIVE_VOLUME" in issue["category"] for issue in report.issues)

    def test_duplicate_dates(self, validator, valid_data):
        """Test detection of duplicate dates."""
        data_with_duplicates = pd.concat([valid_data, valid_data.iloc[[0]]])

        cleaned, report = validator.validate(data_with_duplicates, "AAPL")

        assert report.has_errors()
        assert any("DUPLICATE_DATES" in issue["category"] for issue in report.issues)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
