"""
Data Validation Module
Validates data schema, types, and quality checks.
"""

import pandas as pd
from typing import List, Dict, Optional, Tuple
from ..utils.logger import get_logger


class DataValidator:
    """Validate data quality and schema."""

    def __init__(self):
        """Initialize DataValidator."""
        self.logger = get_logger(__name__)

    def validate_required_columns(
        self, df: pd.DataFrame, required_columns: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that all required columns are present.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            Tuple of (is_valid, missing_columns)
        """
        missing_columns = set(required_columns) - set(df.columns)

        if missing_columns:
            self.logger.warning(f"Missing required columns: {missing_columns}")
            return False, list(missing_columns)

        self.logger.info("All required columns present")
        return True, []

    def validate_no_empty_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame is not empty.

        Args:
            df: DataFrame to validate

        Returns:
            True if DataFrame has data
        """
        if df.empty:
            self.logger.error("DataFrame is empty")
            return False

        self.logger.info(f"DataFrame has {len(df)} rows")
        return True

    def check_missing_values(
        self, df: pd.DataFrame, threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Check for columns with high percentage of missing values.

        Args:
            df: DataFrame to check
            threshold: Threshold for warning (0-1)

        Returns:
            Dictionary of column: missing_percentage for problematic columns
        """
        missing_percentages = df.isnull().sum() / len(df)
        problematic_columns = missing_percentages[missing_percentages > threshold]

        if len(problematic_columns) > 0:
            self.logger.warning(
                f"Columns with >{threshold*100}% missing values: "
                f"{problematic_columns.to_dict()}"
            )
        else:
            self.logger.info(f"No columns exceed {threshold*100}% missing threshold")

        return problematic_columns.to_dict()

    def check_duplicate_rows(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> int:
        """
        Check for duplicate rows.

        Args:
            df: DataFrame to check
            subset: Columns to check for duplicates. If None, checks all columns

        Returns:
            Number of duplicate rows
        """
        duplicate_count = df.duplicated(subset=subset).sum()

        if duplicate_count > 0:
            self.logger.warning(f"Found {duplicate_count} duplicate rows")
        else:
            self.logger.info("No duplicate rows found")

        return duplicate_count

    def validate_data_types(
        self, df: pd.DataFrame, expected_types: Dict[str, str]
    ) -> Dict[str, Tuple[str, str]]:
        """
        Validate column data types.

        Args:
            df: DataFrame to validate
            expected_types: Dictionary of column: expected_type

        Returns:
            Dictionary of columns with type mismatches
        """
        mismatches = {}

        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    mismatches[col] = (actual_type, expected_type)
                    self.logger.warning(
                        f"Column '{col}' has type '{actual_type}', expected '{expected_type}'"
                    )

        if not mismatches:
            self.logger.info("All column types match expectations")

        return mismatches

    def validate_value_ranges(
        self, df: pd.DataFrame, range_checks: Dict[str, Tuple[float, float]]
    ) -> Dict[str, int]:
        """
        Validate that numeric columns are within expected ranges.

        Args:
            df: DataFrame to validate
            range_checks: Dictionary of column: (min_value, max_value)

        Returns:
            Dictionary of columns with out-of-range value counts
        """
        out_of_range = {}

        for col, (min_val, max_val) in range_checks.items():
            if col in df.columns:
                invalid_count = (
                    (df[col] < min_val) | (df[col] > max_val)
                ).sum()

                if invalid_count > 0:
                    out_of_range[col] = invalid_count
                    self.logger.warning(
                        f"Column '{col}' has {invalid_count} values outside range [{min_val}, {max_val}]"
                    )

        if not out_of_range:
            self.logger.info("All values within expected ranges")

        return out_of_range

    def validate_categorical_values(
        self, df: pd.DataFrame, valid_values: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Validate that categorical columns only contain expected values.

        Args:
            df: DataFrame to validate
            valid_values: Dictionary of column: list_of_valid_values

        Returns:
            Dictionary of columns with invalid values
        """
        invalid_values = {}

        for col, expected_values in valid_values.items():
            if col in df.columns:
                actual_values = set(df[col].dropna().unique())
                unexpected = actual_values - set(expected_values)

                if unexpected:
                    invalid_values[col] = list(unexpected)
                    self.logger.warning(
                        f"Column '{col}' has unexpected values: {unexpected}"
                    )

        if not invalid_values:
            self.logger.info("All categorical values are valid")

        return invalid_values

    def generate_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data quality report.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with quality metrics
        """
        self.logger.info("Generating data quality report")

        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "duplicates": self.check_duplicate_rows(df),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df)).to_dict(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
            "categorical_columns": df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist(),
        }

        # Add numeric column statistics
        numeric_stats = {}
        for col in report["numeric_columns"]:
            numeric_stats[col] = {
                "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                "std": float(df[col].std()) if not df[col].isna().all() else None,
                "min": float(df[col].min()) if not df[col].isna().all() else None,
                "max": float(df[col].max()) if not df[col].isna().all() else None,
            }
        report["numeric_statistics"] = numeric_stats

        # Add categorical column statistics
        categorical_stats = {}
        for col in report["categorical_columns"]:
            categorical_stats[col] = {
                "unique_values": int(df[col].nunique()),
                "most_common": str(df[col].mode()[0]) if len(df[col].mode()) > 0 else None,
                "most_common_count": int(df[col].value_counts().iloc[0]) if len(df[col].value_counts()) > 0 else None,
            }
        report["categorical_statistics"] = categorical_stats

        self.logger.info("Data quality report generated")
        return report
