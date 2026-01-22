"""
Feature Engineering Module
Creates new features and prepares data for modeling.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from ..utils.logger import get_logger
from ..utils.config import get_config


class FeatureEngineer:
    """Engineer features for machine learning models."""

    def __init__(self):
        """Initialize FeatureEngineer."""
        self.config = get_config()
        self.logger = get_logger(__name__)

    def create_target_variable(
        self, df: pd.DataFrame, target_col: str = "Number_Victims"
    ) -> pd.DataFrame:
        """
        Create binary target variable Has_Victims.

        Args:
            df: DataFrame with victim count
            target_col: Column name containing victim count

        Returns:
            DataFrame with Has_Victims column
        """
        self.logger.info("Creating target variable")

        df = df.copy()

        if target_col not in df.columns:
            self.logger.warning(f"Target column '{target_col}' not found")
            return df

        # Create binary target: 1 if victims > 0, else 0
        df["Has_Victims"] = df[target_col].apply(lambda x: 1 if x > 0 else 0)

        target_distribution = df["Has_Victims"].value_counts()
        self.logger.info(f"Target variable distribution:\n{target_distribution}")

        return df

    def create_date_features(self, df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
        """
        Create date-related features.

        Args:
            df: DataFrame with date column
            date_col: Name of date column

        Returns:
            DataFrame with date features
        """
        self.logger.info("Creating date features")

        df = df.copy()

        if date_col not in df.columns:
            self.logger.warning(f"Date column '{date_col}' not found")
            return df

        try:
            # Convert to datetime
            df[date_col] = pd.to_datetime(df[date_col])

            # Extract weekday name
            df["Weekday"] = df[date_col].dt.day_name()

            # Extract month name
            df["Month"] = df[date_col].dt.month_name()

            # Order months and weekdays
            months_order = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
            days_order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]

            df["Month"] = pd.Categorical(df["Month"], categories=months_order, ordered=True)
            df["Weekday"] = pd.Categorical(df["Weekday"], categories=days_order, ordered=True)

            self.logger.info("Date features created successfully")

        except Exception as e:
            self.logger.error(f"Error creating date features: {str(e)}")
            raise

        return df

    def create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from text columns.

        Args:
            df: DataFrame with text columns

        Returns:
            DataFrame with text features
        """
        self.logger.info("Creating text features")

        df = df.copy()

        # Summary length
        if "Summary" in df.columns:
            df["Summary_Length"] = df["Summary"].astype(str).apply(len)

        # Narrative length
        if "Narrative" in df.columns:
            df["Narrative_Length"] = df["Narrative"].astype(str).apply(len)

        self.logger.info("Text features created")

        return df

    def select_best_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select best performing features from config.

        Args:
            df: DataFrame with all features

        Returns:
            DataFrame with selected features only
        """
        self.logger.info("Selecting best features")

        selected_features = self.config.get("features.selected_features", [])

        if not selected_features:
            self.logger.warning("No features specified in config, returning all")
            return df

        # Include target if present
        target_col = self.config.get("preprocessing.target_column", "Has_Victims")
        if target_col in df.columns and target_col not in selected_features:
            selected_features = selected_features + [target_col]

        # Filter to available columns
        available_features = [col for col in selected_features if col in df.columns]
        missing_features = set(selected_features) - set(available_features)

        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")

        df_selected = df[available_features].copy()

        self.logger.info(
            f"Selected {len(available_features)} features out of {len(df.columns)} total"
        )

        return df_selected

    def split_features_target(
        self, df: pd.DataFrame, target_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into features (X) and target (y).

        Args:
            df: DataFrame with features and target
            target_col: Name of target column. If None, uses config default

        Returns:
            Tuple of (X, y)
        """
        self.logger.info("Splitting features and target")

        if target_col is None:
            target_col = self.config.get("preprocessing.target_column", "Has_Victims")

        if target_col not in df.columns:
            error_msg = f"Target column '{target_col}' not found in DataFrame"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        self.logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

        return X, y

    def get_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of categorical columns.

        Args:
            df: DataFrame

        Returns:
            List of categorical column names
        """
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        self.logger.info(f"Found {len(categorical_cols)} categorical columns")

        return categorical_cols

    def get_numerical_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of numerical columns.

        Args:
            df: DataFrame

        Returns:
            List of numerical column names
        """
        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()

        self.logger.info(f"Found {len(numerical_cols)} numerical columns")

        return numerical_cols

    def full_feature_engineering_pipeline(
        self, df: pd.DataFrame, select_best: bool = True
    ) -> pd.DataFrame:
        """
        Execute full feature engineering pipeline.

        Args:
            df: Raw preprocessed DataFrame
            select_best: Whether to select only best features

        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Starting full feature engineering pipeline")

        # Create target variable
        df = self.create_target_variable(df)

        # Create date features
        df = self.create_date_features(df)

        # Create text features (if needed)
        # df = self.create_text_features(df)

        # Select best features if requested
        if select_best:
            df = self.select_best_features(df)

        self.logger.info("Feature engineering pipeline complete")

        return df
