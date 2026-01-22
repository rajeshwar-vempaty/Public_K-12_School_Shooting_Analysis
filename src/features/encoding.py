"""
Feature Encoding Module
Handles encoding of categorical features with CountEncoder and LabelEncoder.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
from category_encoders import CountEncoder
from ..utils.logger import get_logger
from ..utils.config import get_config


class FeatureEncoder:
    """Encode categorical features and scale numerical features."""

    def __init__(self):
        """Initialize FeatureEncoder."""
        self.config = get_config()
        self.logger = get_logger(__name__)

        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.count_encoders: Dict[str, CountEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_names_: Optional[List[str]] = None
        self.categorical_columns_: Optional[List[str]] = None

    def fit_encoders(
        self,
        X_train: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
        threshold: Optional[int] = None,
    ) -> None:
        """
        Fit encoders on training data.

        Args:
            X_train: Training feature matrix
            categorical_columns: List of categorical columns. If None, auto-detect
            threshold: Cardinality threshold for CountEncoder vs LabelEncoder
        """
        self.logger.info("Fitting encoders on training data")

        # Auto-detect categorical columns if not provided
        if categorical_columns is None:
            categorical_columns = X_train.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        self.categorical_columns_ = categorical_columns

        # Get threshold from config if not provided
        if threshold is None:
            threshold = self.config.get("preprocessing.high_cardinality_threshold", 3)

        self.logger.info(
            f"Encoding {len(categorical_columns)} categorical columns "
            f"with threshold {threshold}"
        )

        # Get unique counts
        unique_counts = X_train[categorical_columns].nunique()

        # Fit encoders for each categorical column
        for col in categorical_columns:
            if col not in X_train.columns:
                self.logger.warning(f"Column '{col}' not found in training data")
                continue

            n_unique = unique_counts[col]

            if n_unique > threshold:
                # High cardinality: Use CountEncoder
                self.count_encoders[col] = CountEncoder()
                self.count_encoders[col].fit(X_train[[col]])
                self.logger.info(
                    f"Fitted CountEncoder for '{col}' ({n_unique} unique values)"
                )
            else:
                # Low cardinality: Use LabelEncoder
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(X_train[col])
                self.logger.info(
                    f"Fitted LabelEncoder for '{col}' ({n_unique} unique values)"
                )

        self.logger.info(
            f"Fitted {len(self.count_encoders)} CountEncoders and "
            f"{len(self.label_encoders)} LabelEncoders"
        )

    def transform_encoders(
        self,
        X: pd.DataFrame,
        handle_unseen: bool = True
    ) -> pd.DataFrame:
        """
        Transform data using fitted encoders.

        Args:
            X: Feature matrix to transform
            handle_unseen: Whether to handle unseen categories

        Returns:
            Encoded feature matrix
        """
        self.logger.info("Transforming data with encoders")

        X_encoded = X.copy()

        # Apply CountEncoders
        for col, encoder in self.count_encoders.items():
            if col in X_encoded.columns:
                X_encoded[col] = encoder.transform(X_encoded[[col]])

        # Apply LabelEncoders
        for col, encoder in self.label_encoders.items():
            if col in X_encoded.columns:
                if handle_unseen:
                    # Map unseen values to 'Unknown'
                    X_encoded[col] = X_encoded[col].map(
                        lambda s: s if s in encoder.classes_ else "Unknown"
                    )

                    # Add 'Unknown' to encoder classes if not present
                    if "Unknown" not in encoder.classes_:
                        encoder.classes_ = np.append(encoder.classes_, "Unknown")

                X_encoded[col] = encoder.transform(X_encoded[col])

        self.logger.info("Encoding transformation complete")

        return X_encoded

    def fit_transform_encoders(
        self,
        X_train: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
        threshold: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fit and transform encoders in one step.

        Args:
            X_train: Training feature matrix
            categorical_columns: List of categorical columns
            threshold: Cardinality threshold

        Returns:
            Encoded training data
        """
        self.fit_encoders(X_train, categorical_columns, threshold)
        return self.transform_encoders(X_train, handle_unseen=False)

    def fit_scaler(self, X_train: pd.DataFrame) -> None:
        """
        Fit StandardScaler on training data.

        Args:
            X_train: Training feature matrix (should be encoded first)
        """
        self.logger.info("Fitting StandardScaler on training data")

        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        self.feature_names_ = X_train.columns.tolist()

        self.logger.info(f"Scaler fitted on {X_train.shape[1]} features")

    def transform_scaler(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted scaler.

        Args:
            X: Feature matrix to scale

        Returns:
            Scaled feature array

        Raises:
            ValueError: If scaler hasn't been fitted
        """
        if self.scaler is None:
            error_msg = "Scaler not fitted. Call fit_scaler() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info("Scaling data with StandardScaler")

        X_scaled = self.scaler.transform(X)

        return X_scaled

    def fit_transform_scaler(self, X_train: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform scaler in one step.

        Args:
            X_train: Training feature matrix

        Returns:
            Scaled training data
        """
        self.fit_scaler(X_train)
        return self.transform_scaler(X_train)

    def encode_and_scale_train(
        self,
        X_train: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
        threshold: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Complete encoding and scaling pipeline for training data.

        Args:
            X_train: Training feature matrix
            categorical_columns: List of categorical columns
            threshold: Cardinality threshold

        Returns:
            Tuple of (encoded_df, scaled_array)
        """
        self.logger.info("Starting full encoding and scaling pipeline for training data")

        # Encode categorical features
        X_train_encoded = self.fit_transform_encoders(
            X_train, categorical_columns, threshold
        )

        # Scale all features
        X_train_scaled = self.fit_transform_scaler(X_train_encoded)

        self.logger.info("Training data encoding and scaling complete")

        return X_train_encoded, X_train_scaled

    def encode_and_scale_test(self, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Complete encoding and scaling pipeline for test data.

        Args:
            X_test: Test feature matrix

        Returns:
            Tuple of (encoded_df, scaled_array)

        Raises:
            ValueError: If encoders or scaler haven't been fitted
        """
        self.logger.info("Starting encoding and scaling pipeline for test data")

        if not self.label_encoders and not self.count_encoders:
            error_msg = "Encoders not fitted. Call fit_encoders() or encode_and_scale_train() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Encode categorical features
        X_test_encoded = self.transform_encoders(X_test, handle_unseen=True)

        # Scale all features
        X_test_scaled = self.transform_scaler(X_test_encoded)

        self.logger.info("Test data encoding and scaling complete")

        return X_test_encoded, X_test_scaled

    def get_encoder_artifacts(self) -> Dict:
        """
        Get all encoder artifacts for persistence.

        Returns:
            Dictionary with all encoder objects
        """
        return {
            "label_encoders": self.label_encoders,
            "count_encoders": self.count_encoders,
            "scaler": self.scaler,
            "feature_names": self.feature_names_,
            "categorical_columns": self.categorical_columns_,
        }

    def set_encoder_artifacts(self, artifacts: Dict) -> None:
        """
        Set encoder artifacts from loaded objects.

        Args:
            artifacts: Dictionary with encoder objects
        """
        self.label_encoders = artifacts.get("label_encoders", {})
        self.count_encoders = artifacts.get("count_encoders", {})
        self.scaler = artifacts.get("scaler")
        self.feature_names_ = artifacts.get("feature_names")
        self.categorical_columns_ = artifacts.get("categorical_columns")

        self.logger.info("Encoder artifacts loaded successfully")

    def inverse_transform_target(self, y_encoded: np.ndarray, target_name: str = "Has_Victims") -> np.ndarray:
        """
        Inverse transform target variable if it was encoded.

        Args:
            y_encoded: Encoded target values
            target_name: Name of target variable

        Returns:
            Original target values
        """
        if target_name in self.label_encoders:
            return self.label_encoders[target_name].inverse_transform(y_encoded)
        return y_encoded
