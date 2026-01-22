"""
Model Prediction Module
Handles making predictions on new data using trained models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple
from pathlib import Path
from ..utils.logger import get_logger
from ..utils.config import get_config


class ModelPredictor:
    """Make predictions using trained models."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize ModelPredictor.

        Args:
            model_path: Path to saved model artifacts
        """
        self.config = get_config()
        self.logger = get_logger(__name__)

        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.count_encoders = None
        self.feature_names = None
        self.metadata = None

        if model_path:
            self.load_artifacts(model_path)

    def load_artifacts(self, model_path: str) -> None:
        """
        Load model and all necessary artifacts.

        Args:
            model_path: Path to saved model

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        self.logger.info(f"Loading model artifacts from: {model_path}")

        from .trainer import ModelTrainer

        artifacts = ModelTrainer.load_model(model_path)

        self.model = artifacts.get("model")
        self.scaler = artifacts.get("scaler")
        self.label_encoders = artifacts.get("label_encoders", {})
        self.count_encoders = artifacts.get("count_encoders", {})
        self.feature_names = artifacts.get("feature_names", [])
        self.metadata = artifacts.get("metadata", {})

        self.logger.info("Model artifacts loaded successfully")
        self.logger.info(f"Model type: {self.metadata.get('model_type', 'Unknown')}")
        self.logger.info(f"Expected features: {len(self.feature_names)}")

    def preprocess_input(
        self, X: pd.DataFrame, handle_unseen: bool = True
    ) -> np.ndarray:
        """
        Preprocess input data using loaded encoders and scaler.

        Args:
            X: Input features
            handle_unseen: Whether to handle unseen categories

        Returns:
            Preprocessed feature array

        Raises:
            ValueError: If artifacts not loaded or features mismatch
        """
        if self.model is None:
            error_msg = "Model not loaded. Call load_artifacts() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info("Preprocessing input data")

        X_processed = X.copy()

        # Check for missing features
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            self.logger.warning(f"Missing features in input: {missing_features}")
            # Add missing features with default values
            for feature in missing_features:
                X_processed[feature] = 0

        # Ensure correct column order
        X_processed = X_processed[self.feature_names]

        # Apply count encoders
        for col, encoder in self.count_encoders.items():
            if col in X_processed.columns:
                X_processed[col] = encoder.transform(X_processed[[col]])

        # Apply label encoders
        for col, encoder in self.label_encoders.items():
            if col in X_processed.columns:
                if handle_unseen:
                    # Map unseen values to 'Unknown'
                    X_processed[col] = X_processed[col].map(
                        lambda s: s if s in encoder.classes_ else "Unknown"
                    )

                    # Add 'Unknown' to encoder classes if not present
                    if "Unknown" not in encoder.classes_:
                        encoder.classes_ = np.append(encoder.classes_, "Unknown")

                X_processed[col] = encoder.transform(X_processed[col])

        # Apply scaling
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_processed)
        else:
            self.logger.warning("No scaler found, using unscaled data")
            X_scaled = X_processed.values

        self.logger.info(f"Preprocessing complete. Shape: {X_scaled.shape}")

        return X_scaled

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions on input data.

        Args:
            X: Input features (DataFrame or array)

        Returns:
            Predicted labels

        Raises:
            ValueError: If model not loaded
        """
        if self.model is None:
            error_msg = "Model not loaded. Call load_artifacts() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info("Making predictions")

        # Preprocess if DataFrame
        if isinstance(X, pd.DataFrame):
            X_processed = self.preprocess_input(X)
        else:
            X_processed = X

        predictions = self.model.predict(X_processed)

        self.logger.info(f"Generated {len(predictions)} predictions")

        return predictions

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Predicted probabilities

        Raises:
            ValueError: If model doesn't support predict_proba
        """
        if self.model is None:
            error_msg = "Model not loaded. Call load_artifacts() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if not hasattr(self.model, "predict_proba"):
            error_msg = "Model does not support probability predictions"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info("Predicting probabilities")

        # Preprocess if DataFrame
        if isinstance(X, pd.DataFrame):
            X_processed = self.preprocess_input(X)
        else:
            X_processed = X

        probabilities = self.model.predict_proba(X_processed)

        self.logger.info(f"Generated {len(probabilities)} probability predictions")

        return probabilities

    def predict_single(
        self, features: Dict, return_probability: bool = True
    ) -> Dict:
        """
        Make prediction on a single instance.

        Args:
            features: Dictionary of feature values
            return_probability: Whether to return probabilities

        Returns:
            Dictionary with prediction and optionally probability
        """
        self.logger.info("Making single instance prediction")

        # Convert to DataFrame
        X = pd.DataFrame([features])

        # Make prediction
        prediction = self.predict(X)[0]

        result = {"prediction": int(prediction), "has_victims": bool(prediction)}

        # Add probability if requested
        if return_probability and hasattr(self.model, "predict_proba"):
            probabilities = self.predict_proba(X)[0]
            result["probability_no_victims"] = float(probabilities[0])
            result["probability_has_victims"] = float(probabilities[1])
            result["confidence"] = float(max(probabilities))

        return result

    def batch_predict(
        self, X: pd.DataFrame, return_probabilities: bool = False
    ) -> pd.DataFrame:
        """
        Make predictions on batch of data.

        Args:
            X: Input features
            return_probabilities: Whether to include probabilities

        Returns:
            DataFrame with predictions
        """
        self.logger.info(f"Making batch predictions for {len(X)} instances")

        predictions = self.predict(X)

        results = pd.DataFrame({"prediction": predictions, "has_victims": predictions.astype(bool)})

        if return_probabilities and hasattr(self.model, "predict_proba"):
            probabilities = self.predict_proba(X)
            results["probability_no_victims"] = probabilities[:, 0]
            results["probability_has_victims"] = probabilities[:, 1]
            results["confidence"] = probabilities.max(axis=1)

        return results

    def get_model_info(self) -> Dict:
        """
        Get information about loaded model.

        Returns:
            Dictionary with model metadata
        """
        if self.model is None:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_type": self.metadata.get("model_type", "Unknown"),
            "version": self.metadata.get("version", "Unknown"),
            "trained_date": self.metadata.get("trained_date", "Unknown"),
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
        }
