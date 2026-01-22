"""
Model Training Module
Handles training of ML models with hyperparameter tuning and cross-validation.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, Any
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from ..utils.logger import get_logger
from ..utils.config import get_config


class ModelTrainer:
    """Train and tune machine learning models."""

    def __init__(self):
        """Initialize ModelTrainer."""
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.model = None
        self.best_params_ = None
        self.cv_results_ = None

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and test sets.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Test set proportion
            random_state: Random seed
            stratify: Whether to stratify split

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.logger.info("Splitting data into train and test sets")

        if test_size is None:
            test_size = self.config.get("model.test_size", 0.25)

        if random_state is None:
            random_state = self.config.get("model.random_state", 42)

        stratify_arg = y if stratify and self.config.get("model.stratify", True) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
        )

        self.logger.info(
            f"Data split complete: Train={len(X_train)}, Test={len(X_test)}"
        )
        self.logger.info(
            f"Train class distribution:\n{y_train.value_counts()}"
        )
        self.logger.info(
            f"Test class distribution:\n{y_test.value_counts()}"
        )

        return X_train, X_test, y_train, y_test

    def create_model(self, model_type: str = "svm") -> Any:
        """
        Create model instance based on configuration.

        Args:
            model_type: Type of model ('svm', 'logistic_regression')

        Returns:
            Instantiated model

        Raises:
            ValueError: If model type is unknown
        """
        self.logger.info(f"Creating {model_type} model")

        if model_type == "svm":
            params = self.config.get("model.best_model", {})
            model = SVC(
                kernel=params.get("kernel", "poly"),
                C=params.get("C", 3),
                tol=params.get("tol", 0.001),
                probability=params.get("probability", True),
                class_weight=params.get("class_weight", "balanced"),
                random_state=params.get("random_state", 42),
            )

        elif model_type == "logistic_regression":
            params = self.config.get("model.logistic_regression", {})
            model = LogisticRegression(
                solver=params.get("solver", "saga"),
                max_iter=params.get("max_iter", 1000),
                C=params.get("C", 3.0),
                tol=params.get("tol", 0.0001),
                class_weight=params.get("class_weight", "balanced"),
                random_state=params.get("random_state", 42),
            )

        elif model_type == "svm_linear":
            params = self.config.get("model.svm_linear", {})
            model = SVC(
                kernel=params.get("kernel", "linear"),
                C=params.get("C", 3),
                tol=params.get("tol", 0.01),
                probability=params.get("probability", True),
                class_weight=params.get("class_weight", "balanced"),
                random_state=params.get("random_state", 42),
            )

        else:
            error_msg = f"Unknown model type: {model_type}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info(f"Model created: {model.__class__.__name__}")
        return model

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str = "svm",
    ) -> Any:
        """
        Train model on training data.

        Args:
            X_train: Training features
            y_train: Training target
            model_type: Type of model to train

        Returns:
            Trained model
        """
        self.logger.info(f"Training {model_type} model")

        self.model = self.create_model(model_type)

        try:
            self.model.fit(X_train, y_train)
            self.logger.info("Model training complete")

            # Log training score
            train_score = self.model.score(X_train, y_train)
            self.logger.info(f"Training accuracy: {train_score:.4f}")

        except Exception as e:
            error_msg = f"Error training model: {str(e)}"
            self.logger.error(error_msg)
            raise

        return self.model

    def hyperparameter_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str = "svm",
        method: str = "grid",
    ) -> Tuple[Any, Dict]:
        """
        Perform hyperparameter tuning with cross-validation.

        Args:
            X_train: Training features
            y_train: Training target
            model_type: Type of model to tune
            method: Tuning method ('grid' or 'random')

        Returns:
            Tuple of (best_model, best_params)
        """
        self.logger.info(f"Starting hyperparameter tuning for {model_type} using {method} search")

        # Check if tuning is enabled
        tuning_enabled = self.config.get("model.hyperparameter_tuning.enabled", True)
        if not tuning_enabled:
            self.logger.info("Hyperparameter tuning disabled, using default model")
            return self.train_model(X_train, y_train, model_type), {}

        # Get parameter grid
        if model_type == "svm":
            param_grid = self.config.get("model.hyperparameter_tuning.svm_param_grid", {})
            base_model = SVC(probability=True, class_weight="balanced", random_state=42)
        elif model_type == "logistic_regression":
            param_grid = self.config.get("model.hyperparameter_tuning.logistic_param_grid", {})
            base_model = LogisticRegression(class_weight="balanced", random_state=42)
        else:
            error_msg = f"Hyperparameter tuning not configured for {model_type}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Create cross-validation strategy
        cv_folds = self.config.get("model.cv_folds", 5)
        cv_scoring = self.config.get("model.cv_scoring", "roc_auc")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Perform search
        try:
            if method == "grid":
                search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=cv,
                    scoring=cv_scoring,
                    n_jobs=self.config.get("model.hyperparameter_tuning.n_jobs", -1),
                    verbose=self.config.get("model.hyperparameter_tuning.verbose", 2),
                )
            else:  # random search
                from sklearn.model_selection import RandomizedSearchCV
                search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_grid,
                    n_iter=20,
                    cv=cv,
                    scoring=cv_scoring,
                    n_jobs=self.config.get("model.hyperparameter_tuning.n_jobs", -1),
                    verbose=self.config.get("model.hyperparameter_tuning.verbose", 2),
                    random_state=42,
                )

            search.fit(X_train, y_train)

            self.model = search.best_estimator_
            self.best_params_ = search.best_params_
            self.cv_results_ = search.cv_results_

            self.logger.info(f"Best parameters: {self.best_params_}")
            self.logger.info(f"Best CV score: {search.best_score_:.4f}")

            return self.model, self.best_params_

        except Exception as e:
            error_msg = f"Error in hyperparameter tuning: {str(e)}"
            self.logger.error(error_msg)
            raise

    def cross_validate_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Optional[Any] = None,
        cv_folds: Optional[int] = None,
    ) -> Dict:
        """
        Perform cross-validation on model.

        Args:
            X: Feature matrix
            y: Target vector
            model: Model to cross-validate. If None, uses self.model
            cv_folds: Number of CV folds

        Returns:
            Dictionary with CV results
        """
        self.logger.info("Performing cross-validation")

        if model is None:
            if self.model is None:
                error_msg = "No model available. Train a model first."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            model = self.model

        if cv_folds is None:
            cv_folds = self.config.get("model.cv_folds", 5)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        }

        try:
            cv_results = cross_validate(
                model, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1
            )

            # Log results
            self.logger.info("Cross-validation results:")
            for metric in scoring.keys():
                test_scores = cv_results[f"test_{metric}"]
                mean_score = test_scores.mean()
                std_score = test_scores.std()
                self.logger.info(f"{metric.upper()}: {mean_score:.4f} ± {std_score:.4f}")

            return cv_results

        except Exception as e:
            error_msg = f"Error in cross-validation: {str(e)}"
            self.logger.error(error_msg)
            raise

    def save_model(
        self,
        model: Optional[Any] = None,
        encoder_artifacts: Optional[Dict] = None,
        feature_names: Optional[list] = None,
        model_path: Optional[str] = None,
    ) -> str:
        """
        Save model and artifacts to disk.

        Args:
            model: Model to save. If None, uses self.model
            encoder_artifacts: Encoder artifacts (scalers, encoders)
            feature_names: List of feature names
            model_path: Path to save model. If None, uses config default

        Returns:
            Path where model was saved
        """
        self.logger.info("Saving model artifacts")

        if model is None:
            if self.model is None:
                error_msg = "No model to save"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            model = self.model

        # Determine save path
        if model_path is None:
            models_dir = self.config.get("persistence.models_dir", "data/models")
            version = self.config.get("persistence.version", "1.0.0")
            filename_template = self.config.get(
                "persistence.model_filename", "school_shooting_model_v{version}.pkl"
            )
            filename = filename_template.format(version=version)
            model_path = Path(models_dir) / filename

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare artifacts
        artifacts = {
            "model": model,
            "best_params": self.best_params_,
            "feature_names": feature_names,
            "metadata": {
                "model_type": model.__class__.__name__,
                "trained_date": datetime.now().isoformat(),
                "version": self.config.get("persistence.version", "1.0.0"),
            },
        }

        # Add encoder artifacts if provided
        if encoder_artifacts:
            artifacts.update(encoder_artifacts)

        # Save to disk
        try:
            joblib.dump(artifacts, model_path)
            self.logger.info(f"Model saved successfully to: {model_path}")
            return str(model_path)

        except Exception as e:
            error_msg = f"Error saving model: {str(e)}"
            self.logger.error(error_msg)
            raise

    @staticmethod
    def load_model(model_path: str) -> Dict:
        """
        Load model and artifacts from disk.

        Args:
            model_path: Path to saved model

        Returns:
            Dictionary with model and artifacts

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        logger = get_logger(__name__)
        logger.info(f"Loading model from: {model_path}")

        model_path = Path(model_path)

        if not model_path.exists():
            error_msg = f"Model file not found: {model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            artifacts = joblib.load(model_path)
            logger.info("Model loaded successfully")
            logger.info(f"Model type: {artifacts['metadata']['model_type']}")
            logger.info(f"Trained date: {artifacts['metadata']['trained_date']}")

            return artifacts

        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(error_msg)
            raise
