"""
Feature Selection Module
Implements Chi-squared, RFE, and RFECV feature selection methods.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.feature_selection import SelectKBest, chi2, RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from ..utils.logger import get_logger
from ..utils.config import get_config


class FeatureSelector:
    """Select important features using various methods."""

    def __init__(self):
        """Initialize FeatureSelector."""
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.selected_features_ = None

    def chi_squared_selection(
        self, X: pd.DataFrame, y: pd.Series, k: str = "all"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Feature selection using Chi-squared test.

        Args:
            X: Feature matrix (must be non-negative for chi2)
            y: Target vector
            k: Number of features to select ('all' or integer)

        Returns:
            Tuple of (selected_features_df, scores_df)
        """
        self.logger.info(f"Performing Chi-squared feature selection (k={k})")

        try:
            # Ensure all values are non-negative for chi2
            X_positive = X.copy()
            for col in X_positive.columns:
                if X_positive[col].min() < 0:
                    X_positive[col] = X_positive[col] - X_positive[col].min()

            # Apply SelectKBest
            selector = SelectKBest(score_func=chi2, k=k)
            selector.fit(X_positive, y)

            # Get scores
            scores = pd.DataFrame({
                'Feature': X.columns,
                'Chi2_Score': selector.scores_,
                'P_Value': selector.pvalues_
            })
            scores = scores.sort_values('Chi2_Score', ascending=False)

            # Get selected features
            selected_mask = selector.get_support()
            selected_features = X.columns[selected_mask].tolist()

            self.logger.info(f"Selected {len(selected_features)} features using Chi2")

            return scores, selected_features

        except Exception as e:
            self.logger.error(f"Error in chi-squared selection: {str(e)}")
            raise

    def rfe_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features_to_select: Optional[int] = None,
        estimator: Optional[str] = None,
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Feature selection using Recursive Feature Elimination.

        Args:
            X: Feature matrix
            y: Target vector
            n_features_to_select: Number of features to select
            estimator: Estimator type ('RandomForest', 'LogisticRegression')

        Returns:
            Tuple of (selected_features, feature_rankings)
        """
        self.logger.info("Performing RFE feature selection")

        # Get config values
        if n_features_to_select is None:
            n_features_to_select = self.config.get(
                "feature_selection.rfe.n_features_to_select", 15
            )

        if estimator is None:
            estimator = self.config.get(
                "feature_selection.rfe.estimator", "RandomForestClassifier"
            )

        try:
            # Create estimator
            if estimator == "RandomForestClassifier":
                base_estimator = RandomForestClassifier(random_state=42, n_estimators=100)
            else:
                from sklearn.linear_model import LogisticRegression
                base_estimator = LogisticRegression(random_state=42, max_iter=1000)

            # Apply RFE
            rfe = RFE(estimator=base_estimator, n_features_to_select=n_features_to_select)
            rfe.fit(X, y)

            # Get rankings
            rankings = pd.DataFrame({
                'Feature': X.columns,
                'Ranking': rfe.ranking_,
                'Selected': rfe.support_
            })
            rankings = rankings.sort_values('Ranking')

            # Get selected features
            selected_features = X.columns[rfe.support_].tolist()

            self.logger.info(f"RFE selected {len(selected_features)} features")

            return selected_features, rankings

        except Exception as e:
            self.logger.error(f"Error in RFE selection: {str(e)}")
            raise

    def rfecv_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: Optional[int] = None,
        scoring: Optional[str] = None,
    ) -> Tuple[List[str], int, pd.DataFrame]:
        """
        Feature selection using RFECV (RFE with Cross-Validation).

        Args:
            X: Feature matrix
            y: Target vector
            cv: Number of cross-validation folds
            scoring: Scoring metric

        Returns:
            Tuple of (selected_features, optimal_n_features, cv_scores)
        """
        self.logger.info("Performing RFECV feature selection")

        # Get config values
        if cv is None:
            cv = self.config.get("feature_selection.rfecv.cv", 5)

        if scoring is None:
            scoring = self.config.get("feature_selection.rfecv.scoring", "roc_auc")

        try:
            # Create estimator
            estimator = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)

            # Create stratified k-fold
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

            # Apply RFECV
            rfecv = RFECV(
                estimator=estimator,
                step=1,
                cv=cv_splitter,
                scoring=scoring,
                n_jobs=-1,
            )

            rfecv.fit(X, y)

            # Get selected features
            selected_features = X.columns[rfecv.support_].tolist()
            optimal_n = rfecv.n_features_

            # Get CV scores
            cv_scores = pd.DataFrame({
                'N_Features': range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
                'Mean_CV_Score': rfecv.cv_results_['mean_test_score'],
                'Std_CV_Score': rfecv.cv_results_['std_test_score']
            })

            self.logger.info(
                f"RFECV selected {optimal_n} features with best CV score: "
                f"{rfecv.cv_results_['mean_test_score'][optimal_n - 1]:.4f}"
            )

            return selected_features, optimal_n, cv_scores

        except Exception as e:
            self.logger.error(f"Error in RFECV selection: {str(e)}")
            raise

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "rfecv",
        **kwargs
    ) -> List[str]:
        """
        Select features using specified method.

        Args:
            X: Feature matrix
            y: Target vector
            method: Selection method ('chi2', 'rfe', 'rfecv', 'config')
            **kwargs: Additional arguments for selection method

        Returns:
            List of selected feature names
        """
        self.logger.info(f"Selecting features using method: {method}")

        if method == "chi2":
            _, selected_features = self.chi_squared_selection(X, y, **kwargs)

        elif method == "rfe":
            selected_features, _ = self.rfe_selection(X, y, **kwargs)

        elif method == "rfecv":
            selected_features, _, _ = self.rfecv_selection(X, y, **kwargs)

        elif method == "config":
            # Use features from config
            selected_features = self.config.get("features.selected_features", [])
            # Filter to available columns
            selected_features = [f for f in selected_features if f in X.columns]
            self.logger.info(f"Using {len(selected_features)} features from config")

        else:
            error_msg = f"Unknown feature selection method: {method}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.selected_features_ = selected_features

        return selected_features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform X to include only selected features.

        Args:
            X: Feature matrix

        Returns:
            Transformed feature matrix

        Raises:
            ValueError: If features haven't been selected yet
        """
        if self.selected_features_ is None:
            error_msg = "No features selected. Call select_features() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        available_features = [f for f in self.selected_features_ if f in X.columns]

        if len(available_features) < len(self.selected_features_):
            missing = set(self.selected_features_) - set(available_features)
            self.logger.warning(f"Missing features in transform: {missing}")

        return X[available_features].copy()

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "rfecv",
        **kwargs
    ) -> pd.DataFrame:
        """
        Select and transform features in one step.

        Args:
            X: Feature matrix
            y: Target vector
            method: Selection method
            **kwargs: Additional arguments

        Returns:
            Transformed feature matrix
        """
        selected_features = self.select_features(X, y, method, **kwargs)
        return self.transform(X)
