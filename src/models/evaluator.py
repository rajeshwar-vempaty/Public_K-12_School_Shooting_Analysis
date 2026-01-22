"""
Model Evaluation Module
Comprehensive model evaluation with metrics, plots, and reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.inspection import permutation_importance
from ..utils.logger import get_logger
from ..utils.config import get_config


class ModelEvaluator:
    """Evaluate model performance with comprehensive metrics."""

    def __init__(self):
        """Initialize ModelEvaluator."""
        self.config = get_config()
        self.logger = get_logger(__name__)

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for ROC-AUC)

        Returns:
            Dictionary with metrics
        """
        self.logger.info("Calculating evaluation metrics")

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

        # Add ROC-AUC if probabilities provided
        if y_prob is not None:
            if len(y_prob.shape) > 1:
                y_prob = y_prob[:, 1]  # Get probability of positive class
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

        self.logger.info("Metrics calculated:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric.upper()}: {value:.4f}")

        return metrics

    def generate_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> str:
        """
        Generate detailed classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Classification report as string
        """
        self.logger.info("Generating classification report")

        report = classification_report(y_true, y_pred)
        self.logger.info(f"Classification Report:\n{report}")

        return report

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Confusion Matrix",
    ) -> None:
        """
        Plot and optionally save confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save plot. If None, displays plot
            title: Plot title
        """
        self.logger.info("Plotting confusion matrix")

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Victims", "Has Victims"],
            yticklabels=["No Victims", "Has Victims"],
        )
        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.get("evaluation.dpi", 300))
            self.logger.info(f"Confusion matrix saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "ROC Curve",
    ) -> float:
        """
        Plot ROC curve and calculate AUC.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            save_path: Path to save plot
            title: Plot title

        Returns:
            AUC score
        """
        self.logger.info("Plotting ROC curve")

        if len(y_prob.shape) > 1:
            y_prob = y_prob[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.get("evaluation.dpi", 300))
            self.logger.info(f"ROC curve saved to: {save_path}")
        else:
            plt.show()

        plt.close()

        return roc_auc

    def plot_train_test_roc(
        self,
        y_train: np.ndarray,
        y_train_prob: np.ndarray,
        y_test: np.ndarray,
        y_test_prob: np.ndarray,
        save_path: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Plot ROC curves for both train and test sets.

        Args:
            y_train: Training true labels
            y_train_prob: Training predicted probabilities
            y_test: Test true labels
            y_test_prob: Test predicted probabilities
            save_path: Path to save plot

        Returns:
            Tuple of (train_auc, test_auc)
        """
        self.logger.info("Plotting train/test ROC curves")

        if len(y_train_prob.shape) > 1:
            y_train_prob = y_train_prob[:, 1]
        if len(y_test_prob.shape) > 1:
            y_test_prob = y_test_prob[:, 1]

        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)

        auc_train = auc(fpr_train, tpr_train)
        auc_test = auc(fpr_test, tpr_test)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr_train,
            tpr_train,
            color="blue",
            lw=2,
            label=f"Train ROC (AUC = {auc_train:.4f})",
        )
        plt.plot(
            fpr_test, tpr_test, color="red", lw=2, label=f"Test ROC (AUC = {auc_test:.4f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves: Train vs Test")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.get("evaluation.dpi", 300))
            self.logger.info(f"Train/Test ROC curves saved to: {save_path}")
        else:
            plt.show()

        plt.close()

        return auc_train, auc_test

    def calculate_permutation_importance(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[list] = None,
        n_repeats: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Calculate permutation feature importance.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            feature_names: Feature names
            n_repeats: Number of permutations

        Returns:
            DataFrame with feature importances
        """
        self.logger.info("Calculating permutation importance")

        if n_repeats is None:
            n_repeats = self.config.get("evaluation.permutation_importance.n_repeats", 10)

        random_state = self.config.get("evaluation.permutation_importance.random_state", 42)

        try:
            perm_importance = permutation_importance(
                model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
            )

            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance_Mean": perm_importance.importances_mean,
                "Importance_Std": perm_importance.importances_std,
            })

            importance_df = importance_df.sort_values("Importance_Mean", ascending=False)

            self.logger.info("Top 10 important features:")
            for idx, row in importance_df.head(10).iterrows():
                self.logger.info(
                    f"  {row['Feature']}: {row['Importance_Mean']:.4f} ± {row['Importance_Std']:.4f}"
                )

            return importance_df

        except Exception as e:
            self.logger.error(f"Error calculating permutation importance: {str(e)}")
            raise

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot feature importance.

        Args:
            importance_df: DataFrame with feature importances
            top_n: Number of top features to plot
            save_path: Path to save plot
        """
        self.logger.info(f"Plotting top {top_n} important features")

        plot_data = importance_df.head(top_n).sort_values("Importance_Mean")

        plt.figure(figsize=(10, 8))
        plt.barh(plot_data["Feature"], plot_data["Importance_Mean"])
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Feature Importances")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.get("evaluation.dpi", 300))
            self.logger.info(f"Feature importance plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def comprehensive_evaluation(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[list] = None,
        save_dir: Optional[str] = None,
    ) -> Dict:
        """
        Perform comprehensive model evaluation.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            feature_names: Feature names
            save_dir: Directory to save plots

        Returns:
            Dictionary with all evaluation results
        """
        self.logger.info("Starting comprehensive model evaluation")

        results = {}

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        y_train_prob = model.predict_proba(X_train) if hasattr(model, "predict_proba") else None
        y_test_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        # Calculate metrics
        results["train_metrics"] = self.calculate_metrics(y_train, y_train_pred, y_train_prob)
        results["test_metrics"] = self.calculate_metrics(y_test, y_test_pred, y_test_prob)

        # Classification reports
        results["train_report"] = self.generate_classification_report(y_train, y_train_pred)
        results["test_report"] = self.generate_classification_report(y_test, y_test_pred)

        # Setup save directory
        if save_dir and self.config.get("evaluation.save_plots", True):
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Plot confusion matrices
            self.plot_confusion_matrix(
                y_test,
                y_test_pred,
                save_path=save_dir / "confusion_matrix.png",
                title="Test Set Confusion Matrix",
            )

            # Plot ROC curves
            if y_test_prob is not None:
                self.plot_train_test_roc(
                    y_train,
                    y_train_prob,
                    y_test,
                    y_test_prob,
                    save_path=save_dir / "roc_curves.png",
                )

            # Calculate and plot permutation importance
            if self.config.get("evaluation.permutation_importance.enabled", True):
                importance_df = self.calculate_permutation_importance(
                    model, X_test, y_test, feature_names
                )
                results["feature_importance"] = importance_df

                self.plot_feature_importance(
                    importance_df, save_path=save_dir / "feature_importance.png"
                )

        self.logger.info("Comprehensive evaluation complete")

        return results
