"""
model_evaluator.py — Model Evaluation & Visualisation

Provides a ModelEvaluator class with separate methods for
classification metrics and regression metrics, plus plots.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    # classification
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve,
    # regression
    mean_squared_error, mean_absolute_error, r2_score,
)


class ModelEvaluator:
    """Evaluate ML models and generate metric reports / plots."""

    def __init__(self, name="Model", output_dir="outputs/evaluation"):
        self.name = name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def _save(self, fig, filename):
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊  Saved {path}")

    # ══════════════════════════════════════════════════════════════════════════
    #  CLASSIFICATION
    # ══════════════════════════════════════════════════════════════════════════

    def evaluate_classification(self, y_true, y_pred, y_proba=None):
        """
        Compute and print classification metrics.

        Returns:
            dict: Metric name → value.
        """
        metrics = {
            "Accuracy":  accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1-Score":  f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        if y_proba is not None:
            try:
                metrics["ROC-AUC"] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics["ROC-AUC"] = float("nan")

        print(f"\n{'─' * 50}")
        print(f"  Classification Metrics: {self.name}")
        print(f"{'─' * 50}")
        for k, v in metrics.items():
            print(f"    {k:12s}: {v:.4f}")
        print(f"{'─' * 50}")
        print("\n  Classification Report:")
        print(classification_report(y_true, y_pred))
        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, labels=None):
        """Plot and save a confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{self.name} — Confusion Matrix")
        fig.tight_layout()
        self._save(fig, "confusion_matrix.png")

    def plot_roc_curve(self, y_true, y_proba):
        """Plot and save the ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="darkorange", lw=2,
                label=f"ROC curve (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{self.name} — ROC Curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        self._save(fig, "roc_curve.png")

    # ══════════════════════════════════════════════════════════════════════════
    #  REGRESSION
    # ══════════════════════════════════════════════════════════════════════════

    def evaluate_regression(self, y_true, y_pred):
        """
        Compute and print regression metrics.

        Returns:
            dict: Metric name → value.
        """
        mse = mean_squared_error(y_true, y_pred)
        metrics = {
            "MSE":  mse,
            "RMSE": np.sqrt(mse),
            "MAE":  mean_absolute_error(y_true, y_pred),
            "R²":   r2_score(y_true, y_pred),
        }

        print(f"\n{'─' * 50}")
        print(f"  Regression Metrics: {self.name}")
        print(f"{'─' * 50}")
        for k, v in metrics.items():
            print(f"    {k:6s}: {v:.4f}")
        print(f"{'─' * 50}\n")
        return metrics

    def plot_residuals(self, y_true, y_pred):
        """Plot residual distribution."""
        residuals = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.4, color="steelblue", s=10)
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs Predicted")

        # Residual histogram
        sns.histplot(residuals, kde=True, ax=axes[1], color="coral")
        axes[1].set_title("Residual Distribution")

        fig.suptitle(f"{self.name} — Residual Analysis", fontsize=14)
        fig.tight_layout()
        self._save(fig, "residuals.png")

    def plot_actual_vs_predicted(self, y_true, y_pred):
        """Scatter plot of actual vs predicted values."""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, y_pred, alpha=0.4, color="steelblue", s=10)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", lw=1.5)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{self.name} — Actual vs Predicted")
        fig.tight_layout()
        self._save(fig, "actual_vs_predicted.png")

    # ══════════════════════════════════════════════════════════════════════════
    #  MODEL COMPARISON
    # ══════════════════════════════════════════════════════════════════════════

    def compare_models(self, comparison_dict):
        """
        Print a comparison table from a dict of
        {model_name: {metric: value, ...}}.
        """
        df = pd.DataFrame(comparison_dict).T
        df.index.name = "Model"
        print(f"\n{'─' * 50}")
        print(f"  Model Comparison: {self.name}")
        print(f"{'─' * 50}")
        print(df.round(4).to_string())
        print(f"{'─' * 50}\n")
        return df
