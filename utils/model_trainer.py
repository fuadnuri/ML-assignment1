"""
model_trainer.py — Model Training & Tuning

Provides a ModelTrainer class that trains, compares,
tunes, and persists ML models.
"""

import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score


class ModelTrainer:
    """Train, compare, tune, and save ML models."""

    def __init__(self, name="Model"):
        self.name = name
        self.best_model = None
        self.best_model_name = None
        self.results = {}

    # ── Single model ──────────────────────────────────────────────────────────

    def train(self, X, y, model, model_name="model"):
        """
        Fit a single model.

        Args:
            X (pd.DataFrame | np.ndarray): Features.
            y (pd.Series | np.ndarray): Target.
            model: A scikit-learn estimator.
            model_name (str): Label for display.

        Returns:
            The fitted model.
        """
        print(f"  🏋️  Training {model_name}...")
        start = time.time()
        model.fit(X, y)
        elapsed = time.time() - start
        print(f"     Done in {elapsed:.2f}s")
        return model

    # ── Compare multiple models ───────────────────────────────────────────────

    def train_multiple(self, X, y, models_dict, cv=5, scoring=None):
        """
        Train and cross-validate several models, picking the best.

        Args:
            X: Features.
            y: Target.
            models_dict (dict): {name: estimator} pairs.
            cv (int): Number of cross-validation folds.
            scoring (str | None): Scoring metric (e.g. 'accuracy', 'r2').

        Returns:
            dict: {name: {'model': fitted_model, 'cv_mean': float, 'cv_std': float}}
        """
        self.results = {}
        print(f"\n{'─' * 50}")
        print(f"  Comparing models ({self.name})")
        print(f"{'─' * 50}")

        for name, model in models_dict.items():
            print(f"\n  🏋️  {name}")
            start = time.time()

            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            model.fit(X, y)

            elapsed = time.time() - start
            self.results[name] = {
                "model": model,
                "cv_mean": scores.mean(),
                "cv_std": scores.std(),
                "time": elapsed,
            }
            print(f"     CV Score: {scores.mean():.4f} ± {scores.std():.4f}  ({elapsed:.2f}s)")

        # Pick the best
        best_name = max(self.results, key=lambda k: self.results[k]["cv_mean"])
        self.best_model_name = best_name
        self.best_model = self.results[best_name]["model"]
        print(f"\n  🏆  Best model: {best_name} "
              f"(CV={self.results[best_name]['cv_mean']:.4f})")
        print(f"{'─' * 50}\n")

        return self.results

    # ── Summary table ─────────────────────────────────────────────────────────

    def summary_table(self):
        """Return a DataFrame comparing all trained models."""
        rows = []
        for name, info in self.results.items():
            rows.append({
                "Model": name,
                "CV Mean": round(info["cv_mean"], 4),
                "CV Std": round(info["cv_std"], 4),
                "Time (s)": round(info["time"], 2),
            })
        return pd.DataFrame(rows).sort_values("CV Mean", ascending=False)

    # ── Hyperparameter tuning ─────────────────────────────────────────────────

    def tune_hyperparameters(self, X, y, model, param_grid, cv=5, scoring=None):
        """
        Grid-search for the best hyperparameters.

        Args:
            X: Features.
            y: Target.
            model: Base estimator.
            param_grid (dict): Parameter grid.
            cv (int): Cross-validation folds.
            scoring (str | None): Scoring metric.

        Returns:
            The best estimator found by GridSearchCV.
        """
        print(f"\n  🔧  Tuning hyperparameters...")
        grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring,
                            n_jobs=-1, verbose=0)
        grid.fit(X, y)

        self.best_model = grid.best_estimator_
        print(f"     Best params : {grid.best_params_}")
        print(f"     Best score  : {grid.best_score_:.4f}\n")
        return grid.best_estimator_

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_model(self, model, path):
        """Save a model to disk with joblib."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        print(f"  💾  Model saved → {path}")

    @staticmethod
    def save_metrics(path, metrics_dict):
        """Save metrics to a JSON file."""
        import json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"  📊  Metrics saved → {path}")

    @staticmethod
    def load_model(path):
        """Load a model from disk."""
        return joblib.load(path)
