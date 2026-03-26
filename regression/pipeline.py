"""
regression/pipeline.py — Regression Pipeline

End-to-end workflow for the Student Performance Factors dataset:
  Load → EDA → Clean → Feature-engineer → Train → Evaluate → Tune → Test → Save
"""

import os
import sys
import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from utils.data_cleaner import DataCleaner
from utils.eda import EDAAnalyzer
from utils.feature_engineer import FeatureEngineer
from utils.model_trainer import ModelTrainer
from utils.model_evaluator import ModelEvaluator


class RegressionPipeline:
    """Orchestrates the full regression ML workflow."""

    TARGET = "Exam_Score"
    OUTPUT_BASE = "outputs/regression"

    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.feature_engineer = FeatureEngineer(name="Regression")
        self.trainer = ModelTrainer(name="Regression")
        self.evaluator = ModelEvaluator(
            name="Regression",
            output_dir=os.path.join(self.OUTPUT_BASE, "evaluation"),
        )

    # ── 1. Load ───────────────────────────────────────────────────────────────

    def load_data(self):
        self.train_raw = pd.read_csv(self.train_path)
        self.test_raw = pd.read_csv(self.test_path)
        print(f"  📂  Loaded train={self.train_raw.shape}, test={self.test_raw.shape}")

    # ── 2. EDA ────────────────────────────────────────────────────────────────

    def run_eda(self):
        eda = EDAAnalyzer(
            self.train_raw, name="Regression (raw)",
            output_dir=os.path.join(self.OUTPUT_BASE, "eda"),
        )
        eda.run_all(target_col=self.TARGET)

    # ── 3. Clean ──────────────────────────────────────────────────────────────

    def clean_data(self):
        for label, attr in [("train", "train_raw"), ("test", "test_raw")]:
            df = getattr(self, attr)
            cleaner = DataCleaner(df, name=f"Regression {label}")
            cleaned = cleaner.clean(
                missing_strategy="auto",
                remove_dupes=True,
                remove_outliers=True,
            )
            setattr(self, f"{label}_clean", cleaned)
            for entry in cleaner.get_log():
                print(f"    {entry}")

    # ── 4. Feature Engineering ────────────────────────────────────────────────

    def engineer_features(self):
        self.X_train, self.y_train = self.feature_engineer.transform(
            self.train_clean, target_col=self.TARGET,
            problem_type="regression",
        )
        self.X_test, self.y_test = self.feature_engineer.transform(
            self.test_clean, target_col=self.TARGET,
            problem_type="regression",
        )
        print(f"  🔧  Features: X_train={self.X_train.shape}, X_test={self.X_test.shape}")

    # ── 5. Train & Compare ────────────────────────────────────────────────────

    def train_models(self):
        models = {
            "Linear Regression":    LinearRegression(),
            "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting":    GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        self.results = self.trainer.train_multiple(
            self.X_train, self.y_train, models, cv=5, scoring="r2",
        )
        print(self.trainer.summary_table().to_string(index=False))

    # ── 6. Evaluate Best Model ────────────────────────────────────────────────

    def evaluate(self):
        model = self.trainer.best_model
        y_pred = model.predict(self.X_test)
        self.test_metrics = self.evaluator.evaluate_regression(self.y_test, y_pred)
        self.evaluator.plot_residuals(self.y_test, y_pred)
        self.evaluator.plot_actual_vs_predicted(self.y_test, y_pred)

    # ── 7. Hyperparameter Tuning ──────────────────────────────────────────────

    def tune(self):
        if self.trainer.best_model_name == "Random Forest":
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5],
            }
            base = RandomForestRegressor(random_state=42)
        elif self.trainer.best_model_name == "Gradient Boosting":
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            }
            base = GradientBoostingRegressor(random_state=42)
        else:
            # LinearRegression has no useful hyperparams to tune
            print("  ℹ️  Linear Regression selected — no hyperparameter tuning needed.")
            self.tuned_model = self.trainer.best_model
            return

        self.tuned_model = self.trainer.tune_hyperparameters(
            self.X_train, self.y_train, base, param_grid,
            cv=5, scoring="r2",
        )

        # Re-evaluate with tuned model
        print("  📐  Evaluating tuned model on test set:")
        y_pred = self.tuned_model.predict(self.X_test)
        self.test_metrics = self.evaluator.evaluate_regression(self.y_test, y_pred)

    # ── 8. Save ───────────────────────────────────────────────────────────────

    def save(self):
        """Persist the final model, feature engineer, and metrics."""
        model_dir = os.path.join(self.OUTPUT_BASE, "models")
        os.makedirs(model_dir, exist_ok=True)

        self.trainer.save_model(
            self.tuned_model,
            os.path.join(model_dir, "regression_model.pkl"),
        )
        joblib.dump(self.feature_engineer, os.path.join(model_dir, "feature_engineer.pkl"))
        print(f"  💾  Feature engineer saved → {model_dir}/feature_engineer.pkl")

        # Save metrics
        metrics = {
            "best_model": self.trainer.best_model_name,
            "cv_results": [
                {
                    "model": name,
                    "cv_mean": info["cv_mean"],
                    "cv_std": info["cv_std"]
                }
                for name, info in self.trainer.results.items()
            ],
            "test_metrics": self.test_metrics
        }
        self.trainer.save_metrics(
            os.path.join(self.OUTPUT_BASE, "metrics.json"),
            metrics
        )

    # ── Full run ──────────────────────────────────────────────────────────────

    def run(self):
        print("\n" + "=" * 60)
        print("  REGRESSION PIPELINE")
        print("=" * 60)
        self.load_data()
        self.run_eda()
        self.clean_data()
        self.engineer_features()
        self.train_models()
        self.evaluate()
        self.tune()
        self.save()
        print("  ✅  Regression pipeline complete.\n")
