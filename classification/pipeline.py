"""
classification/pipeline.py — Classification Pipeline

End-to-end workflow for the Online Course Engagement dataset:
  Load → EDA → Clean → Feature-engineer → Train → Evaluate → Tune → Test → Save
"""

import os
import sys
import joblib
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from utils.data_cleaner import DataCleaner
from utils.eda import EDAAnalyzer
from utils.feature_engineer import FeatureEngineer
from utils.model_trainer import ModelTrainer
from utils.model_evaluator import ModelEvaluator


class ClassificationPipeline:
    """Orchestrates the full classification ML workflow."""

    TARGET = "CourseCompletion"
    DROP_COLS = ["UserID"]
    OUTPUT_BASE = "outputs/classification"

    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.feature_engineer = FeatureEngineer(name="Classification")
        self.trainer = ModelTrainer(name="Classification")
        self.evaluator = ModelEvaluator(
            name="Classification",
            output_dir=os.path.join(self.OUTPUT_BASE, "evaluation"),
        )

    # ── 1. Load ───────────────────────────────────────────────────────────────

    def load_data(self):
        """Load train and test CSVs."""
        self.train_raw = pd.read_csv(self.train_path)
        self.test_raw = pd.read_csv(self.test_path)
        print(f"  📂  Loaded train={self.train_raw.shape}, test={self.test_raw.shape}")

    # ── 2. EDA ────────────────────────────────────────────────────────────────

    def run_eda(self):
        """Run exploratory data analysis on raw training data."""
        eda = EDAAnalyzer(
            self.train_raw, name="Classification (raw)",
            output_dir=os.path.join(self.OUTPUT_BASE, "eda"),
        )
        eda.run_all(target_col=self.TARGET)

    # ── 3. Clean ──────────────────────────────────────────────────────────────

    def clean_data(self):
        """Clean train and test sets."""
        for label, attr in [("train", "train_raw"), ("test", "test_raw")]:
            df = getattr(self, attr)
            cleaner = DataCleaner(df, name=f"Classification {label}")
            cleaned = cleaner.clean(
                missing_strategy="auto",
                remove_dupes=True,
                remove_outliers=True,
            )
            setattr(self, f"{label}_clean", cleaned)
            for entry in cleaner.get_log():
                print(f"    {entry}")

        # Drop unnecessary columns
        for attr in ("train_clean", "test_clean"):
            df = getattr(self, attr)
            cols_to_drop = [c for c in self.DROP_COLS if c in df.columns]
            setattr(self, attr, df.drop(columns=cols_to_drop))

    # ── 4. Feature Engineering ────────────────────────────────────────────────

    def engineer_features(self):
        """Transform train & test into model-ready features."""
        self.X_train, self.y_train = self.feature_engineer.transform(
            self.train_clean, target_col=self.TARGET,
            problem_type="classification",
        )
        self.X_test, self.y_test = self.feature_engineer.transform(
            self.test_clean, target_col=self.TARGET,
            problem_type="classification",
        )
        print(f"  🔧  Features: X_train={self.X_train.shape}, X_test={self.X_test.shape}")

    # ── 5. Train & Compare ────────────────────────────────────────────────────

    def train_models(self):
        """Train and cross-validate multiple classifiers."""
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest":      RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting":  GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        self.results = self.trainer.train_multiple(
            self.X_train, self.y_train, models, cv=5, scoring="accuracy",
        )
        print(self.trainer.summary_table().to_string(index=False))

    # ── 6. Evaluate Best Model ────────────────────────────────────────────────

    def evaluate(self):
        """Evaluate the best model on the test set."""
        model = self.trainer.best_model
        y_pred = model.predict(self.X_test)

        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(self.X_test)[:, 1]

        self.test_metrics = self.evaluator.evaluate_classification(
            self.y_test, y_pred, y_proba,
        )
        self.evaluator.plot_confusion_matrix(self.y_test, y_pred, labels=[0, 1])
        if y_proba is not None:
            self.evaluator.plot_roc_curve(self.y_test, y_proba)

    # ── 7. Hyperparameter Tuning ──────────────────────────────────────────────

    def tune(self):
        """Tune the best model's hyperparameters."""
        if self.trainer.best_model_name == "Random Forest":
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5],
            }
            base = RandomForestClassifier(random_state=42)
        elif self.trainer.best_model_name == "Gradient Boosting":
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            }
            base = GradientBoostingClassifier(random_state=42)
        else:
            param_grid = {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs", "liblinear"],
            }
            base = LogisticRegression(max_iter=1000, random_state=42)

        self.tuned_model = self.trainer.tune_hyperparameters(
            self.X_train, self.y_train, base, param_grid,
            cv=5, scoring="accuracy",
        )

        # Re-evaluate with tuned model
        print("  📐  Evaluating tuned model on test set:")
        y_pred = self.tuned_model.predict(self.X_test)
        y_proba = None
        if hasattr(self.tuned_model, "predict_proba"):
            y_proba = self.tuned_model.predict_proba(self.X_test)[:, 1]
        self.evaluator.evaluate_classification(self.y_test, y_pred, y_proba)

    # ── 8. Save ───────────────────────────────────────────────────────────────

    def save(self):
        """Persist the final model and feature engineer."""
        model_dir = os.path.join(self.OUTPUT_BASE, "models")
        os.makedirs(model_dir, exist_ok=True)
        
        self.trainer.save_model(
            self.tuned_model,
            os.path.join(model_dir, "classification_model.pkl"),
        )
        joblib.dump(self.feature_engineer, os.path.join(model_dir, "feature_engineer.pkl"))
        print(f"  💾  Feature engineer saved → {model_dir}/feature_engineer.pkl")

    # ── Full run ──────────────────────────────────────────────────────────────

    def run(self):
        """Execute the complete pipeline."""
        print("\n" + "=" * 60)
        print("  CLASSIFICATION PIPELINE")
        print("=" * 60)
        self.load_data()
        self.run_eda()
        self.clean_data()
        self.engineer_features()
        self.train_models()
        self.evaluate()
        self.tune()
        self.save()
        print("  ✅  Classification pipeline complete.\n")
