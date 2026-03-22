"""
main.py — ML Assignment Entry Point

Runs both the Classification and Regression pipelines end-to-end.

Usage:
    python main.py                 (run both)
    python main.py classification  (classification only)
    python main.py regression      (regression only)
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from classification.pipeline import ClassificationPipeline
from regression.pipeline import RegressionPipeline


def main():
    args = sys.argv[1:]
    run_classification = not args or "classification" in args
    run_regression = not args or "regression" in args

    # ── Classification ────────────────────────────────────────────────────
    if run_classification:
        clf_pipeline = ClassificationPipeline(
            train_path="datasets/classification/train.csv",
            test_path="datasets/classification/test.csv",
        )
        clf_pipeline.run()

    # ── Regression ────────────────────────────────────────────────────────
    if run_regression:
        reg_pipeline = RegressionPipeline(
            train_path="datasets/regression/train.csv",
            test_path="datasets/regression/test.csv",
        )
        reg_pipeline.run()

    print("\n🎉  All pipelines finished!\n")


if __name__ == "__main__":
    main()
