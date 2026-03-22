"""
data_utils.py — Dataset Splitting Utility
Splits each dataset (classification & regression) into 80% training / 20% testing
and saves the results as separate CSV files.

Usage:
    python -m utils.data_utils          (from the project root)
    python utils/data_utils.py          (from the project root)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def train_test_split(df, test_size=0.2, random_state=None):
    """
    Split a pandas DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): The input dataframe.
        test_size (float): Proportion of the dataset for the test split (default 0.2).
        random_state (int | None): Seed for reproducibility (default None).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    if random_state is not None:
        np.random.seed(random_state)

    indices = df.index.tolist()
    np.random.shuffle(indices)

    split_idx = int(len(df) * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    return (
        df.loc[train_indices].reset_index(drop=True),
        df.loc[test_indices].reset_index(drop=True),
    )


def split_features_target(df, target_column):
    """
    Split a DataFrame into features (X) and target (y).

    Args:
        df (pd.DataFrame): The input dataframe.
        target_column (str): Name of the target column.

    Returns:
        tuple[pd.DataFrame, pd.Series]: (X, y)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


# ── Main: split both datasets and save ────────────────────────────────────────
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    DATASETS = ROOT / "datasets"
    RANDOM_STATE = 42

    splits = [
        {
            "name": "Classification (Online Course Engagement)",
            "src": DATASETS / "classification" / "online_course_engagement_data.csv",
            "dst_dir": DATASETS / "classification",
        },
        {
            "name": "Regression (Student Performance Factors)",
            "src": DATASETS / "regression" / "StudentPerformanceFactors.csv",
            "dst_dir": DATASETS / "regression",
        },
    ]

    for split in splits:
        src = split["src"]
        if not src.exists():
            print(f"⚠  Skipping '{split['name']}' — file not found: {src}")
            continue

        df = pd.read_csv(src)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

        dst = split["dst_dir"]
        dst.mkdir(parents=True, exist_ok=True)

        train_path = dst / "train.csv"
        test_path = dst / "test.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"✅ {split['name']}")
        print(f"   Original : {len(df):>6,} rows")
        print(f"   Train    : {len(train_df):>6,} rows → {train_path}")
        print(f"   Test     : {len(test_df):>6,} rows → {test_path}")
        print()
