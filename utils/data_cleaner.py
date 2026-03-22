"""
data_cleaner.py — Object-Oriented Data Cleaning Utility

Provides a DataCleaner class that handles:
  1. Missing values   (drop or impute with mode/median/mean)
  2. Duplicate rows   (detect and remove)
  3. Outliers         (IQR-based detection and removal for numeric columns)

Usage:
    from utils.data_cleaner import DataCleaner

    cleaner = DataCleaner(df)
    cleaner.report()
    cleaned_df = cleaner.clean()
"""

import pandas as pd
import numpy as np


class DataCleaner:
    """Encapsulates data-cleaning operations on a pandas DataFrame."""

    def __init__(self, df, name="Dataset"):
        self.df = df.copy()
        self.name = name
        self._log = []

    # ── Reporting ─────────────────────────────────────────────────────────────

    def report(self):
        """Print a concise summary of data quality issues."""
        print(f"\n{'=' * 60}")
        print(f"  Report: {self.name}")
        print(f"{'=' * 60}")
        print(f"  Shape           : {self.df.shape[0]:,} rows × {self.df.shape[1]} cols")
        print(f"  Duplicates      : {self.df.duplicated().sum():,}")

        missing = self.df.isnull().sum()
        total_missing = missing.sum()
        print(f"  Total missing   : {total_missing:,}")
        if total_missing > 0:
            for col in missing[missing > 0].index:
                pct = missing[col] / len(self.df) * 100
                print(f"    • {col:30s} {missing[col]:>5,} ({pct:.1f}%)")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        for col in numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            if n_outliers > 0:
                outlier_count += n_outliers
                print(f"  Outliers in {col:20s}: {n_outliers:>5,}")
        if outlier_count == 0:
            print(f"  Outliers        : 0")
        print(f"{'=' * 60}\n")

    def get_log(self):
        """Return list of cleaning actions performed."""
        return list(self._log)

    # ── Missing Values ────────────────────────────────────────────────────────

    def handle_missing(self, strategy="auto", fill_value=None):
        """
        Handle missing values.

        Args:
            strategy: "auto" (mode for cat, median for num), "drop",
                      "mean", "median", "mode", or "value".
            fill_value: Constant when strategy="value".
        """
        before = self.df.isnull().sum().sum()
        if before == 0:
            self._log.append("Missing values: none found, skipped.")
            return self

        if strategy == "drop":
            self.df.dropna(inplace=True)
            self.df.reset_index(drop=True, inplace=True)

        elif strategy == "value":
            self.df.fillna(fill_value, inplace=True)

        elif strategy in ("auto", "mean", "median", "mode"):
            for col in self.df.columns:
                if self.df[col].isnull().sum() == 0:
                    continue
                if self.df[col].dtype in (np.number, "int64", "float64"):
                    if strategy in ("auto", "median"):
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    elif strategy == "mean":
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                    else:
                        self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        after = self.df.isnull().sum().sum()
        self._log.append(
            f"Missing values: handled {before - after} cells (strategy='{strategy}')."
        )
        return self

    # ── Duplicates ────────────────────────────────────────────────────────────

    def remove_duplicates(self, subset=None, keep="first"):
        """Remove duplicate rows."""
        before = len(self.df)
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        removed = before - len(self.df)
        self._log.append(f"Duplicates: removed {removed:,} duplicate rows.")
        return self

    # ── Outliers ──────────────────────────────────────────────────────────────

    def remove_outliers(self, columns=None, method="iqr", threshold=1.5):
        """Remove outlier rows using IQR method on numeric columns."""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        before = len(self.df)
        mask = pd.Series(True, index=self.df.index)

        if method == "iqr":
            for col in columns:
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                mask &= (self.df[col] >= lower) & (self.df[col] <= upper)

        self.df = self.df[mask].reset_index(drop=True)
        removed = before - len(self.df)
        self._log.append(
            f"Outliers: removed {removed:,} rows (method='{method}', threshold={threshold})."
        )
        return self

    # ── Full Pipeline ─────────────────────────────────────────────────────────

    def clean(self, missing_strategy="auto", remove_dupes=True,
              remove_outliers=True, outlier_columns=None, outlier_threshold=1.5):
        """
        Run the full cleaning pipeline:
            1. Handle missing values
            2. Remove duplicates
            3. Remove outliers

        Returns:
            pd.DataFrame: The cleaned dataframe.
        """
        self.handle_missing(strategy=missing_strategy)
        if remove_dupes:
            self.remove_duplicates()
        if remove_outliers:
            self.remove_outliers(columns=outlier_columns, threshold=outlier_threshold)
        return self.df
