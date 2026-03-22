"""
eda.py — Exploratory Data Analysis

Provides an EDAAnalyzer class that generates summary statistics  
and visualisations, saving all plots to an output directory.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless backend
import matplotlib.pyplot as plt
import seaborn as sns


class EDAAnalyzer:
    """Run and save exploratory-data-analysis artefacts."""

    def __init__(self, df, name="Dataset", output_dir="outputs/eda"):
        """
        Args:
            df (pd.DataFrame): The dataframe to analyse.
            name (str): Friendly label used in titles.
            output_dir (str): Where to save plots.
        """
        self.df = df.copy()
        self.name = name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _save(self, fig, filename):
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊  Saved {path}")

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self):
        """Print a textual summary of the dataframe."""
        print(f"\n{'=' * 60}")
        print(f"  EDA Summary: {self.name}")
        print(f"{'=' * 60}")
        print(f"  Shape  : {self.df.shape[0]:,} rows × {self.df.shape[1]} cols")
        print(f"  Dtypes :")
        for col in self.df.columns:
            dtype = self.df[col].dtype
            nuniq = self.df[col].nunique()
            nulls = self.df[col].isnull().sum()
            print(f"    {col:35s} {str(dtype):10s} unique={nuniq:<6} null={nulls}")
        print(f"\n  Describe (numeric):")
        print(self.df.describe().round(2).to_string())
        print(f"{'=' * 60}\n")

    # ── visualisations ────────────────────────────────────────────────────────

    def plot_distributions(self):
        """Histogram for every numeric column."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        n = len(num_cols)
        if n == 0:
            return
        cols_per_row = 3
        rows = (n + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 4 * rows))
        axes = np.array(axes).flatten()
        for i, col in enumerate(num_cols):
            sns.histplot(self.df[col], kde=True, ax=axes[i], color="steelblue")
            axes[i].set_title(col, fontsize=11)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"{self.name} — Numeric Distributions", fontsize=14, y=1.02)
        fig.tight_layout()
        self._save(fig, "distributions.png")

    def plot_correlations(self):
        """Heatmap of numeric-column correlations."""
        num_df = self.df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            return
        corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(max(8, num_df.shape[1]), max(6, num_df.shape[1] * 0.7)))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, ax=ax)
        ax.set_title(f"{self.name} — Correlation Matrix", fontsize=14)
        fig.tight_layout()
        self._save(fig, "correlations.png")

    def plot_target_distribution(self, target_col):
        """Bar chart (classification) or histogram (regression) of the target."""
        fig, ax = plt.subplots(figsize=(6, 4))
        if self.df[target_col].nunique() <= 10:
            sns.countplot(x=target_col, data=self.df, ax=ax, palette="Set2")
        else:
            sns.histplot(self.df[target_col], kde=True, ax=ax, color="coral")
        ax.set_title(f"{self.name} — Target: {target_col}", fontsize=13)
        fig.tight_layout()
        self._save(fig, "target_distribution.png")

    def plot_categorical_counts(self, max_cols=10):
        """Bar charts for categorical columns."""
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns[:max_cols]
        if len(cat_cols) == 0:
            return
        cols_per_row = 3
        rows = (len(cat_cols) + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 4 * rows))
        axes = np.array(axes).flatten()
        for i, col in enumerate(cat_cols):
            sns.countplot(x=col, data=self.df, ax=axes[i], palette="pastel",
                          order=self.df[col].value_counts().index)
            axes[i].set_title(col, fontsize=11)
            axes[i].tick_params(axis="x", rotation=30)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"{self.name} — Categorical Features", fontsize=14, y=1.02)
        fig.tight_layout()
        self._save(fig, "categorical_counts.png")

    # ── run all ───────────────────────────────────────────────────────────────

    def run_all(self, target_col):
        """Execute every analysis step."""
        self.summary()
        self.plot_distributions()
        self.plot_correlations()
        self.plot_target_distribution(target_col)
        self.plot_categorical_counts()
        print(f"  ✅  EDA complete for {self.name}\n")
