"""
feature_engineer.py — Feature Engineering

Provides a FeatureEngineer class that encodes categoricals,
scales numerics, and creates derived features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


class FeatureEngineer:
    """Transform raw dataframes into model-ready features."""

    def __init__(self, name="Dataset"):
        self.name = name
        self._label_encoders = {}
        self._scaler = None
        self._numeric_cols = []
        self._fitted = False

    # ── Categorical Encoding ──────────────────────────────────────────────────

    def encode_categoricals(self, df, columns=None):
        """
        Label-encode categorical columns.

        Args:
            df (pd.DataFrame): Input dataframe.
            columns (list[str] | None): Columns to encode.
                If None, all object/category columns are encoded.

        Returns:
            pd.DataFrame: Dataframe with encoded columns.
        """
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in columns:
            if col not in self._label_encoders:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._label_encoders[col] = le
            else:
                le = self._label_encoders[col]
                # Handle unseen labels gracefully
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df[col] = le.transform(df[col])
        return df

    # ── Numeric Scaling ───────────────────────────────────────────────────────

    def scale_numerics(self, df, columns=None, method="standard"):
        """
        Scale numeric columns.

        Args:
            df (pd.DataFrame): Input dataframe.
            columns (list[str] | None): Columns to scale.
                If None, all numeric columns are scaled.
            method (str): "standard" (z-score) or "minmax" (0-1).

        Returns:
            pd.DataFrame: Dataframe with scaled columns.
        """
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        self._numeric_cols = columns

        if self._scaler is None:
            self._scaler = StandardScaler() if method == "standard" else MinMaxScaler()
            df[columns] = self._scaler.fit_transform(df[columns])
        else:
            df[columns] = self._scaler.transform(df[columns])

        return df

    # ── Derived Features (problem-specific) ───────────────────────────────────

    @staticmethod
    def create_classification_features(df):
        """
        Create derived features for the classification problem.
        (Online Course Engagement → CourseCompletion)
        """
        df = df.copy()

        # Engagement score: weighted combination
        df["EngagementScore"] = (
            (df["TimeSpentOnCourse"] / 100) * 0.25 +
            (df["NumberOfVideosWatched"] / 20) * 0.25 +
            (df["NumberOfQuizzesTaken"] / 10) * 0.20 +
            (df["QuizScores"] / 100) * 0.15 +
            (df["CompletionRate"] / 100) * 0.15
        )

        # Quiz efficiency
        df["QuizEfficiency"] = np.where(
            df["NumberOfVideosWatched"] > 0,
            df["NumberOfQuizzesTaken"] / df["NumberOfVideosWatched"],
            0
        )
        df["QuizEfficiency"] = df["QuizEfficiency"].clip(upper=5)

        # Low engagement flag
        df["LowEngagement"] = ((df["CompletionRate"] < 30) &
                               (df["NumberOfVideosWatched"] < 5)).astype(int)

        return df

    @staticmethod
    def create_regression_features(df):
        """
        Create derived features for the regression problem.
        (Student Performance Factors → Exam_Score)
        """
        df = df.copy()

        # Study efficiency: hours studied relative to attendance
        df["StudyEfficiency"] = np.where(
            df["Attendance"] > 0,
            df["Hours_Studied"] / df["Attendance"] * 100,
            0
        )

        # Previous score to tutoring ratio
        df["ScoreTutoringRatio"] = np.where(
            df["Tutoring_Sessions"] > 0,
            df["Previous_Scores"] / df["Tutoring_Sessions"],
            df["Previous_Scores"]
        )

        return df

    # ── Full Transform Pipeline ───────────────────────────────────────────────

    def transform(self, df, target_col, problem_type="classification",
                  scale=True, create_features=True):
        """
        Run the full feature-engineering pipeline.

        Args:
            df (pd.DataFrame): Raw dataframe.
            target_col (str): Name of the target column.
            problem_type (str): "classification" or "regression".
            scale (bool): Whether to scale numeric features.
            create_features (bool): Whether to create derived features.

        Returns:
            tuple[pd.DataFrame, pd.Series]: (X, y)
        """
        df = df.copy()

        # Create derived features first (before encoding)
        if create_features:
            if problem_type == "classification":
                df = self.create_classification_features(df)
            else:
                df = self.create_regression_features(df)

        # Encode categoricals
        df = self.encode_categoricals(df)

        # Separate target
        y = df[target_col]
        X = df.drop(columns=[target_col])

        # Scale numerics
        if scale:
            num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            X = self.scale_numerics(X, columns=num_cols)

        self._fitted = True
        return X, y
