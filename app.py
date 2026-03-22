"""
app.py — Streamlit Deployment Interface

Interactive dashboard for both ML models:
  • EDA visualisations
  • Model evaluation results
  • Live predictions

Run:
    streamlit run app.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from utils.feature_engineer import FeatureEngineer

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ML Assignment — Dashboard",
    page_icon="🤖",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🤖 ML Dashboard")
problem = st.sidebar.radio(
    "Select Problem",
    ["Classification", "Regression"],
    index=0,
)
section = st.sidebar.radio(
    "Section",
    ["📊 EDA", "📈 Evaluation", "🔮 Predict"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("ML Assignment — AAU AI Year 1, Semester 2")
st.sidebar.caption("By: Fuad Nuri")
st.sidebar.caption("ID: GSE/1569/18")

# ── Helpers ───────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(ROOT, "outputs")


def show_image(path, caption=""):
    """Display an image if it exists."""
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Plot not found: {path}")


def load_artifacts(problem_type):
    """Load saved model and feature engineer."""
    base_path = os.path.join(OUTPUT_DIR, problem_type.lower(), "models")
    model_path = os.path.join(base_path, f"{problem_type.lower()}_model.pkl")
    fe_path = os.path.join(base_path, "feature_engineer.pkl")
    
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    fe = joblib.load(fe_path) if os.path.exists(fe_path) else None
    return model, fe


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

if problem == "Classification":
    st.title("🎓 Online Course Engagement — Classification")
    st.markdown("Predict whether a student will **complete** an online course.")
    eda_dir = os.path.join(OUTPUT_DIR, "classification", "eda")
    eval_dir = os.path.join(OUTPUT_DIR, "classification", "evaluation")

    # ── EDA ────────────────────────────────────────────────────────────────
    if section == "📊 EDA":
        st.header("Exploratory Data Analysis")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Distributions", "Correlations", "Target", "Categoricals"
        ])
        with tab1:
            show_image(os.path.join(eda_dir, "distributions.png"),
                       "Numeric Feature Distributions")
        with tab2:
            show_image(os.path.join(eda_dir, "correlations.png"),
                       "Feature Correlation Heatmap")
        with tab3:
            show_image(os.path.join(eda_dir, "target_distribution.png"),
                       "Target Distribution (CourseCompletion)")
        with tab4:
            show_image(os.path.join(eda_dir, "categorical_counts.png"),
                       "Categorical Feature Counts")

        # Data preview
        st.subheader("📋 Data Preview")
        train_path = os.path.join(ROOT, "datasets", "classification", "train.csv")
        if os.path.exists(train_path):
            df = pd.read_csv(train_path)
            st.dataframe(df.head(20), use_container_width=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", f"{len(df):,}")
            col2.metric("Features", df.shape[1])
            col3.metric("Duplicates", df.duplicated().sum())

    # ── Evaluation ─────────────────────────────────────────────────────────
    elif section == "📈 Evaluation":
        st.header("Model Evaluation")

        col1, col2 = st.columns(2)
        with col1:
            show_image(os.path.join(eval_dir, "confusion_matrix.png"),
                       "Confusion Matrix")
        with col2:
            show_image(os.path.join(eval_dir, "roc_curve.png"),
                       "ROC Curve")

        st.subheader("Model Comparison")
        comparison = pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest", "Gradient Boosting"],
            "CV Accuracy": [0.7994, 0.9587, 0.9585],
        })
        st.dataframe(comparison.style.highlight_max(subset=["CV Accuracy"]),
                     use_container_width=True)

        st.subheader("Test Set Metrics (Best Model — Random Forest)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", "95.5%")
        m2.metric("Precision", "95.5%")
        m3.metric("Recall", "95.5%")
        m4.metric("ROC-AUC", "96.25%")

    # ── Predict ────────────────────────────────────────────────────────────
    elif section == "🔮 Predict":
        st.header("Make a Prediction")
        model, fe = load_artifacts("classification")
        if model is None or fe is None:
            st.error("Model or Feature Engineer not found. Run `python main.py` first.")
        else:
            st.markdown("Fill in the student's course engagement data:")

            col1, col2 = st.columns(2)
            with col1:
                course_category = st.selectbox(
                    "Course Category",
                    ["Arts", "Business", "Health", "Programming", "Science"],
                )
                time_spent = st.slider("Time Spent on Course (hrs)", 0.0, 100.0, 50.0)
                videos_watched = st.slider("Videos Watched", 0, 20, 10)
                quizzes_taken = st.slider("Quizzes Taken", 0, 10, 5)

            with col2:
                quiz_scores = st.slider("Quiz Scores", 50.0, 100.0, 75.0)
                completion_rate = st.slider("Completion Rate (%)", 0.0, 100.0, 50.0)
                device_type = st.selectbox("Device Type", [0, 1],
                                           format_func=lambda x: "Mobile" if x == 0 else "Desktop")

            if st.button("🔮 Predict Completion", use_container_width=True):
                input_df = pd.DataFrame([{
                    "CourseCategory": course_category,
                    "TimeSpentOnCourse": time_spent,
                    "NumberOfVideosWatched": videos_watched,
                    "NumberOfQuizzesTaken": quizzes_taken,
                    "QuizScores": quiz_scores,
                    "CompletionRate": completion_rate,
                    "DeviceType": device_type,
                }])

                try:
                    # Apply pre-processing using fitted FeatureEngineer
                    X = fe.create_classification_features(input_df)
                    X = fe.encode_categoricals(X)
                    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                    X = fe.scale_numerics(X, columns=num_cols)

                    pred = model.predict(X)[0]
                    proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None

                    if pred == 1:
                        st.success("✅ **Prediction: Will Complete the Course**")
                    else:
                        st.error("❌ **Prediction: Will NOT Complete the Course**")

                    if proba is not None:
                        st.progress(float(proba[1]))
                        st.caption(f"Completion probability: {proba[1]:.1%}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

elif problem == "Regression":
    st.title("📚 Student Performance — Regression")
    st.markdown("Predict a student's **exam score** based on study habits and environment.")
    eda_dir = os.path.join(OUTPUT_DIR, "regression", "eda")
    eval_dir = os.path.join(OUTPUT_DIR, "regression", "evaluation")

    # ── EDA ────────────────────────────────────────────────────────────────
    if section == "📊 EDA":
        st.header("Exploratory Data Analysis")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Distributions", "Correlations", "Target", "Categoricals"
        ])
        with tab1:
            show_image(os.path.join(eda_dir, "distributions.png"),
                       "Numeric Feature Distributions")
        with tab2:
            show_image(os.path.join(eda_dir, "correlations.png"),
                       "Feature Correlation Heatmap")
        with tab3:
            show_image(os.path.join(eda_dir, "target_distribution.png"),
                       "Target Distribution (Exam_Score)")
        with tab4:
            show_image(os.path.join(eda_dir, "categorical_counts.png"),
                       "Categorical Feature Counts")

        # Data preview
        st.subheader("📋 Data Preview")
        train_path = os.path.join(ROOT, "datasets", "regression", "train.csv")
        if os.path.exists(train_path):
            df = pd.read_csv(train_path)
            st.dataframe(df.head(20), use_container_width=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", f"{len(df):,}")
            col2.metric("Features", df.shape[1])
            col3.metric("Missing Values", df.isnull().sum().sum())

    # ── Evaluation ─────────────────────────────────────────────────────────
    elif section == "📈 Evaluation":
        st.header("Model Evaluation")

        col1, col2 = st.columns(2)
        with col1:
            show_image(os.path.join(eval_dir, "actual_vs_predicted.png"),
                       "Actual vs Predicted")
        with col2:
            show_image(os.path.join(eval_dir, "residuals.png"),
                       "Residual Analysis")

        st.subheader("Model Comparison")
        comparison = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest", "Gradient Boosting"],
            "CV R²": [0.8854, 0.8682, 0.9390],
        })
        st.dataframe(comparison.style.highlight_max(subset=["CV R²"]),
                     use_container_width=True)

        st.subheader("Test Set Metrics (Best Model — Gradient Boosting, Tuned)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²", "0.975")
        m2.metric("RMSE", "0.506")
        m3.metric("MAE", "0.404")
        m4.metric("MSE", "0.256")

    # ── Predict ────────────────────────────────────────────────────────────
    elif section == "🔮 Predict":
        st.header("Make a Prediction")
        model, fe = load_artifacts("regression")
        if model is None or fe is None:
            st.error("Model or Feature Engineer not found. Run `python main.py` first.")
        else:
            st.markdown("Fill in the student profile:")

            col1, col2, col3 = st.columns(3)
            with col1:
                hours_studied = st.slider("Hours Studied", 1, 44, 20)
                attendance = st.slider("Attendance (%)", 60, 100, 80)
                sleep_hours = st.slider("Sleep Hours", 4, 10, 7)
                previous_scores = st.slider("Previous Scores", 50, 100, 75)
                tutoring = st.slider("Tutoring Sessions", 0, 8, 1)
                physical = st.slider("Physical Activity (hrs/wk)", 0, 6, 3)
                learning_dis = st.selectbox("Learning Disabilities", ["No", "Yes"])

            with col2:
                parental_inv = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
                access_res = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
                motivation = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
                family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
                teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
                peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
                parental_edu = st.selectbox("Parental Education", ["High School", "College", "Postgraduate"])

            with col3:
                extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"])
                internet = st.selectbox("Internet Access", ["No", "Yes"])
                school_type = st.selectbox("School Type", ["Public", "Private"])
                distance = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
                gender = st.selectbox("Gender", ["Male", "Female"])

            if st.button("🔮 Predict Exam Score", use_container_width=True):
                input_df = pd.DataFrame([{
                    "Hours_Studied": hours_studied,
                    "Attendance": attendance,
                    "Parental_Involvement": parental_inv,
                    "Access_to_Resources": access_res,
                    "Extracurricular_Activities": extracurricular,
                    "Sleep_Hours": sleep_hours,
                    "Previous_Scores": previous_scores,
                    "Motivation_Level": motivation,
                    "Internet_Access": internet,
                    "Tutoring_Sessions": tutoring,
                    "Family_Income": family_income,
                    "Teacher_Quality": teacher_quality,
                    "School_Type": school_type,
                    "Peer_Influence": peer_influence,
                    "Physical_Activity": physical,
                    "Learning_Disabilities": learning_dis,
                    "Parental_Education_Level": parental_edu,
                    "Distance_from_Home": distance,
                    "Gender": gender,
                }])

                try:
                    # Apply pre-processing using fitted FeatureEngineer
                    X = fe.create_regression_features(input_df)
                    X = fe.encode_categoricals(X)
                    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                    X = fe.scale_numerics(X, columns=num_cols)

                    pred = model.predict(X)[0]
                    st.success(f"📝 **Predicted Exam Score: {pred:.1f}**")

                    # Visual gauge
                    score_pct = max(0, min(100, (pred - 50) / 50))
                    st.progress(float(score_pct))

                    if pred >= 75:
                        st.balloons()
                        st.caption("🌟 Excellent predicted performance!")
                    elif pred >= 65:
                        st.caption("👍 Good predicted performance.")
                    else:
                        st.caption("📖 Consider additional study support.")
                except Exception as e:
                    st.error(f"Prediction error: {e}")
