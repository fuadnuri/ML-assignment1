"""
app.py — Streamlit Deployment App
Online Learning Dropout Risk Predictor

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dropout Risk Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .risk-card-high {
        background: linear-gradient(135deg, #FF6B6B, #EE0979);
        color: white; padding: 20px; border-radius: 12px; text-align: center;
        font-size: 1.4rem; font-weight: bold; margin: 10px 0;
    }
    .risk-card-low {
        background: linear-gradient(135deg, #11998E, #38EF7D);
        color: white; padding: 20px; border-radius: 12px; text-align: center;
        font-size: 1.4rem; font-weight: bold; margin: 10px 0;
    }
    .metric-card {
        background: #F8FAFC; border: 1px solid #E2E8F0; padding: 15px;
        border-radius: 10px; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("dropout_model.pkl")
        scaler = joblib.load("scaler.pkl")
        le = joblib.load("label_encoder.pkl")
        return model, scaler, le
    except FileNotFoundError:
        return None, None, None

model, scaler, le = load_artifacts()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎓 Online Learning Dropout Risk Predictor")
st.markdown("""
> **ML Classification Assignment** — Predict whether a student is at risk of dropping out
> of an online course based on their engagement behaviour.
""")

if model is None:
    st.error(
        "⚠️ Model files not found. Please run `classification_dropout_risk.py` first "
        "to train and save the model artifacts."
    )
    st.stop()

# ── Sidebar: Model Info ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Information")
    st.info(f"""
**Model**: Gradient Boosting Classifier  
**Task**: Binary Classification  
**Target**: Dropout Risk (Yes / No)  
**Dataset**: Online Learning Engagement  
**Rows**: 9,000 students  
**Features**: 10  
    """)
    st.header("🎯 Success Criteria")
    st.success("""
- **F1-Score** ≥ 0.80  
- **ROC-AUC** ≥ 0.85  
- **Recall** ≥ 0.80  
    """)

# ── Input Form ────────────────────────────────────────────────────────────────
st.subheader("📋 Enter Student Engagement Data")
col1, col2, col3 = st.columns(3)

with col1:
    course_category = st.selectbox(
        "Course Category",
        options=["Arts", "Business", "Programming"],
        help="Category of the online course"
    )
    time_spent = st.slider(
        "Time Spent on Course (hrs)",
        min_value=0.0, max_value=120.0, value=15.0, step=0.5,
        help="Total hours the student has engaged with the course"
    )
    videos_watched = st.number_input(
        "Number of Videos Watched",
        min_value=0, max_value=300, value=10,
        help="Total video lessons watched"
    )

with col2:
    quizzes_taken = st.number_input(
        "Number of Quizzes Taken",
        min_value=0, max_value=100, value=3,
        help="Total quizzes attempted"
    )
    quiz_scores = st.slider(
        "Average Quiz Score (%)",
        min_value=0.0, max_value=100.0, value=65.0, step=0.5,
        help="Mean percentage score on quizzes"
    )
    completion_rate = st.slider(
        "Completion Rate (%)",
        min_value=0.0, max_value=100.0, value=30.0, step=1.0,
        help="Percentage of course content completed so far"
    )

with col3:
    device_type = st.radio(
        "Device Type",
        options=["Desktop (0)", "Mobile (1)"],
        help="Primary device used for learning"
    )
    st.markdown("")
    predict_btn = st.button("🔍 Predict Dropout Risk", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    # Encode inputs
    cat_enc = {"Arts": 0, "Business": 1, "Programming": 2}[course_category]
    dev_enc = 0 if "Desktop" in device_type else 1

    # Derived features
    engagement_score = (
        (time_spent / 120) * 0.25 +
        (videos_watched / 200) * 0.25 +
        (quizzes_taken / 50) * 0.20 +
        (quiz_scores / 100) * 0.15 +
        (completion_rate / 100) * 0.15
    ) * 100

    quiz_efficiency = (quizzes_taken / videos_watched) if videos_watched > 0 else 0
    quiz_efficiency = min(quiz_efficiency, 5)

    low_engagement = int(completion_rate < 30 and videos_watched < 5)

    input_data = np.array([[
        cat_enc, time_spent, videos_watched, quizzes_taken,
        quiz_scores, completion_rate, dev_enc,
        engagement_score, quiz_efficiency, low_engagement
    ]])

    # Predict
    prob = model.predict_proba(input_data)[0][1]
    prediction = int(prob >= 0.5)

    # Display result
    st.subheader("📈 Prediction Result")
    result_col, metrics_col = st.columns([1, 2])

    with result_col:
        if prediction == 1:
            st.markdown(
                f'<div class="risk-card-high">⚠️ HIGH DROPOUT RISK<br>'
                f'<span style="font-size:0.9rem">Probability: {prob:.1%}</span></div>',
                unsafe_allow_html=True
            )
            st.warning("**Recommended Actions:**\n"
                       "- Send personalised re-engagement email\n"
                       "- Offer mentor session\n"
                       "- Provide adaptive content path\n"
                       "- Set milestone reminders")
        else:
            st.markdown(
                f'<div class="risk-card-low">✅ LOW DROPOUT RISK<br>'
                f'<span style="font-size:0.9rem">Probability: {prob:.1%}</span></div>',
                unsafe_allow_html=True
            )
            st.success("Student is engaged and on track. No intervention needed.")

    with metrics_col:
        st.markdown("**📊 Engagement Snapshot**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Engagement Score", f"{engagement_score:.1f}/100")
        m2.metric("Quiz Efficiency", f"{quiz_efficiency:.2f}")
        m3.metric("Low Engagement Flag", "Yes ⚠️" if low_engagement else "No ✅")

        # Risk gauge (progress bar)
        risk_label = "Dropout Probability"
        st.markdown(f"**{risk_label}: {prob:.1%}**")
        st.progress(float(prob))

        # Factor breakdown
        st.markdown("**Key Factors:**")
        factors_df = pd.DataFrame({
            "Feature": ["Completion Rate", "Quiz Scores", "Videos Watched", "Time Spent"],
            "Value": [f"{completion_rate:.1f}%", f"{quiz_scores:.1f}%",
                      str(videos_watched), f"{time_spent:.1f} hrs"],
            "Signal": [
                "🔴 Low" if completion_rate < 30 else ("🟡 Medium" if completion_rate < 70 else "🟢 High"),
                "🔴 Low" if quiz_scores < 50 else ("🟡 Medium" if quiz_scores < 75 else "🟢 High"),
                "🔴 Low" if videos_watched < 5 else ("🟡 Medium" if videos_watched < 20 else "🟢 High"),
                "🔴 Low" if time_spent < 5 else ("🟡 Medium" if time_spent < 20 else "🟢 High"),
            ]
        })
        st.dataframe(factors_df, hide_index=True, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("ML Assignment — Classification Problem | Online Learning Engagement Dataset")