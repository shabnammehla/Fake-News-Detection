import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="🧠",
    layout="wide"
)

# ---------------- Load Model ----------------
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# ---------------- Header ----------------
st.markdown("## 🧠 AI-Powered News Authenticator")
st.markdown("### **Fake News Detector**")
st.caption("Analyze news credibility using machine-learning patterns")

st.divider()

# ---------------- Layout ----------------
left, right = st.columns([1.2, 1])

with left:
    news_text = st.text_area(
        "📰 Paste your news article",
        height=220,
        placeholder="Paste the complete news article text here..."
    )
    analyze = st.button("🔍 Analyze News", use_container_width=True)

with right:
    if analyze and news_text.strip():
        X = vectorizer.transform([news_text])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        real_conf = proba[1] * 100
        fake_conf = proba[0] * 100

        verdict = "REAL" if prediction == 1 else "FAKE"
        verdict_color = "green" if prediction == 1 else "red"

        st.subheader("🧾 Result")
        st.markdown(
            f"<h2 style='color:{verdict_color}'>{verdict}</h2>",
            unsafe_allow_html=True
        )

        # -------- Gauge Charts --------
        col_r, col_f = st.columns(2)

        with col_r:
            fig_real = go.Figure(go.Indicator(
                mode="gauge+number",
                value=real_conf,
                title={"text": "Real News Confidence"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#00c896"},
                }
            ))
            fig_real.update_layout(height=260)
            st.plotly_chart(fig_real, use_container_width=True)

        with col_f:
            fig_fake = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fake_conf,
                title={"text": "Fake News Confidence"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ff4b4b"},
                }
            ))
            fig_fake.update_layout(height=260)
            st.plotly_chart(fig_fake, use_container_width=True)

        st.info(
            "Analysis is based on linguistic patterns, tone, and structure "
            "learned from historical datasets."
        )

    elif analyze:
        st.warning("Please enter some news text to analyze.")

# ---------------- Footer ----------------
st.divider()
st.caption(
    "⚠️ This system does not verify news from live internet sources. "
    "It classifies news using trained machine-learning models."
)