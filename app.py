import streamlit as st

st.set_page_config(
    page_title="ShopSmart | Purchase Intent Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.main-header h1 { color: #ffffff; font-size: 2.4rem; margin: 0; font-weight: 700; }
.main-header p  { color: #a8d8ea; font-size: 1rem; margin: 0.4rem 0 0; }

.metric-card {
    background: #1a1a2e;
    border: 1px solid #16213e;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    color: white;
}
.metric-card .metric-value { font-size: 2rem; font-weight: 700; color: #00d2ff; }
.metric-card .metric-label { font-size: 0.8rem; color: #8892b0; margin-top: 4px; }

.predict-btn > button {
    background: linear-gradient(135deg, #00d2ff, #3a7bd5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
}
.result-buy {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    border-radius: 14px; padding: 2rem; text-align: center; color: white;
}
.result-nobuy {
    background: linear-gradient(135deg, #c0392b, #e74c3c);
    border-radius: 14px; padding: 2rem; text-align: center; color: white;
}
.result-title { font-size: 1.8rem; font-weight: 700; margin: 0; }
.result-sub   { font-size: 1rem; opacity: 0.9; margin-top: 0.5rem; }

.insight-card {
    background: #0f3460;
    border-left: 4px solid #00d2ff;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    color: white;
}
.sidebar-brand {
    background: linear-gradient(135deg, #0f2027, #2c5364);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    color: white;
    margin-bottom: 1rem;
}

section[data-testid="stSidebar"] { background: #0a0e1a; }
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div style="font-size:2rem">🛒</div>
        <div style="font-weight:700;font-size:1.1rem">ShopSmart</div>
        <div style="font-size:0.75rem;opacity:0.7">Purchase Intent Predictor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📍 Navigate")
    page = st.radio(
        "",
        ["🏠 Home", "🔮 Predict Intent", "📊 EDA & Insights", "📈 Model Performance"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 🧠 About the Model")
    st.markdown("""
    - **Algorithm:** Decision Tree  
    - **Tuning:** GridSearchCV  
    - **Accuracy:** 84.71%  
    - **Recall:** 83.3%  
    - **Dataset:** 12,329 sessions  
    """)
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;font-size:0.8rem;opacity:0.6">
        Built by <b>@techwithronak</b><br>
        <a href="https://github.com/ronakrajput8882" style="color:#00d2ff">GitHub</a> ·
        <a href="https://www.linkedin.com/in/ronaksinh-rajput8882/" style="color:#00d2ff">LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)

# ── Page Router ───────────────────────────────────────────────────────────────
if page == "🏠 Home":
    from views.home import show; show()
elif page == "🔮 Predict Intent":
    from views.predict import show; show()
elif page == "📊 EDA & Insights":
    from views.eda import show; show()
elif page == "📈 Model Performance":
    from views.performance import show; show()
