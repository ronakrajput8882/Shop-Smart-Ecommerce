import streamlit as st

def show():
    st.markdown("""
    <div class="main-header">
        <h1>🛒 ShopSmart Purchase Intent Predictor</h1>
        <p>Binary Classification · Decision Tree · sklearn Pipeline · GridSearchCV</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Key metrics row ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">84.71%</div>
            <div class="metric-label">Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">83.3%</div>
            <div class="metric-label">Recall</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">12,329</div>
            <div class="metric-label">Sessions</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">17</div>
            <div class="metric-label">Features</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── About + Pipeline ────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 🎯 Problem Statement")
        st.markdown("""
        E-commerce businesses lose millions every year to **cart abandonment** and missed conversions.

        This model predicts whether an online shopping session will end in a **purchase** using
        17 behavioral signals — page views, exit rates, bounce rates, session metadata, and more.

        > **Goal:** Give e-commerce teams a real-time signal to personalize UX or trigger
        interventions for high-intent visitors.
        """)

        st.markdown("### 🔄 ML Pipeline")
        st.code("""Raw CSV
  → EDA & Feature Inspection
  → Numerical / Categorical Split
  → Train/Test Split (80/20, stratified)
  → ColumnTransformer
      ├─ StandardScaler   (numerical)
      └─ OneHotEncoder    (categorical)
  → DecisionTreeClassifier
      └─ class_weight="balanced"
  → GridSearchCV (F1 scoring)
  → Evaluation""", language="text")

    with col2:
        st.markdown("### 💡 Key Insights")

        insights = [
            ("📈 PageValues", "+0.49 correlation with Revenue — strongest single predictor. Users visiting high-value pages convert far more."),
            ("📉 ExitRates", "−0.21 correlation — high exit rates are the clearest abandonment signal."),
            ("📉 BounceRates", "−0.15 correlation — single-page sessions rarely convert."),
            ("🛍️ ProductRelated", "+0.15 — deeper product browsing = higher purchase probability."),
            ("📅 SpecialDay", "−0.08 — holiday-period browsing tends to be window-shopping, not buying."),
        ]
        for title, body in insights:
            st.markdown(f"""
            <div class="insight-card">
                <b>{title}</b><br>
                <span style="font-size:0.85rem;opacity:0.85">{body}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dataset overview ────────────────────────────────────────────────────
    st.markdown("### 📂 Dataset Overview")
    col_a, col_b = st.columns(2)
    with col_a:
        import pandas as pd
        df_info = pd.DataFrame({
            "Property": ["Total Samples", "Features", "Target", "Class Imbalance", "Missing Values"],
            "Value": ["12,329", "17 (10 numerical, 7 categorical)", "Revenue (True/False → 1/0)", "~84.5% No Purchase / ~15.5% Purchase", "None"]
        })
        st.dataframe(df_info, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("**Key Features:**")
        features = {
            "PageValues": "Avg value of pages before transaction",
            "ExitRates": "Avg exit rate of pages visited",
            "BounceRates": "Avg bounce rate of visited pages",
            "ProductRelated": "Number of product pages visited",
            "Month": "Month of the session",
            "VisitorType": "New / Returning / Other",
            "Weekend": "Whether session was on weekend",
        }
        for feat, desc in features.items():
            st.markdown(f"- **`{feat}`** — {desc}")

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:1rem;opacity:0.6;font-size:0.85rem">
        Built by <b>Ronak Rajput</b> · AIML Engineer · Mind Inventory Internship 2025 ·
        <a href="https://github.com/ronakrajput8882/Shop-Smart-Ecommerce" style="color:#00d2ff">View on GitHub</a>
    </div>
    """, unsafe_allow_html=True)
