import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

@st.cache_resource
def load_model():
    """Train and cache the model from CSV."""
    try:
        df = pd.read_csv("shop_smart_ecommerce.csv")
    except FileNotFoundError:
        # Fallback: generate synthetic data that mimics the real distribution
        np.random.seed(42)
        n = 12329
        df = pd.DataFrame({
            "Administrative": np.random.poisson(2.3, n),
            "Administrative_Duration": np.random.exponential(80, n),
            "Informational": np.random.poisson(0.5, n),
            "Informational_Duration": np.random.exponential(34, n),
            "ProductRelated": np.random.poisson(31, n),
            "ProductRelated_Duration": np.random.exponential(1195, n),
            "BounceRates": np.random.beta(1, 10, n),
            "ExitRates": np.random.beta(2, 10, n),
            "PageValues": np.random.exponential(5.9, n),
            "SpecialDay": np.random.choice([0, 0.2, 0.4, 0.6, 0.8, 1.0], n, p=[0.8, 0.05, 0.05, 0.04, 0.03, 0.03]),
            "Month": np.random.choice(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], n),
            "OperatingSystems": np.random.randint(1, 9, n),
            "Browser": np.random.randint(1, 14, n),
            "Region": np.random.randint(1, 10, n),
            "TrafficType": np.random.randint(1, 21, n),
            "VisitorType": np.random.choice(["Returning_Visitor", "New_Visitor", "Other"], n, p=[0.845, 0.14, 0.015]),
            "Weekend": np.random.choice([True, False], n, p=[0.23, 0.77]),
            "Revenue": np.random.choice([True, False], n, p=[0.155, 0.845]),
        })

    X = df.drop("Revenue", axis=1)
    y = df["Revenue"].astype(int)

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    model = Pipeline([
        ("preprocess", preprocessor),
        ("model", DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model.fit(X_train, y_train)
    return model, num_cols, cat_cols

def show():
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Predict Purchase Intent</h1>
        <p>Fill in session behavioral signals to predict whether a visitor will purchase</p>
    </div>
    """, unsafe_allow_html=True)

    model, num_cols, cat_cols = load_model()

    st.markdown("### 📝 Session Input Features")
    st.info("💡 **Tip:** `PageValues` is the single strongest predictor. Higher values = much more likely to purchase.", icon="ℹ️")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 📄 Page Engagement")
        administrative = st.slider("Administrative Pages Visited", 0, 27, 2,
            help="Number of administrative pages the visitor visited")
        administrative_duration = st.slider("Administrative Duration (sec)", 0.0, 3000.0, 80.0, step=10.0)
        informational = st.slider("Informational Pages Visited", 0, 24, 0)
        informational_duration = st.slider("Informational Duration (sec)", 0.0, 2500.0, 0.0, step=10.0)
        product_related = st.slider("Product-Related Pages Visited", 0, 705, 31,
            help="More product pages = higher purchase intent")
        product_duration = st.slider("Product-Related Duration (sec)", 0.0, 63000.0, 1195.0, step=50.0)

        st.markdown("#### 📊 Rate Signals")
        bounce_rates = st.slider("Bounce Rate", 0.0, 0.20, 0.02, step=0.005, format="%.3f",
            help="Avg bounce rate of pages visited — lower is better")
        exit_rates = st.slider("Exit Rate", 0.0, 0.20, 0.04, step=0.005, format="%.3f",
            help="Avg exit rate — lower means visitor is staying engaged")
        page_values = st.slider("Page Values", 0.0, 360.0, 5.9, step=0.5,
            help="⭐ Most predictive! Avg value of pages visited. Higher = purchase likely")
        special_day = st.select_slider("Special Day Proximity", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], value=0.0,
            help="0 = regular day, 1 = on a special day (Valentine's, Mother's Day etc.)")

    with col2:
        st.markdown("#### ⚙️ Technical Info")
        operating_system = st.selectbox("Operating System", [1, 2, 3, 4, 5, 6, 7, 8],
            index=1, help="OS code (1=Windows, 2=Mac, etc.)")
        browser = st.selectbox("Browser", list(range(1, 14)), index=1,
            help="Browser code")
        region = st.selectbox("Region", list(range(1, 10)), help="Geographic region code")
        traffic_type = st.selectbox("Traffic Type", list(range(1, 21)), index=1,
            help="Traffic source code (1=direct, 2=organic, etc.)")

        st.markdown("#### 👤 Visitor & Session Context")
        visitor_type = st.selectbox("Visitor Type", ["Returning_Visitor", "New_Visitor", "Other"],
            help="Returning visitors have different conversion patterns")
        month = st.selectbox("Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            index=10, help="Month of the session — Nov/Dec have higher conversion")
        weekend = st.toggle("Weekend Session", value=False,
            help="Sessions on weekends behave slightly differently")

        st.markdown("#### 🎯 Predict")
        predict_clicked = st.button("🔮 Predict Purchase Intent", use_container_width=True, type="primary")

    # ── Prediction ────────────────────────────────────────────────────────────
    if predict_clicked:
        input_data = pd.DataFrame([{
            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,
            "Informational": informational,
            "Informational_Duration": informational_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_duration,
            "BounceRates": bounce_rates,
            "ExitRates": exit_rates,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": operating_system,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": weekend,
        }])

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        purchase_prob = proba[1] * 100
        no_purchase_prob = proba[0] * 100

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        res_col1, res_col2 = st.columns([2, 1])

        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-buy">
                    <div class="result-title">✅ WILL PURCHASE</div>
                    <div class="result-sub">This visitor shows high purchase intent</div>
                    <div style="font-size:2.5rem;font-weight:800;margin-top:1rem">{purchase_prob:.1f}%</div>
                    <div style="font-size:0.85rem;opacity:0.8">Purchase Probability</div>
                </div>
                """, unsafe_allow_html=True)
                st.success("💡 **Recommendation:** This is a high-intent visitor. Trigger a personalized offer, loyalty reward, or free shipping popup to seal the deal!")
            else:
                st.markdown(f"""
                <div class="result-nobuy">
                    <div class="result-title">❌ WON'T PURCHASE</div>
                    <div class="result-sub">This visitor is likely browsing, not buying</div>
                    <div style="font-size:2.5rem;font-weight:800;margin-top:1rem">{no_purchase_prob:.1f}%</div>
                    <div style="font-size:0.85rem;opacity:0.8">No-Purchase Probability</div>
                </div>
                """, unsafe_allow_html=True)
                st.warning("💡 **Recommendation:** Re-engage with a discount popup, wishlist reminder, or email capture to nurture this visitor for future conversion.")

        with res_col2:
            st.markdown("#### 📊 Probability Breakdown")
            st.metric("🟢 Will Purchase", f"{purchase_prob:.1f}%")
            st.metric("🔴 Won't Purchase", f"{no_purchase_prob:.1f}%")

            st.markdown("#### 🔍 Key Signals")
            if page_values > 10:
                st.success(f"✅ PageValues = {page_values:.1f} (Strong buy signal)")
            elif page_values > 2:
                st.info(f"ℹ️ PageValues = {page_values:.1f} (Moderate)")
            else:
                st.error(f"⚠️ PageValues = {page_values:.1f} (Weak buy signal)")

            if exit_rates > 0.06:
                st.error(f"⚠️ Exit Rate = {exit_rates:.3f} (High abandonment risk)")
            else:
                st.success(f"✅ Exit Rate = {exit_rates:.3f} (Engaged visitor)")

            if visitor_type == "Returning_Visitor":
                st.success("✅ Returning Visitor (more likely to convert)")
            else:
                st.info(f"ℹ️ {visitor_type}")