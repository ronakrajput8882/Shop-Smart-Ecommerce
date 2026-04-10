import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("shop_smart_ecommerce.csv")
        df["Revenue"] = df["Revenue"].astype(int)
        return df
    except FileNotFoundError:
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
            "Revenue": np.random.choice([1, 0], n, p=[0.155, 0.845]),
        })
        return df

DARK_COLORS = ["#00d2ff", "#e74c3c", "#f39c12", "#2ecc71", "#9b59b6"]
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,32,39,0.6)",
    font=dict(color="#e0e0e0", family="Inter"),
    margin=dict(t=40, b=30, l=30, r=20),
)

def show():
    st.markdown("""
    <div class="main-header">
        <h1>📊 EDA & Insights</h1>
        <p>Exploratory Data Analysis of 12,329 online shopping sessions</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()

    # ── 1. Class Distribution ────────────────────────────────────────────────
    st.markdown("### 🎯 Target Distribution")
    col1, col2 = st.columns([1, 2])

    with col1:
        purchase_counts = df["Revenue"].value_counts()
        fig_pie = px.pie(
            values=purchase_counts.values,
            names=["No Purchase", "Purchase"],
            color_discrete_sequence=["#e74c3c", "#00d2ff"],
            hole=0.5,
        )
        fig_pie.update_traces(textinfo="percent+label", textfont_size=13)
        fig_pie.update_layout(
            **PLOTLY_LAYOUT,
            title="Purchase vs No Purchase",
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-card">
            <b>⚠️ Class Imbalance: 84.5% / 15.5%</b><br>
            <span style="font-size:0.85rem">Only 1 in 6.5 sessions ends in a purchase. This severe imbalance
            is why we use <code>class_weight="balanced"</code> in the Decision Tree — 
            without it, the model would just predict "No Purchase" for everything and still get 84% accuracy.</span>
        </div>
        <div class="insight-card" style="margin-top:0.8rem">
            <b>🎯 Why Recall over Accuracy?</b><br>
            <span style="font-size:0.85rem">Missing a real buyer (False Negative) costs far more than 
            incorrectly flagging a browser (False Positive). So we optimize for Recall (83.3%) 
            over raw accuracy, using F1 as our GridSearch scoring metric.</span>
        </div>
        """, unsafe_allow_html=True)

    # ── 2. Monthly Conversion ────────────────────────────────────────────────
    st.markdown("### 📅 Monthly Session & Conversion Analysis")
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly = df.groupby("Month").agg(
        sessions=("Revenue", "count"),
        purchases=("Revenue", "sum")
    ).reset_index()
    monthly["conversion_rate"] = (monthly["purchases"] / monthly["sessions"] * 100).round(2)
    # Sort by proper month order
    monthly["Month"] = pd.Categorical(monthly["Month"], categories=month_order, ordered=True)
    monthly = monthly.sort_values("Month")

    fig_month = make_subplots(specs=[[{"secondary_y": True}]])
    fig_month.add_trace(go.Bar(
        x=monthly["Month"], y=monthly["sessions"],
        name="Total Sessions", marker_color="#1a3a5c", opacity=0.9
    ), secondary_y=False)
    fig_month.add_trace(go.Scatter(
        x=monthly["Month"], y=monthly["conversion_rate"],
        name="Conversion Rate %", mode="lines+markers",
        line=dict(color="#00d2ff", width=3),
        marker=dict(size=8, color="#00d2ff")
    ), secondary_y=True)
    fig_month.update_layout(**PLOTLY_LAYOUT, height=360)
    fig_month.update_yaxes(title_text="Sessions", secondary_y=False, gridcolor="#1a2a3a")
    fig_month.update_yaxes(title_text="Conversion Rate %", secondary_y=True)
    st.plotly_chart(fig_month, use_container_width=True)

    # ── 3. PageValues Distribution ───────────────────────────────────────────
    st.markdown("### 📈 PageValues — Strongest Predictor (+0.49 correlation)")
    col_a, col_b = st.columns(2)

    with col_a:
        fig_pv = px.histogram(
            df[df["PageValues"] < 100], x="PageValues",
            color=df[df["PageValues"] < 100]["Revenue"].map({0: "No Purchase", 1: "Purchase"}),
            color_discrete_map={"No Purchase": "#e74c3c", "Purchase": "#00d2ff"},
            nbins=60, barmode="overlay", opacity=0.75,
            labels={"color": ""},
            title="PageValues Distribution by Revenue"
        )
        fig_pv.update_layout(**PLOTLY_LAYOUT, height=330)
        st.plotly_chart(fig_pv, use_container_width=True)

    with col_b:
        page_val_summary = df.groupby("Revenue")["PageValues"].describe()[["mean","50%","max"]].reset_index()
        page_val_summary["Revenue"] = page_val_summary["Revenue"].map({0: "No Purchase", 1: "Purchase"})
        page_val_summary.columns = ["Outcome", "Mean PageValue", "Median PageValue", "Max PageValue"]
        fig_bar_pv = px.bar(
            page_val_summary.melt(id_vars="Outcome"),
            x="variable", y="value", color="Outcome", barmode="group",
            color_discrete_map={"No Purchase": "#e74c3c", "Purchase": "#00d2ff"},
            title="PageValues Stats: Purchase vs No Purchase"
        )
        fig_bar_pv.update_layout(**PLOTLY_LAYOUT, height=330)
        st.plotly_chart(fig_bar_pv, use_container_width=True)

    # ── 4. Exit & Bounce Rates ───────────────────────────────────────────────
    st.markdown("### 📉 Exit Rate & Bounce Rate — Abandonment Signals")
    col_c, col_d = st.columns(2)

    with col_c:
        fig_exit = px.box(
            df[df["ExitRates"] < 0.15], x=df[df["ExitRates"] < 0.15]["Revenue"].map({0: "No Purchase", 1: "Purchase"}),
            y="ExitRates",
            color=df[df["ExitRates"] < 0.15]["Revenue"].map({0: "No Purchase", 1: "Purchase"}),
            color_discrete_map={"No Purchase": "#e74c3c", "Purchase": "#00d2ff"},
            title="Exit Rate by Purchase Outcome"
        )
        fig_exit.update_layout(**PLOTLY_LAYOUT, height=330, showlegend=False)
        st.plotly_chart(fig_exit, use_container_width=True)

    with col_d:
        fig_bounce = px.box(
            df[df["BounceRates"] < 0.15], x=df[df["BounceRates"] < 0.15]["Revenue"].map({0: "No Purchase", 1: "Purchase"}),
            y="BounceRates",
            color=df[df["BounceRates"] < 0.15]["Revenue"].map({0: "No Purchase", 1: "Purchase"}),
            color_discrete_map={"No Purchase": "#e74c3c", "Purchase": "#00d2ff"},
            title="Bounce Rate by Purchase Outcome"
        )
        fig_bounce.update_layout(**PLOTLY_LAYOUT, height=330, showlegend=False)
        st.plotly_chart(fig_bounce, use_container_width=True)

    # ── 5. Visitor Type & Weekend ────────────────────────────────────────────
    st.markdown("### 👤 Visitor Type & Weekend Effect")
    col_e, col_f = st.columns(2)

    with col_e:
        vtype_conv = df.groupby("VisitorType")["Revenue"].agg(["count","sum"]).reset_index()
        vtype_conv.columns = ["VisitorType", "Total", "Purchases"]
        vtype_conv["Conversion %"] = (vtype_conv["Purchases"] / vtype_conv["Total"] * 100).round(1)
        fig_vtype = px.bar(
            vtype_conv, x="VisitorType", y="Conversion %",
            color="Conversion %", color_continuous_scale=["#e74c3c","#f39c12","#00d2ff"],
            title="Conversion Rate by Visitor Type", text="Conversion %"
        )
        fig_vtype.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_vtype.update_layout(**PLOTLY_LAYOUT, height=330, coloraxis_showscale=False)
        st.plotly_chart(fig_vtype, use_container_width=True)

    with col_f:
        wknd_conv = df.groupby("Weekend")["Revenue"].agg(["count","sum"]).reset_index()
        wknd_conv["Day Type"] = wknd_conv["Weekend"].map({True: "Weekend", False: "Weekday"})
        wknd_conv["Conversion %"] = (wknd_conv["sum"] / wknd_conv["count"] * 100).round(1)
        fig_wknd = px.bar(
            wknd_conv, x="Day Type", y="Conversion %",
            color="Day Type",
            color_discrete_map={"Weekend": "#00d2ff", "Weekday": "#e74c3c"},
            title="Weekday vs Weekend Conversion", text="Conversion %"
        )
        fig_wknd.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_wknd.update_layout(**PLOTLY_LAYOUT, height=330, showlegend=False)
        st.plotly_chart(fig_wknd, use_container_width=True)

    # ── 6. Correlation heatmap ───────────────────────────────────────────────
    st.markdown("### 🔥 Correlation with Revenue (Numerical Features)")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr_with_rev = df[num_cols].corr()["Revenue"].drop("Revenue").sort_values()

    fig_corr = px.bar(
        x=corr_with_rev.values,
        y=corr_with_rev.index,
        orientation="h",
        color=corr_with_rev.values,
        color_continuous_scale=["#e74c3c","#333","#00d2ff"],
        color_continuous_midpoint=0,
        title="Pearson Correlation of Each Feature with Revenue"
    )
    fig_corr.update_layout(**PLOTLY_LAYOUT, height=400, coloraxis_showscale=False)
    fig_corr.add_vline(x=0, line_color="#ffffff", line_width=1, opacity=0.3)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <b>📌 Takeaway:</b> <code>PageValues</code> (+0.49) dominates all features.
        A single analytics-derived metric carries more signal than session-time, region, or device features combined.
        <code>ExitRates</code> (−0.21) and <code>BounceRates</code> (−0.15) are the strongest negative signals.
    </div>
    """, unsafe_allow_html=True)
