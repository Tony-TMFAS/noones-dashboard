import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from PIL import Image

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="noOnes User Insights Dashboard", layout="wide")

# =====================================
# CUSTOM STYLING
# =====================================
st.markdown("""
    <style>
        /* Compact, centered banner styling */
        .banner-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 15px;
        }
        .banner-container img {
            height: 90px !important;   /* ⬅️ Adjust height as needed (try 70–100px) */
            width: auto;
            object-fit: contain;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# =====================================
# HEADER & SIDEBAR
# =====================================
st.markdown('<div class="banner-container">', unsafe_allow_html=True)
st.image("assets/noones_banner.png", width=800)
st.markdown('</div>', unsafe_allow_html=True)

st.sidebar.image("assets/noones_logo.png", width=120)
st.sidebar.title("noOnes Dashboard")
st.sidebar.markdown("Navigate between insights and analytics.")

# =====================================
# LOAD DATA
# =====================================


@st.cache_data
def load_data():
    df = pd.read_csv("noones_p2p_data.csv")
    return df


df = load_data()

# Check if engineered columns exist, otherwise create them
if 'success_rate' not in df.columns:
    df['success_rate'] = df['successful_trades'] / (df['trades_made'] + 1e-6)
if 'engagement_index' not in df.columns:
    df['engagement_index'] = (
        df['app_logins'] * 0.3
        + df['wallet_balance'] * 0.2
        + df['referrals'] * 0.2
        + df['satisfaction_score'] * 0.3
    )

# =====================================
# LOAD MODEL + SCALER
# =====================================


@st.cache_resource
def load_model():
    model = joblib.load("models/random_forest.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


# =====================================
# SIDEBAR NAVIGATION
# =====================================
page = st.sidebar.radio(
    "Go to Section:",
    ["🏠 Overview", "📈 Engagement Analytics",
     "🤖 Model Performance", "💡 Strategic Insights", "🔮 Churn Predictor"]
)

# =====================================
# OVERVIEW
# =====================================
if page == "🏠 Overview":
    st.title("📊 noOnes User Engagement & Retention Dashboard")
    st.markdown(
        "This dashboard provides synthetic key insights into user behavior, engagement, and churn patterns for the **noOnes** platform."
    )

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users", f"{len(df):,}")
    col2.metric(
        "Churn Rate", f"{df['churned'].mean()*100:.2f}%" if 'churned' in df.columns else "N/A")
    col3.metric("Avg Satisfaction", f"{df['satisfaction_score'].mean():.2f}")
    col4.metric("Avg Engagement Index", f"{df['engagement_index'].mean():.2f}")

    st.divider()

    st.subheader("🔍 Key Insights")
    st.markdown("""
    - Users with **higher satisfaction** and **frequent logins** are less likely to churn.  
    - **Wallet balance** and **referrals** positively influence engagement.  
    - Both **shop users** and **non-shop users** show similar engagement — an opportunity to enhance shop-related features.
    """)

# =====================================
# ENGAGEMENT ANALYTICS
# =====================================
elif page == "📈 Engagement Analytics":
    st.title("📈 Engagement & Behavior Analysis")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['engagement_index'], bins=20,
                     kde=True, ax=ax, color='skyblue')
        ax.set_title("Engagement Index Distribution")
        st.pyplot(fig)

    with col2:
        if 'has_shop' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df, x='has_shop',
                        y='engagement_index', palette='coolwarm')
            ax.set_title("Engagement: Shop Users vs Non-Shop Users")
            st.pyplot(fig)

    st.subheader("Satisfaction vs Engagement")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x='satisfaction_score',
        y='engagement_index',
        hue='churned' if 'churned' in df.columns else None,
        palette='coolwarm'
    )
    ax.set_title("Satisfaction vs Engagement (by Churn)")
    st.pyplot(fig)

# =====================================
# MODEL PERFORMANCE
# =====================================
elif page == "🤖 Model Performance":
    st.title("🤖 Feature Insights")

    st.subheader("Feature Importance")
    st.image("assets/noones_feature_importance.png",
             caption="Top Predictive Features", use_container_width=True)

    st.markdown("""
    **Model Insights:**
    - **Top predictors:** app_logins, satisfaction_score, wallet_balance.  
    - Random Forest achieves strong recall for churned users.  
    - Balanced Logistic Regression improves fairness between active vs churned classes.
    """)

# =====================================
# STRATEGIC INSIGHTS
# =====================================
elif page == "💡 Strategic Insights":
    st.title("💡 Strategic Recommendations for noOnes")

    st.markdown("""
    ### 1️⃣ Boost User Retention
    - Launch **targeted campaigns** for low-engagement or low-satisfaction users.  
    - Reward consistent logins and frequent trading.

    ### 2️⃣ Increase Engagement
    - Encourage referrals and repeat activity via **tiered incentive systems**.  
    - Personalize in-app content to enhance satisfaction and loyalty.

    ### 3️⃣ Optimize Platform Experience
    - Monitor churn risk in real time using the predictive model.  
    - Use dashboards like this to track **monthly engagement** and **churn movement**.

    ---
    **Executive Summary:**  
    The noOnes ecosystem shows strong correlation between **user activity**, **satisfaction**, and **retention**.  
    Increasing consistent interaction and satisfaction are key levers for sustainable growth.
    """)

# =====================================
# CHURN PREDICTOR
# =====================================
elif page == "🔮 Churn Predictor":
    st.title("🔮 Noones Customer Churn Prediction Dashboard")
    st.markdown(
        "Use this dashboard to predict whether a customer is likely to churn.")

    model, scaler = load_model()

    # USER INPUT FORM (conditionally in sidebar)
    st.sidebar.header("Enter Customer Details")
    st.sidebar.markdown("Fill in customer metrics below:")

    input_data = {
        'account_age_days': st.sidebar.number_input("Account Age (days)", 0, 2000, 180),
        'trades_made': st.sidebar.number_input("Trades Made", 0, 500, 15),
        'successful_trades': st.sidebar.number_input("Successful Trades", 0, 500, 10),
        'cancelled_trades': st.sidebar.number_input("Cancelled Trades", 0, 100, 2),
        'avg_trade_value': st.sidebar.number_input("Average Trade Value ($)", 0.0, 10000.0, 250.0),
        'app_logins': st.sidebar.number_input("App Logins", 0, 1000, 45),
        'referrals': st.sidebar.number_input("Referrals", 0, 100, 3),
        'wallet_balance': st.sidebar.number_input("Wallet Balance ($)", 0.0, 5000.0, 120.0),
        'satisfaction_score': st.sidebar.slider("Satisfaction Score", 0.0, 5.0, 3.5),
        'success_rate': st.sidebar.slider("Success Rate", 0.0, 1.0, 0.65),
        'engagement_index': st.sidebar.slider("Engagement Index", 0.0, 1.0, 0.5),
        'has_shop': st.sidebar.selectbox("Has Shop?", [0, 1]),
        'giftcard_trades': st.sidebar.number_input("Giftcard Trades", 0, 200, 5)
    }

    input_df = pd.DataFrame([input_data])

    # MAKE PREDICTION
    if st.button("🔮 Predict Churn Probability"):
        scaled = scaler.transform(input_df)
        prediction = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]  # probability of churn

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error(
                f"⚠️ This customer is **likely to churn** (Probability: {prob:.2%})")
        else:
            st.success(
                f"✅ This customer is **not likely to churn** (Probability: {prob:.2%})")

    # UPLOAD BATCH PREDICTIONS
    st.markdown("---")
    st.subheader("📤 Upload Batch Data for Prediction")

    uploaded = st.file_uploader(
        "Upload a CSV file with customer data", type=["csv"])
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        st.write("Preview of Uploaded Data:")
        st.dataframe(data.head())

        scaled_data = scaler.transform(data)
        batch_pred = model.predict(scaled_data)
        data['Predicted_Churn'] = batch_pred
        data['Churn_Probability'] = model.predict_proba(scaled_data)[:, 1]

        st.success("✅ Predictions Complete")
        st.write(data.head())

        # Downloadable CSV
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Predictions CSV",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

    # FOOTER
    st.markdown("---")
    st.caption("Powered by Noones AI | Built using Streamlit")
