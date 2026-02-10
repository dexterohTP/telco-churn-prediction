"""
ChurnShield Analytics - Customer Churn Prediction App

Author: Dexter Oh Han Yu
Course: CAI2C08 - Machine Learning for Developers
Institution: Temasek Polytechnic
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')


# PAGE CONFIGURATION


st.set_page_config(
    page_title="ChurnShield Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# LOAD MODEL AND PREPROCESSING ARTIFACTS


@st.cache_resource
def load_model_artifacts():
    """Load the trained model, scaler, and feature columns"""
    try:
        model = joblib.load('churn_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.info("Please ensure churn_model.pkl, scaler.pkl, and feature_columns.pkl are in the same directory")
        return None, None, None

model, scaler, feature_columns = load_model_artifacts()


# HEADER


st.title("📊 ChurnShield Analytics")
st.markdown("### AI-Powered Customer Churn Prediction System")
st.markdown("---")

# Sidebar with company info
st.sidebar.image("https://via.placeholder.com/300x100/2c3e50/ffffff?text=ChurnShield", use_container_width=True)
st.sidebar.markdown("### About")
st.sidebar.info(
    "ChurnShield Analytics uses advanced machine learning to predict customer churn risk, "
    "enabling proactive retention strategies and revenue protection."
)
st.sidebar.markdown("**Developed by:** Dexter Oh Han Yu")
st.sidebar.markdown("**Institution:** Temasek Polytechnic")


# INPUT SECTION


st.header("1️⃣ Customer Information")
st.markdown("Enter customer details below to predict churn risk")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Demographics & Account")
    
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure_months = st.slider("Tenure (Months)", 0, 72, 12, 
                              help="How long the customer has been with the company")
    
    st.subheader("💳 Billing & Contract")
    
    contract = st.selectbox("Contract Type", 
                            ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", 
                                   ["Electronic check", "Mailed check", 
                                    "Bank transfer (automatic)", 
                                    "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges ($)", 
                                       min_value=0.0, max_value=200.0, 
                                       value=70.0, step=5.0)

with col2:
    st.subheader("📞 Services")
    
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    
    # Multiple lines depends on phone service
    if phone_service == "Yes":
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
    else:
        multiple_lines = "No phone service"
        st.info("Multiple Lines: Not applicable (no phone service)")
    
    internet_service = st.selectbox("Internet Service", 
                                     ["No", "DSL", "Fiber optic"])
    
    # These depend on internet service
    if internet_service != "No":
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
    else:
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"
        st.info("Internet-dependent services: Not applicable (no internet)")


# PREDICTION SECTION


st.markdown("---")
st.header("2️⃣ Churn Risk Prediction")

if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
    
    if model is None:
        st.error("Model not loaded. Cannot make prediction.")
    else:
        # Calculate total charges (approximate)
        total_charges = monthly_charges * tenure_months if tenure_months > 0 else monthly_charges
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Senior Citizen': [1 if senior_citizen == "Yes" else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'Tenure Months': [tenure_months],
            'Phone Service': [phone_service],
            'Multiple Lines': [multiple_lines],
            'Internet Service': [internet_service],
            'Online Security': [online_security],
            'Online Backup': [online_backup],
            'Device Protection': [device_protection],
            'Tech Support': [tech_support],
            'Streaming TV': [streaming_tv],
            'Streaming Movies': [streaming_movies],
            'Contract': [contract],
            'Paperless Billing': [paperless_billing],
            'Payment Method': [payment_method],
            'Monthly Charges': [monthly_charges],
            'Total Charges': [total_charges]
        })
        
        # Feature Engineering (same as training)
        # Count services
        service_cols = ["Phone Service", "Multiple Lines", "Internet Service",
                       "Online Security", "Online Backup", "Device Protection",
                       "Tech Support", "Streaming TV", "Streaming Movies"]
        input_data["Num_Services"] = input_data[service_cols].apply(
            lambda row: sum(val not in ["No", "No internet service", "No phone service"] for val in row),
            axis=1
        )
        
        # Is new customer
        input_data["Is_New_Customer"] = (input_data["Tenure Months"] <= 12).astype(int)
        
        # Contract commitment
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        input_data["Contract_Commitment"] = input_data["Contract"].map(contract_map)
        
        # Premium services
        premium = ["Tech Support", "Online Security", "Device Protection"]
        input_data["Has_Premium_Services"] = input_data[premium].apply(
            lambda row: (row != "No").sum(), axis=1
        )
        
        # High risk payment
        input_data["High_Risk_Payment"] = (input_data["Payment Method"] == "Electronic check").astype(int)
        
        # Service categories
        has_phone = (input_data["Phone Service"] == "Yes").astype(int)
        has_internet = (input_data["Internet Service"] != "No").astype(int)
        has_security = (input_data[["Online Security", "Online Backup", "Device Protection", "Tech Support"]] != "No").any(axis=1).astype(int)
        has_streaming = (input_data[["Streaming TV", "Streaming Movies"]] != "No").any(axis=1).astype(int)
        input_data["Service_Categories"] = has_phone + has_internet + has_security + has_streaming
        
        # Tenure risk score
        input_data["Tenure_Risk_Score"] = (input_data["Is_New_Customer"] * 2) + (input_data["Contract_Commitment"] == 0).astype(int)
        
        # Drop Contract column
        input_data = input_data.drop(['Contract'], axis=1)
        
        # One-hot encode
        categorical_cols = input_data.select_dtypes(include=['object']).columns.tolist()
        input_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)
        
        # Align with training features
        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        input_encoded = input_encoded[feature_columns]
        
        # Scale
        input_scaled = scaler.transform(input_encoded)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Display results
        st.markdown("---")
        
        # Create columns for results
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric(
                label="Churn Prediction",
                value="WILL CHURN" if prediction == 1 else "WILL STAY",
                delta="High Risk" if prediction == 1 else "Low Risk",
                delta_color="inverse"
            )
        
        with result_col2:
            churn_prob = probability[1] * 100
            st.metric(
                label="Churn Probability",
                value=f"{churn_prob:.1f}%",
                delta=f"Stay: {probability[0]*100:.1f}%"
            )
        
        with result_col3:
            # Risk level
            if churn_prob >= 70:
                risk_level = "🔴 CRITICAL"
                risk_color = "red"
            elif churn_prob >= 50:
                risk_level = "🟠 HIGH"
                risk_color = "orange"
            elif churn_prob >= 30:
                risk_level = "🟡 MEDIUM"
                risk_color = "yellow"
            else:
                risk_level = "🟢 LOW"
                risk_color = "green"
            
            st.metric(
                label="Risk Level",
                value=risk_level
            )
        
        # Recommendations
        st.markdown("---")
        st.header("3️⃣ Recommended Actions")
        
        if prediction == 1:
            st.error("⚠️ **HIGH CHURN RISK DETECTED** - Immediate intervention recommended")
            
            recommendations = []
            
            # Contract-based recommendations
            if contract == "Month-to-month":
                recommendations.append("🎯 **Offer Contract Upgrade:** Incentivize switching to 1-year or 2-year contract with 15-20% discount")
            
            # Tenure-based
            if tenure_months <= 12:
                recommendations.append("📞 **New Customer Retention:** Schedule onboarding call, offer 3-month loyalty bonus")
            
            # Payment method
            if payment_method == "Electronic check":
                recommendations.append("💳 **Payment Method Change:** Encourage automatic payment setup with $5-10/month discount")
            
            # Services
            if input_data["Num_Services"].iloc[0] < 3:
                recommendations.append("📦 **Service Bundle Offer:** Promote value-added services (security, tech support) with trial period")
            
            # Monthly charges
            if monthly_charges > 70:
                recommendations.append("💰 **Price Retention Offer:** Provide limited-time discount or price lock guarantee")
            
            # Display recommendations
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
            
            # Expected impact
            st.success(f"✅ **Expected Impact:** Implementing these actions could reduce churn probability by 20-40%")
            
        else:
            st.success("✅ **LOW CHURN RISK** - Customer is likely to stay")
            st.markdown("**Suggested Actions:**")
            st.markdown("- Continue current service level")
            st.markdown("- Consider upsell opportunities for additional services")
            st.markdown("- Include in loyalty rewards program")
            st.markdown("- Monitor monthly for any behavior changes")


# FOOTER


st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>ChurnShield Analytics | Powered by Machine Learning</p>
    <p>Developed by Dexter Oh Han Yu | Temasek Polytechnic | 2026</p>
    </div>
    """,
    unsafe_allow_html=True
)


# SIDEBAR - MODEL INFO


with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    
    if model is not None:
        st.markdown(f"**Algorithm:** Gradient Boosting")
        st.markdown(f"**Features:** {len(feature_columns)}")
        st.markdown(f"**Accuracy:** ~81%")
        st.markdown(f"**F1-Score (Churn):** ~68%")
        st.markdown(f"**Recall (Churn):** ~60%")
    
    st.markdown("---")
    st.markdown("### 💡 How It Works")
    st.markdown(
        """
        1. Enter customer details
        2. Model analyzes 36+ features
        3. Predicts churn probability
        4. Provides actionable recommendations
        """
    )
