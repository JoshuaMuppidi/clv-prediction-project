import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
import joblib

# Page Configuration
st.set_page_config(page_title="CLV Predictor", layout="wide", page_icon="💰")

# Custom CSS for advanced UI
st.markdown("""
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #4CAF50;
    }
    .sub-title {
        font-size: 22px;
        font-weight: 500;
        color: #555;
        margin-bottom: 20px;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load model and features
model = joblib.load("xgb_model.pkl")
with open('model_features.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

# Sidebar Menu
with st.sidebar:
    selected = option_menu(
        menu_title="CLV Dashboard",
        options=["Predict CLV", "Feature Importance"],
        icons=["calculator", "bar-chart"],
        menu_icon="cast",
        default_index=0,
    )

# Preprocessing function for raw data
def preprocess_data(df_raw):
    # Convert TotalCharges to numeric
    df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce').fillna(0)

    # Binary mapping
    df_raw['SeniorCitizen'] = df_raw['SeniorCitizen'].astype(int)
    df_raw['Churn'] = df_raw['Churn'].map({'Yes': 1, 'No': 0})
    df_raw['Partner'] = df_raw['Partner'].map({'Yes': 1, 'No': 0})
    df_raw['Dependents'] = df_raw['Dependents'].map({'Yes': 1, 'No': 0})
    df_raw['PhoneService'] = df_raw['PhoneService'].map({'Yes': 1, 'No': 0})

    # One-hot encoding for categorical columns
    df_encoded = pd.get_dummies(df_raw, columns=[
        'gender', 'InternetService', 'Contract', 'PaymentMethod'
    ])

    # Reindex to match training features
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # Scaling numerical columns
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

    return df_encoded

# Predict CLV Page
if selected == "Predict CLV":
    st.markdown("<div class='main-title'>Customer Lifetime Value Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Fill in the customer details to get a CLV prediction.</div>", unsafe_allow_html=True)

    with st.form(key='clv_form'):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox('Gender', ['Male', 'Female'])
            senior_citizen = st.selectbox('Senior Citizen', [0, 1])
            contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
            tenure = st.slider('Tenure (months)', 0, 72, 12)
            monthly_charges = st.number_input('Monthly Charges', 0.0, 500.0, 50.0)
            total_charges = st.number_input('Total Charges', 0.0, 10000.0, 1000.0)
            churn = st.selectbox('Churn', ['Yes', 'No'])

        with col2:
            internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
            payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            dependents = st.selectbox('Dependents', ['Yes', 'No'])
            partner = st.selectbox('Partner', ['Yes', 'No'])
            phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
            paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])

        submitted = st.form_submit_button("💡 Predict CLV")

    if submitted:
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Contract': [contract],
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'Churn': [churn],
            'InternetService': [internet_service],
            'PaymentMethod': [payment_method],
            'Dependents': [dependents],
            'Partner': [partner],
            'PhoneService': [phone_service],
            'PaperlessBilling': [paperless_billing]
        })

        processed_input = preprocess_data(input_data)
        prediction = model.predict(processed_input)[0]

        st.markdown(f"""
        <div class='prediction-box'>
            <h3>✅ Predicted Customer Lifetime Value:</h3>
            <h1 style='color:#1f77b4;'>${prediction:,.2f}</h1>
        </div>
        """, unsafe_allow_html=True)

# Feature Importance Page
elif selected == "Feature Importance":
    st.markdown("<div class='main-title'>📊 Feature Importance</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Top 10 features that influenced the CLV prediction model.</div>", unsafe_allow_html=True)

    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_columns, 'Importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='coolwarm')
    plt.title('Top 10 Important Features for CLV Prediction')
    st.pyplot(plt.gcf())
    plt.clf()

# Bulk Predictions
st.sidebar.header("Bulk Predictions")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file for bulk predictions", type=["csv"])

if uploaded_file is not None:
    df_bulk = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(df_bulk.head())

    processed_bulk = preprocess_data(df_bulk)
    bulk_predictions = model.predict(processed_bulk)
    df_bulk["Predicted_CLV"] = bulk_predictions

    st.write("Predictions:")
    st.write(df_bulk)

    csv = df_bulk.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="bulk_predictions.csv",
        mime="text/csv"
    )

# Final separator
st.markdown("---")
st.subheader("Feature Importance Visualization")
st.image("xgboost_feature_importance.png", caption="XGBoost Feature Importance Plot")
