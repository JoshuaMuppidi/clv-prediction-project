# 🚀 Model Test for CLV Prediction using Linear Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
df['CLV'] = df['MonthlyCharges'] * df['tenure']

# Prepare features
features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
            'InternetService', 'Contract', 'PaymentMethod', 'tenure', 'MonthlyCharges', 
            'TotalCharges', 'Churn']

df_model = df[features].copy()
df_model['Churn'] = df_model['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
df_model_encoded = pd.get_dummies(df_model, drop_first=True)

# Model
X = df_model_encoded
y = df['CLV']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n✅ Linear Regression Model Evaluation Metrics:")
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)

input("\nPress Enter to exit...")