import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np

# Load dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
df['CLV'] = df['MonthlyCharges'] * df['tenure']

# Prepare features
features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
            'InternetService', 'Contract', 'PaymentMethod', 'tenure', 'MonthlyCharges', 
            'TotalCharges', 'Churn']

df_model = df[features]
df_model['Churn'] = df_model['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
df_model_encoded = pd.get_dummies(df_model, drop_first=True)

# Model
X = df_model_encoded
y = df['CLV']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Perform 5-fold cross-validation on the full dataset
scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print("\n✅ Cross-validation R² scores for each fold:", scores)
print("✅ Mean R² score:", np.mean(scores))

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n✅ XGBoost Model Evaluation Metrics:")
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)

# Feature importance (directly plot and save!)
ax = xgb.plot_importance(model, max_num_features=20, height=0.6, importance_type='weight')
ax.set_title('XGBoost Feature Importance')
plt.tight_layout()

# Save as PNG
plt.savefig('xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
print("✅ Feature importance plot saved as: xgboost_feature_importance.png")

# Show
plt.show()

input("\nPress Enter to exit...")
