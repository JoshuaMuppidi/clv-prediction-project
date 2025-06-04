import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance
import joblib
import os

print("\nCurrent Working Directory:", os.getcwd())

# Load data
df = pd.read_csv('C:/Users/joshu/OneDrive/Desktop/clv_prediction_project/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
df['CLV'] = df['MonthlyCharges'] * df['tenure']

# Select features
features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'InternetService', 'Contract', 'PaymentMethod', 'tenure', 'MonthlyCharges',
            'TotalCharges', 'Churn']

df_model = df[features]
df_model['Churn'] = df_model['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
df_model_encoded = pd.get_dummies(df_model, drop_first=True)

X = df_model_encoded
y = df['CLV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define XGBoost and parameter grid
xgb = XGBRegressor(random_state=42)
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

# Perform GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='r2',
    cv=3,
    verbose=2,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Results
print("\n✅ Best Parameters from GridSearchCV:")
print(grid_search.best_params_)
print("\n✅ Best R2 Score from GridSearchCV:")
print(grid_search.best_score_)

# Evaluate best model on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\n✅ Tuned XGBoost Model Evaluation on Test Data:")
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)

# Plot feature importance
plt.figure(figsize=(10, 6))
plot_importance(best_model, importance_type='weight', title='XGBoost Tuned Feature Importance')
plt.tight_layout()
plt.savefig("xgboost_tuned_feature_importance.png")
plt.show()

# Save the best model
joblib.dump(best_model, 'xgb_model.pkl')

# Save feature names
joblib.dump(X.columns.tolist(), 'model_features.pkl')

print("\n✅ Model and feature names saved to 'xgb_model.pkl' and 'model_features.pkl'!")
