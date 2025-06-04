import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np

# Load dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Data Cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Feature Engineering
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
df_model = pd.get_dummies(df.drop('customerID', axis=1), drop_first=True)

# Features & target
X = df_model.drop('Churn', axis=1)
y = df_model['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomizedSearchCV parameter grid
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
}

# Initialize model
xgb_model = xgb.XGBRegressor(random_state=42)

# RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=xgb_model,
                                   param_distributions=param_dist,
                                   n_iter=20,
                                   scoring='r2',
                                   cv=3,
                                   verbose=2,
                                   random_state=42,
                                   n_jobs=-1)

random_search.fit(X_train, y_train)

# Best parameters and score
print("✅ Best Parameters from RandomizedSearchCV:")
print(random_search.best_params_)

print("\n✅ Best R2 Score from RandomizedSearchCV:")
print(random_search.best_score_)

# Evaluate best model on test set
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n✅ Tuned XGBoost Model Evaluation on Test Data:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R2 Score: {r2}")

# Feature importance
xgb.plot_importance(best_model, max_num_features=15, importance_type='weight')
plt.title('XGBoost Tuned Feature Importance')
plt.show()

# Best model from RandomizedSearchCV
best_model = random_search.best_estimator_

# Perform 5-fold cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')

print("\n✅ Cross-Validation R² Scores for Tuned XGBoost Model:")
print(cv_scores)

print("\n✅ Mean R² Score from Cross-Validation:")
print(np.mean(cv_scores))