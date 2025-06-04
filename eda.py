# 🚀 Data Science Project: CLV Prediction for SaaS Customers

# 1️⃣ Import essential libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2️⃣ Load the dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 3️⃣ Basic exploration
print(df.head())
print("\nShape of the dataset:", df.shape)
print("\nColumns in the dataset:", df.columns.tolist())
print("\nMissing values in each column:\n", df.isnull().sum())

# 4️⃣ Convert 'TotalCharges' to numeric (fix missing data)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("\nMissing values after conversion:\n", df.isnull().sum())

# 5️⃣ Drop rows with missing 'TotalCharges'
df = df.dropna(subset=['TotalCharges'])
print("\nShape after dropping NaNs:", df.shape)

# 6️⃣ Summary statistics
print("\nSummary statistics:\n", df.describe())

# 7️⃣ Visualize distributions
plt.figure(figsize=(8, 4))
sns.histplot(df['TotalCharges'], bins=30, kde=True)
plt.title('Total Charges Distribution')
plt.tight_layout()
plt.savefig('plot_total_charges_distribution.png')
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(df['MonthlyCharges'], bins=30, kde=True)
plt.title('Monthly Charges Distribution')
plt.tight_layout()
plt.savefig('plot_monthly_charges_distribution.png')
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(df['tenure'], bins=30, kde=True)
plt.title('Tenure Distribution')
plt.tight_layout()
plt.savefig('plot_tenure_distribution.png')
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x='Churn', y='TotalCharges', data=df)
plt.title('Total Charges by Churn')
plt.tight_layout()
plt.savefig('plot_total_charges_by_churn.png')
plt.show()

# 8️⃣ Create Customer Lifetime Value (CLV)
df['CLV'] = df['MonthlyCharges'] * df['tenure']
print("\nSample CLV values:\n", df[['customerID', 'MonthlyCharges', 'tenure', 'CLV']].head())
print("\nCLV summary statistics:\n", df['CLV'].describe())

plt.figure(figsize=(8, 4))
sns.histplot(df['CLV'], bins=30, kde=True)
plt.title('Customer Lifetime Value (CLV) Distribution')
plt.tight_layout()
plt.savefig('plot_clv_distribution.png')
plt.show()

# 9️⃣ Prepare features for modeling
features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
            'InternetService', 'Contract', 'PaymentMethod', 'tenure', 'MonthlyCharges', 
            'TotalCharges', 'Churn']

df_model = df[features]
df_model['Churn'] = df_model['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
df_model_encoded = pd.get_dummies(df_model, drop_first=True)

print("\nFirst rows of encoded data:\n", df_model_encoded.head())
print("\nShape of the dataset after encoding:", df_model_encoded.shape)

# 🔟 Model training and evaluation
X = df_model_encoded
y = df['CLV']

print("\n✅ Features and Target shapes:")
print("X:", X.shape)
print("y:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\n✅ Train-test split shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n✅ Model Evaluation Metrics:")
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)

input("\nPress Enter to exit...")