import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1️⃣ Load the data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2️⃣ Drop 'customerID' (not useful)
df.drop('customerID', axis=1, inplace=True)

# 3️⃣ Convert 'TotalCharges' to numeric (some blank values cause errors)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 4️⃣ Fill missing TotalCharges with 0 (or median)
df['TotalCharges'].fillna(0, inplace=True)

# 5️⃣ Encode 'Churn' target variable (Yes=1, No=0)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# 6️⃣ Identify categorical features
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# 7️⃣ Encode categorical features using LabelEncoder
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# 8️⃣ Split data into features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# 9️⃣ Standardize numeric features (optional but recommended)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1️⃣0️⃣ Final train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save preprocessed data if needed
pd.DataFrame(X_train).to_csv('X_train_preprocessed.csv', index=False)
pd.DataFrame(X_test).to_csv('X_test_preprocessed.csv', index=False)
pd.DataFrame(y_train).to_csv('y_train_preprocessed.csv', index=False)
pd.DataFrame(y_test).to_csv('y_test_preprocessed.csv', index=False)

print("✅ Data preprocessing complete and saved to CSV files!")
