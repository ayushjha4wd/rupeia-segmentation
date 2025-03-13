import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
file_path = "user_profiles_clean.csv"   
df = pd.read_csv(file_path)

# Display basic dataset info
print("\nDataset Info Before Preprocessing:")
print(df.info())

# Check for missing values
print("\nMissing Values Before Handling:")
print(df.isnull().sum())

# Fill missing values (if any) with the most frequent value
df.fillna(df.mode().iloc[0], inplace=True)

# Convert categorical numerical-like columns to numerical values
numeric_mappings = {
    "Annual Income": {
        "Less than ₹5 lakh": 2.5, "₹5 lakh – ₹10 lakh": 7.5,
        "₹10 lakh – ₹20 lakh": 15, "More than ₹20 lakh": 25
    },
    "Monthly Expenses": {
        "Less than 50% of my income": 0.4, "50–70% of my income": 0.6,
        "70–90% of my income": 0.8, "More than 90% of my income": 0.95
    },
    "Emergency Funds": {
        "No emergency savings": 0, "Less than 3 months of expenses saved": 3,
        "3–6 months of expenses saved": 6, "More than 6 months of expenses saved": 12
    },
    "Total Assets": {
        "Less than ₹5 lakh": 2.5, "₹5 lakh – ₹20 lakh": 12.5,
        "₹20 lakh – ₹50 lakh": 35, "More than ₹50 lakh": 55
    },
    "Liabilities": {
        "No loans": 0, "Less than ₹2 lakh": 1, "₹2 lakh – ₹5 lakh": 3.5,
        "More than ₹5 lakh": 6
    }
}

for col, mapping in numeric_mappings.items():
    df[col] = df[col].map(mapping)

# Label encoding for categorical features
categorical_cols = ["Risk Appetite", "Financial Goal", "Employment Type"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for future use

# Normalize numerical features using StandardScaler
scaler = StandardScaler()
numerical_cols = ["Annual Income", "Monthly Expenses", "Emergency Funds", "Total Assets", "Liabilities"]
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Display preprocessed dataset info
print("\nDataset Info After Preprocessing:")
print(df.info())

print("\nFirst 5 Rows After Preprocessing:")
print(df.head())

# Save the preprocessed data
df.to_csv("preprocessed_user_profiles.csv", index=False)

print("\nData Preprocessing Completed Successfully! Preprocessed data saved as 'preprocessed_user_profiles.csv'.")
