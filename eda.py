import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "cleaned_user_profiles.csv"  
df = pd.read_csv(file_path)

# Display basic dataset info
print("\nDataset Information:")
print(df.info())

# Check for missing values
print("\nMissing Values in Dataset:")
print(df.isnull().sum())

# Fill missing values (if any) with the most frequent value
df.fillna(df.mode().iloc[0], inplace=True)

# Convert categorical numerical-like columns to numeric values
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

# Descriptive statistics
print("\nSummary Statistics:")
print(df.describe())

# Visualization: Distribution of Annual Income
plt.figure(figsize=(8, 5))
sns.histplot(df["Annual Income"], bins=10, kde=True, color="blue")
plt.title("Distribution of Annual Income")
plt.xlabel("Annual Income (Lakhs)")
plt.ylabel("Frequency")
plt.show()

# Visualization: Risk Appetite Count Plot
plt.figure(figsize=(8, 5))
sns.countplot(x="Risk Appetite", data=df, palette="coolwarm", order=df["Risk Appetite"].value_counts().index)
plt.title("Risk Appetite Distribution")
plt.xlabel("Risk Appetite")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

print("\nEDA Completed Successfully!")
