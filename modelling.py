import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
file_path = "cleaned_user_profiles.csv"  
df = pd.read_csv(file_path)

# Display dataset info
print("\nDataset Information:")
print(df.info())

# Handling missing values
df.fillna(df.mode().iloc[0], inplace=True)

# Convert categorical numerical-like columns to numerical values
import re

def convert_currency_to_float(value):
    if isinstance(value, str):
        numbers = re.findall(r'\d+', value.replace('₹', '').replace(',', ''))
        if len(numbers) == 1:
            return float(numbers[0])
        elif len(numbers) == 2:  # If range is given, take the average
            return (float(numbers[0]) + float(numbers[1])) / 2
    return float(value) if isinstance(value, (int, float)) else None

# Select numeric-like columns for conversion
numeric_cols_to_convert = ["Annual Income", "Monthly Expenses", "Emergency Funds", "Total Assets", "Liabilities"]

for col in numeric_cols_to_convert:
    df[col] = df[col].apply(convert_currency_to_float)

# Handle missing values again after conversion
df[numeric_cols_to_convert] = df[numeric_cols_to_convert].fillna(df[numeric_cols_to_convert].median())

# Encode categorical features using Label Encoding
categorical_cols = ["Risk Appetite", "Financial Goal", "Employment Type"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Standardize numerical features
scaler = StandardScaler()
df[numeric_cols_to_convert] = scaler.fit_transform(df[numeric_cols_to_convert])

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df[numeric_cols_to_convert])

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
df[['PCA1', 'PCA2']] = pca.fit_transform(df[numeric_cols_to_convert])

# Plot the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["PCA1"], y=df["PCA2"], hue=df["Cluster"], palette="viridis", alpha=0.7)
plt.title("Clustering Visualization using PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()

# Display first few rows of clustered data
print("\nClustered Data (First 5 Rows):")
print(df.head())

# Save the processed dataset with clusters
df.to_csv("clustered_user_profiles.csv", index=False)
print("\nClustered data saved as 'clustered_user_profiles.csv'")
