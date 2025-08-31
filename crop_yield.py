# ============================
#  Import Libraries
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plots
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10,6)

# ============================
#  Load Dataset
# ============================

df = pd.read_csv("crop_yield.csv")

print("Dataset Shape:", df.shape)
print("\nPreview:")
print(df.head())

# ============================
#  Basic Info
# ============================
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe(include="all"))

# ============================
#  Data Cleaning & Preprocessing
# ============================
# Handle missing values
# ============================
#  Handle Missing Values
# ============================
# Drop rows with missing target "Yield"
df = df.dropna(subset=["Yield"])

# Fill numeric NaN with median
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical NaN with mode
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nAfter Handling Missing Values:")
print(df.isnull().sum())

# ============================
#  Encode Categorical Variables
# ============================
from sklearn.preprocessing import LabelEncoder

label_enc = LabelEncoder()
for col in ["State", "Season", "Crop"]:
    if col in df.columns:
        df[col] = label_enc.fit_transform(df[col])

print("\nData After Encoding:")
print(df.head())

df.to_csv("clean_crop_yield.csv", index=False)
print("\n✅ Cleaned dataset saved as 'clean_crop_yield.csv'")

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Features")
plt.show()
plt.savefig("correlation_matrix.png")

# ============================
# Visualizations & EDA
# ============================
# Distribution of Yield

sns.histplot(df["Yield"], kde=True, bins=30, color="green")
plt.title("Distribution of Crop Yield")
plt.xlabel("Yield")
plt.ylabel("Frequency")
plt.show()

# ============================
#  Rainfall vs Yield
# ============================
if "Annual_Rainfall" in df.columns:
    sns.scatterplot(x="Annual_Rainfall", y="Yield", data=df, hue="Season", alpha=0.6)
    plt.title("Annual Rainfall vs Yield")
    plt.show()

# ============================
#  Fertilizer vs Yield
# ============================
if "Fertilizer" in df.columns:
    sns.scatterplot(x="Fertilizer", y="Yield", data=df, hue="Crop", alpha=0.6, legend=False)
    plt.title("Fertilizer Usage vs Yield")
    plt.show()

# ============================
#  Average Yield per Crop
# ============================
if "Crop" in df.columns:
    avg_yield_crop = df.groupby("Crop")["Yield"].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=avg_yield_crop.values, y=avg_yield_crop.index, palette="viridis")
    plt.title("Top 10 Crops by Average Yield")
    plt.xlabel("Average Yield")
    plt.ylabel("Crop")
    plt.show()

# ============================
#  Average Yield per State
# ============================
if "State" in df.columns:
    avg_yield_state = df.groupby("State")["Yield"].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=avg_yield_state.values, y=avg_yield_state.index, palette="magma")
    plt.title("Top 10 States by Average Yield")
    plt.xlabel("Average Yield")
    plt.ylabel("State")
    plt.show()
