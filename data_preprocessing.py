
import pandas as pd

# Load the housing dataset
df = pd.read_csv("india_housing_prices.csv")

# Check dataset structure
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# Remove duplicate records
df = df.drop_duplicates()

print("\nShape after removing duplicates:", df.shape)
print("Remaining duplicates:", df.duplicated().sum())

# Handle missing values
numeric_cols = df.select_dtypes(include="number").columns
categorical_cols = df.select_dtypes(include=["object", "category"]).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nTotal missing values after cleaning:", df.isnull().sum().sum())

# Feature Engineering

# Calculate price per square foot
df["Calculated_Price_per_SqFt"] = (
    df["Price_in_Lakhs"] * 100000
) / df["Size_in_SqFt"].replace(0, float("nan"))

# Calculate nearby facilities score
df["Nearby_Facilities_Score"] = (
    df["Nearby_Schools"] + df["Nearby_Hospitals"]
)

print("\nNew Features Created:")
print(df[[
    "Calculated_Price_per_SqFt",
    "Nearby_Facilities_Score"
]].head())

# Save cleaned and feature-engineered dataset
df.to_csv("cleaned_housing_data.csv", index=False)

print("\nCleaned dataset saved successfully!")
print("Final dataset shape:", df.shape)



