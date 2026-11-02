
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned housing dataset
df = pd.read_csv("cleaned_housing_data.csv")

# Check dataset
print("Dataset Shape:", df.shape)
print("\nAvailable Columns:")
print(df.columns.tolist())

print("\nDataset loaded successfully!")

# EDA 1: Average Property Price by City

city_prices = (
    df.groupby("City")["Price_in_Lakhs"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))
sns.barplot(
    x=city_prices.values,
    y=city_prices.index
)

plt.title("Top 10 Cities by Average Property Price")
plt.xlabel("Average Property Price (Lakhs)")
plt.ylabel("City")
plt.tight_layout()

plt.savefig("eda_city_prices.png")
plt.show()

# EDA 2: Property Size vs Price

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df.sample(n=5000, random_state=42),
    x="Size_in_SqFt",
    y="Price_in_Lakhs",
    alpha=0.5
)

plt.title("Property Size vs Property Price")
plt.xlabel("Property Size (Sq Ft)")
plt.ylabel("Property Price (Lakhs)")
plt.tight_layout()

plt.savefig("eda_size_vs_price.png")
plt.show()

# Visualization 3: Property Type Distribution

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="Property_Type",
    order=df["Property_Type"].value_counts().index
)

plt.title("Distribution of Property Types")
plt.xlabel("Property Type")
plt.ylabel("Number of Properties")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualization 4: Average Price by Property Type

avg_price = (
    df.groupby("Property_Type")["Price_in_Lakhs"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=avg_price.index,
    y=avg_price.values
)

plt.title("Average Property Price by Property Type")
plt.xlabel("Property Type")
plt.ylabel("Average Price (Lakhs)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Visualization 5: Property Price Distribution

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Price_in_Lakhs",
    bins=30,
    kde=True
)

plt.title("Distribution of Property Prices")
plt.xlabel("Property Price (Lakhs)")
plt.ylabel("Number of Properties")
plt.tight_layout()
plt.show()
# Visualization 5: Property Price Distribution

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Price_in_Lakhs",
    bins=30,
    kde=True
)

plt.title("Distribution of Property Prices")
plt.xlabel("Property Price (Lakhs)")
plt.ylabel("Number of Properties")
plt.tight_layout()
plt.show()

# Visualization 6: Average Price by Furnishing Status

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="Furnished_Status",
    y="Price_in_Lakhs",
    estimator="mean",
    errorbar=None
)

plt.title("Average Property Price by Furnishing Status")
plt.xlabel("Furnishing Status")
plt.ylabel("Average Property Price (Lakhs)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Visualization 7: Average Price by Number of Bedrooms

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="BHK",
    y="Price_in_Lakhs",
    estimator="mean",
    errorbar=None
)

plt.title("Average Property Price by Number of Bedrooms")
plt.xlabel("Number of Bedrooms (BHK)")
plt.ylabel("Average Property Price (Lakhs)")
plt.tight_layout()
plt.show()
# Visualization 7: Average Price by Number of Bedrooms

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="BHK",
    y="Price_in_Lakhs",
    estimator="mean",
    errorbar=None
)

plt.title("Average Property Price by Number of Bedrooms")
plt.xlabel("Number of Bedrooms (BHK)")
plt.ylabel("Average Property Price (Lakhs)")
plt.tight_layout()
plt.show()

# Visualization 8: Average Property Price by Parking Space

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="Parking_Space",
    y="Price_in_Lakhs",
    estimator="mean",
    errorbar=None
)

plt.title("Average Property Price by Parking Space")
plt.xlabel("Number of Parking Spaces")
plt.ylabel("Average Property Price (Lakhs)")
plt.tight_layout()
plt.show()

# Visualization 9: Average Property Price by Security Status

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="Security",
    y="Price_in_Lakhs",
    estimator="mean",
    errorbar=None
)

plt.title("Average Property Price by Security Status")
plt.xlabel("Security Status")
plt.ylabel("Average Property Price (Lakhs)")
plt.tight_layout()
plt.show()

# Visualization 10: Property Age vs Average Price

plt.figure(figsize=(10, 6))

sns.lineplot(
    data=df,
    x="Age_of_Property",
    y="Price_in_Lakhs",
    estimator="mean",
    errorbar=None
)

plt.title("Average Property Price by Property Age")
plt.xlabel("Age of Property (Years)")
plt.ylabel("Average Property Price (Lakhs)")
plt.tight_layout()
plt.show()

# Visualization 11: Average Property Price by Parking Space

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="Parking_Space",
    y="Price_in_Lakhs",
    estimator="mean",
    errorbar=None
)

plt.title("Average Property Price by Parking Space")
plt.xlabel("Number of Parking Spaces")
plt.ylabel("Average Property Price (Lakhs)")
plt.tight_layout()
plt.show()

# Visualization 12: Average Property Price by Owner Type

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="Owner_Type",
    y="Price_in_Lakhs",
    estimator="mean",
    errorbar=None
)

plt.title("Average Property Price by Owner Type")
plt.xlabel("Owner Type")
plt.ylabel("Average Property Price (Lakhs)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Visualization 13: Average Property Price by Availability Status

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="Availability_Status",
    y="Price_in_Lakhs",
    estimator="mean",
    errorbar=None
)

plt.title("Average Property Price by Availability Status")
plt.xlabel("Availability Status")
plt.ylabel("Average Property Price (Lakhs)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Visualization 14: Average Property Price by Nearby Schools

plt.figure(figsize=(10, 6))

sns.lineplot(
    data=df,
    x="Nearby_Schools",
    y="Price_in_Lakhs",
    estimator="mean",
    errorbar=None
)

plt.title("Average Property Price by Nearby Schools")
plt.xlabel("Number of Nearby Schools")
plt.ylabel("Average Property Price (Lakhs)")
plt.tight_layout()
plt.show()

# Visualization 15: Average Property Price by Nearby Hospitals

plt.figure(figsize=(10, 6))

sns.lineplot(
    data=df,
    x="Nearby_Hospitals",
    y="Price_in_Lakhs",
    estimator="mean",
    errorbar=None
)

plt.title("Average Property Price by Nearby Hospitals")
plt.xlabel("Number of Nearby Hospitals")
plt.ylabel("Average Property Price (Lakhs)")
plt.tight_layout()
plt.show()

# Visualization 16: Property Price by Public Transport Accessibility

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="Public_Transport_Accessibility",
    y="Price_in_Lakhs",
    estimator="mean",
    errorbar=None
)

plt.title("Average Property Price by Public Transport Accessibility")
plt.xlabel("Public Transport Accessibility")
plt.ylabel("Average Property Price (Lakhs)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Visualization 17: Correlation Heatmap

plt.figure(figsize=(14, 10))

numeric_df = df.select_dtypes(include="number")

correlation_matrix = numeric_df.corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numerical Features")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Visualization 18: Distribution of Property Price per Sq Ft

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Price_per_SqFt",
    bins=30,
    kde=True
)

plt.title("Distribution of Property Price per Sq Ft")
plt.xlabel("Property Price per Sq Ft")
plt.ylabel("Number of Properties")
plt.tight_layout()
plt.show()














