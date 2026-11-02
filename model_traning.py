# Real Estate Price Prediction - Model Training

# Import libraries
import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("cleaned_housing_data.csv")

# Check dataset
print("Dataset Shape:", df.shape)
print("\nDataset Columns:")
print(df.columns.tolist())

print("\nDataset loaded successfully!")
# Step 4: Select Features and Target

# Target variable
y = df["Price_in_Lakhs"]

# Features used for prediction
feature_columns = [
    "BHK",
    "Size_in_SqFt",
    "Floor_No",
    "Total_Floors",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals"
]

X = df[feature_columns]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)
print("\nFeatures and target selected successfully!") 
# Step 5: Split Dataset into Training and Testing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

print("\nDataset split successfully!")
# Train Linear Regression Model
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")
# Predict property prices
y_pred = lr_model.predict(X_test)

print("\nPredictions generated successfully!")
print("First 5 Predicted Prices:", y_pred[:5])
print("First 5 Actual Prices:", y_test.iloc[:5].values)
# Evaluate Linear Regression Model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Model Evaluation")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)
# Check target price distribution
print("\nTarget Price Statistics:")
print(y.describe())

print("\nCorrelation with Property Price:")
print(df.select_dtypes(include="number")
      .corr()["Price_in_Lakhs"]
      .sort_values(ascending=False))
# Check relationship between price and property size
print("\nProperty Size vs Price Correlation:")
print(df["Size_in_SqFt"].corr(df["Price_in_Lakhs"]))

# Check price variation across cities
print("\nAverage Price by City:")
print(df.groupby("City")["Price_in_Lakhs"].mean())
# Check how much property prices vary across cities
city_prices = df.groupby("City")["Price_in_Lakhs"].mean()

print("\nCity Average Price Statistics:")
print(city_prices.describe())

print("\nOverall Price Standard Deviation:")
print(df["Price_in_Lakhs"].std())
# Check price calculation consistency
calculated_price = (
    df["Size_in_SqFt"] * df["Price_per_SqFt"] / 100000
)

difference = abs(
    df["Price_in_Lakhs"] - calculated_price
)

print("\nPrice Calculation Check:")
print("Mean Absolute Difference:", difference.mean())
print("Maximum Difference:", difference.max())
print("Matching Prices:", (difference < 0.1).mean() * 100, "%")
# Inspect original property price values
print("\nOriginal Property Data:")
print(df[[
    "Size_in_SqFt",
    "Price_per_SqFt",
    "Price_in_Lakhs"
]].head(10))
print("\nPrice Per SqFt Statistics:")
print(df["Price_per_SqFt"].describe())

print("\nProperty Price Statistics:")
print(df["Price_in_Lakhs"].describe())
# Check price per square foot calculation
actual_rate = (
    df["Price_in_Lakhs"] / df["Size_in_SqFt"]
)

difference = abs(
    df["Price_per_SqFt"] - actual_rate
)

print("\nPrice Per SqFt Verification:")
print("Mean Difference:", difference.mean())
print("Maximum Difference:", difference.max())
print("Matching Within 0.005:",
      (difference <= 0.005).mean() * 100, "%")
# Check correlation with property price
print("\nCorrelation with Property Price:")

numeric_df = df.select_dtypes(include="number")

correlation = numeric_df.corr()["Price_in_Lakhs"]

print(correlation.sort_values(ascending=False))
print("\nAverage Property Price by City:")
print(
    df.groupby("City")["Price_in_Lakhs"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print("\nAverage Property Price by Property Type:")
print(
    df.groupby("Property_Type")["Price_in_Lakhs"]
    .mean()
)
print("\nAverage Property Price by Furnishing Status:")
print(
    df.groupby("Furnished_Status")["Price_in_Lakhs"]
    .mean()
    .sort_values(ascending=False)
)
print("\nAverage Property Price by BHK:")
print(
    df.groupby("BHK")["Price_in_Lakhs"]
    .agg(["count", "mean", "min", "max"])
    .sort_index()
)
print("\nProperty Size vs Price Correlation:")

print(
    df[["Size_in_SqFt", "Price_in_Lakhs"]]
    .corr()
)

print("\nAverage Price by Property Size Range:")

df["Size_Range"] = pd.cut(
    df["Size_in_SqFt"],
    bins=5
)

print(
    df.groupby("Size_Range", observed=True)["Price_in_Lakhs"]
    .mean()
)
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

print("\nBaseline Model Evaluation:")
print("MAE:", mean_absolute_error(y_test, baseline_predictions))
print("R2 Score:", r2_score(y_test, baseline_predictions))
print("\nDataset Quality Check:")

print("Duplicate Rows:", df.duplicated().sum())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Shape:", df.shape)
print("\nDuplicate Properties (excluding ID):")

duplicate_count = df.drop(
    columns=["ID"]
).duplicated().sum()

print("Duplicate Count:", duplicate_count)
print("\nProperty Price Distribution:")

price_ranges = pd.cut(
    df["Price_in_Lakhs"],
    bins=[0, 100, 200, 300, 400, 500],
    include_lowest=True
)

print(price_ranges.value_counts().sort_index())
print("\nPrice Correlation Investigation:")

columns_to_check = [
    "Size_in_SqFt",
    "BHK",
    "Age_of_Property",
    "Floor_No",
    "Price_per_SqFt"
]

print(
    df[columns_to_check + ["Price_in_Lakhs"]]
    .corr()["Price_in_Lakhs"]
    .sort_values(ascending=False)
)
print("\nDataset Price Relationship Check:")

print(
    df.groupby("City")["Price_in_Lakhs"]
    .agg(["count", "mean", "std"])
    .sort_values("mean", ascending=False)
    .head(10)
)
print("\nCity-wise Price Variation:")

city_means = df.groupby("City")["Price_in_Lakhs"].mean()

print("Lowest City Average:", city_means.min())
print("Highest City Average:", city_means.max())
print("Difference:", city_means.max() - city_means.min())
print("Overall Average:", df["Price_in_Lakhs"].mean())
print("\nPrice per SqFt Data Leakage Check:")

print(
    df[[
        "Price_in_Lakhs",
        "Price_per_SqFt",
        "Calculated_Price_per_SqFt"
    ]].corr()
)
print("\nPreparing Features for Model Training:")

X = df.drop(
    columns=[
        "Price_in_Lakhs",
        "Price_per_SqFt",
        "Calculated_Price_per_SqFt",
        "ID",
        "Size_Range"
    ],
    errors="ignore"
)

y = df["Price_in_Lakhs"]

print("Feature Shape:", X.shape)
print("Target Shape:", y.shape)
print("Remaining Features:", X.columns.tolist())
print("\nIdentifying Feature Types:")

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

print("Numerical Features:", numerical_features)
print("Categorical Features:", categorical_features)

print("Total Numerical:", len(numerical_features))
print("Total Categorical:", len(categorical_features))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain-Test Split Completed:")
print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)
print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nPreprocessing Pipeline Created Successfully!")
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

linear_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

print("\nTraining Linear Regression Model...")

linear_model.fit(X_train, y_train)

print("Linear Regression Model Trained Successfully!")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("\nEvaluating Linear Regression Model...")

y_pred = linear_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)
print("\nModel Performance Comparison")

print("Baseline MAE:", mean_absolute_error(y_test, baseline_predictions))
print("Linear Regression MAE:", mae)

print("Baseline R2:", r2_score(y_test, baseline_predictions))
print("Linear Regression R2:", r2)
print("\nTarget Variable Analysis:")

print("Training Price Mean:", y_train.mean())
print("Testing Price Mean:", y_test.mean())

print("Training Price Standard Deviation:", y_train.std())
print("Testing Price Standard Deviation:", y_test.std())

print("\nPrice Distribution:")
print(y_train.describe())
print("\nProperty Price Percentile Analysis:")

print(
    df["Price_in_Lakhs"].quantile(
        [0.01, 0.05, 0.10, 0.25, 0.50,
         0.75, 0.90, 0.95, 0.99]
    )
)

print("\nPrice Variability:")
print("Minimum:", y_train.min())
print("Maximum:", y_train.max())
print("Median:", y_train.median())
print("\nTarget Distribution Skewness:")

print("Skewness:", df["Price_in_Lakhs"].skew())

print("\nTarget Variance:")
print(df["Price_in_Lakhs"].var())

print("\nTarget Standard Deviation:")
print(df["Price_in_Lakhs"].std())
print("\nFeature-Target Correlation Analysis:")

numeric_features_df = X_train.select_dtypes(
    include=["number"]
)

correlations = numeric_features_df.corrwith(y_train)

print(
    correlations.reindex(
        correlations.abs().sort_values(
            ascending=False
        ).index
    )
)

print("\nStrongest Absolute Correlation:")
print(correlations.abs().max())
print("\nCategorical Feature Price Analysis:")

for col in categorical_features:
    grouped = df.groupby(col)["Price_in_Lakhs"].mean()

    print(f"\nFeature: {col}")
    print("Categories:", df[col].nunique())
    print("Price Mean Variation:", grouped.max() - grouped.min())
    print("\nTarget Price Distribution:")

print(
    df["Price_in_Lakhs"].describe()
)

print("\nTarget Price Quantiles:")

print(
    df["Price_in_Lakhs"].quantile(
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )
)
print("\nActual Price Statistics:")
predictions = linear_model.predict(X_test)
print("Mean:", y_test.mean())
print("Standard Deviation:", y_test.std())

print("\nPredicted Price Statistics:")
print("Mean:", predictions.mean())
print("Standard Deviation:", predictions.std())

print("\nFirst 10 Actual vs Predicted Prices:")

comparison = pd.DataFrame({
    "Actual": y_test.iloc[:10].values,
    "Predicted": predictions[:10]
})

print(comparison)
print("\nPrediction Range Analysis:")

print("Actual Minimum:", y_test.min())
print("Actual Maximum:", y_test.max())

print("Predicted Minimum:", predictions.min())
print("Predicted Maximum:", predictions.max())

print("\nPrediction Standard Deviation:", predictions.std())
from sklearn.ensemble import RandomForestRegressor

print("\nTraining Random Forest Model...")

rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)

print("Random Forest Model Trained Successfully!")
print("\nEvaluating Random Forest Model...")

rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

print("Random Forest MAE:", rf_mae)
print("Random Forest RMSE:", rf_rmse)
print("Random Forest R2 Score:", rf_r2)
print("\nFinal Model Comparison:")

model_comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [mae, rf_mae],
    "RMSE": [rmse, rf_rmse],
    "R2 Score": [r2, rf_r2]
})

print(model_comparison)
print("\nRandom Forest Overfitting Check:")

rf_train_predictions = rf_model.predict(X_train)

print("Training R2:", r2_score(y_train, rf_train_predictions))
print("Testing R2:", rf_r2)

print("Training MAE:",
      mean_absolute_error(y_train, rf_train_predictions))

print("Testing MAE:", rf_mae)
print("\nNumeric Feature Correlation with Price:")

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
)

correlations = numeric_features.corrwith(y_train)

print(correlations.sort_values(
    key=abs,
    ascending=False
))
print("\nCategorical Features:")

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
)

print(categorical_features.columns.tolist())

print("\nNumber of Unique Categories:")

print(categorical_features.nunique())
print("\nAmenities Feature Analysis:")

print("\nTop 10 Most Frequent Amenities:")
print(X_train["Amenities"].value_counts().head(10))

print("\nMissing Amenities Values:")
print(X_train["Amenities"].isna().sum())
print("\nAverage Property Price by Amenities:")

amenities_price = (
    df.groupby("Amenities")["Price_in_Lakhs"]
    .agg(["count", "mean"])
    .sort_values("count", ascending=False)
    .head(10)
)

print(amenities_price)
print("\nLocation Price Analysis:")

location_columns = [
    col for col in df.columns
    if "location" in col.lower() or "city" in col.lower()
]

print("Location Columns:", location_columns)

for col in location_columns:
    print(f"\nAverage Price by {col}:")
    print(
        df.groupby(col)["Price_in_Lakhs"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    print("\nDataset Duplicate Analysis:")

print("Total Rows:", len(df))

print("Duplicate Rows:", df.duplicated().sum())

print("Duplicate Percentage:",
      round(df.duplicated().mean() * 100, 2))
print("\nProperty Size vs Price Analysis:")

print("Correlation:",
      df["Size_in_SqFt"].corr(df["Price_in_Lakhs"]))

print("\nAverage Price by Size Range:")

size_groups = pd.cut(
    df["Size_in_SqFt"],
    bins=5
)

print(
    df.groupby(size_groups, observed=True)["Price_in_Lakhs"]
    .agg(["count", "mean"])
)
import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(rf_model, "models/random_forest_model.pkl")

print("Random Forest Model Saved Successfully!")
print("\nChecking Saved Model File...")

model_path = "models/random_forest_model.pkl"

if os.path.exists(model_path):
    print("Model File Verified Successfully!")
else:
    print("Model File Not Found!")