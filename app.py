import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Real Estate Investment Advisor")
st.write("Property Price Prediction and Investment Analysis")

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    return joblib.load(
        BASE_DIR / "models" / "random_forest_model.pkl"
    )

try:
    model = load_model()
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

city_states = {
    "Bangalore": "Karnataka",
    "Mumbai": "Maharashtra",
    "Delhi": "Delhi",
    "Chennai": "Tamil Nadu",
    "Hyderabad": "Telangana",
    "Pune": "Maharashtra",
    "Kolkata": "West Bengal"
}

st.header("Enter Property Details")

with st.form("property_form"):
    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("City", list(city_states))
        locality = st.text_input("Locality", "Central")
        property_type = st.selectbox(
            "Property Type",
            ["Apartment", "Villa", "Independent House"]
        )
        bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])
        area = st.number_input(
            "Area (Sq Ft)", min_value=100, value=1000
        )
        budget = st.number_input(
            "Budget (₹)", min_value=100000, value=5000000
        )
        furnished = st.selectbox(
            "Furnished Status",
            ["Semi-Furnished", "Furnished", "Unfurnished"]
        )

    with col2:
        floor_no = st.number_input(
            "Floor Number", min_value=0, value=2
        )
        total_floors = st.number_input(
            "Total Floors", min_value=1, value=5
        )
        age = st.number_input(
            "Property Age", min_value=0, value=5
        )
        schools = st.number_input(
            "Nearby Schools", min_value=0, value=3
        )
        hospitals = st.number_input(
            "Nearby Hospitals", min_value=0, value=2
        )
        parking = st.selectbox("Parking Space", ["Yes", "No"])
        security = st.selectbox("Security", ["Yes", "No"])
        facing = st.selectbox(
            "Facing", ["East", "West", "North", "South"]
        )

    submitted = st.form_submit_button("Analyze Property")

if submitted:
    if floor_no > total_floors:
        st.error("Floor number cannot exceed total floors.")
        st.stop()

    property_data = {
        "State": city_states[city],
        "City": city,
        "Locality": locality,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": area,
        "Year_Built": 2026 - age,
        "Furnished_Status": furnished,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Age_of_Property": age,
        "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals,
        "Public_Transport_Accessibility": "Good",
        "Parking_Space": parking,
        "Security": security,
        "Amenities": "Basic",
        "Facing": facing,
        "Owner_Type": "Individual",
        "Availability_Status": "Ready to Move",
        "Nearby_Facilities_Score": 7
    }

    input_data = pd.DataFrame([property_data])

    if hasattr(model, "feature_names_in_"):
        input_data = input_data.reindex(
            columns=list(model.feature_names_in_)
        )

    try:
        prediction = model.predict(input_data)
        predicted_price = float(prediction[0]) * 100000

        st.header("Property Price Prediction")
        st.metric(
            "Estimated Property Price",
            f"₹{predicted_price:,.2f}"
        )

        st.header("Estimated Property Value After 5 Years")

        growth_rate = st.slider(
            "Assumed Annual Appreciation Rate (%)",
            min_value=0.0,
            max_value=15.0,
            value=5.0,
            step=0.5
        )

        future_price = predicted_price * (
            1 + growth_rate / 100
        ) ** 5

        st.metric(
            "Projected Value After 5 Years",
            f"₹{future_price:,.2f}"
        )

        st.caption(
            "The five-year value is a scenario projection "
            "based on an assumed annual appreciation rate, "
            "not a trained future-price ML forecast."
        )

        st.header("Investment Budget Analysis")

        if predicted_price <= budget:
            st.success(
                "Within Budget: YES"
            )
        else:
            st.warning(
                "Within Budget: NO"
            )

        difference = budget - predicted_price

        if difference >= 0:
            st.write(
                f"Remaining Budget: ₹{difference:,.2f}"
            )
        else:
            st.write(
                f"Additional Budget Required: "
                f"₹{abs(difference):,.2f}"
            )

        st.info(
            "The budget indicator is not a trained "
            "investment-quality classification."
        )

        st.subheader("Property Details Used for Prediction")
        st.dataframe(input_data, width="stretch")

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()
st.header("Real Estate Market Insights")

data_path = BASE_DIR / "cleaned_housing_sample.csv"

if data_path.exists():
    try:
        df = pd.read_csv(data_path)

        if (
            "City" in df.columns
            and "Price_in_Lakhs" in df.columns
        ):
            df["Price_in_Lakhs"] = pd.to_numeric(
                df["Price_in_Lakhs"],
                errors="coerce"
            )

            city_prices = (
                df.groupby("City")["Price_in_Lakhs"]
                .mean()
                .dropna()
                .sort_values(ascending=False)
            )

            st.subheader(
                "Average Property Price by City (₹ Lakhs)"
            )
            st.bar_chart(city_prices)

        else:
            st.info(
                "Required columns for city-wise "
                "analysis are unavailable."
            )

    except Exception as error:
        st.warning(f"Market insights unavailable: {error}")
else:
    st.info("Cleaned housing dataset not found.")

st.divider()
st.caption("Real Estate ML Project | Python + Streamlit")
# PROPERTY FILTERING SECTION
st.divider()
st.header("🔎 Explore and Filter Properties")

if data_path.exists():
    try:
        properties = pd.read_csv(data_path)

        required = ["City", "BHK", "Price_in_Lakhs"]

        if all(col in properties.columns for col in required):
            properties["Price_in_Lakhs"] = pd.to_numeric(
                properties["Price_in_Lakhs"],
                errors="coerce"
            )
            properties["BHK"] = pd.to_numeric(
                properties["BHK"],
                errors="coerce"
            )

            properties = properties.dropna(subset=required)

            filter_city = st.selectbox(
                "Select City",
                ["All Cities"] + sorted(properties["City"].unique().tolist())
            )

            filter_bhk = st.selectbox(
                "Select BHK",
                ["All BHK"] + sorted(properties["BHK"].unique().tolist())
            )

            max_budget = st.number_input(
                "Maximum Property Budget (₹)",
                min_value=100000,
                value=10000000,
                step=100000
            )

            filtered = properties.copy()

            if filter_city != "All Cities":
                filtered = filtered[filtered["City"] == filter_city]

            if filter_bhk != "All BHK":
                filtered = filtered[filtered["BHK"] == filter_bhk]

            filtered = filtered[
                filtered["Price_in_Lakhs"] * 100000 <= max_budget
            ]

            st.success(f"Properties Found: {len(filtered)}")

            st.dataframe(
                filtered.head(100),
                width="stretch"
            )

            st.caption("Showing up to 100 matching properties.")
            st.download_button(
                "Download Filtered Properties (CSV)",
                data=filtered.head(100).to_csv(index=False),
                file_name="filtered_properties.csv",
                mime="text/csv"
            )

        else:
            st.warning("Required property filtering columns are missing.")

    except Exception as error:
        st.error(f"Filtering error: {error}")
else:
    st.warning("Cleaned housing dataset not found.")
            
