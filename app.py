import streamlit as st
import pandas as pd
import pickle

with open("xgb_housing_model.pkl", "rb") as f:
    model = pickle.load(f)


st.title("🏡 California Housing Price Predictor (XGBoost)")
st.markdown("### Predict house prices based on location, demographics, and housing features.")
st.write("Enter the details below and click **Predict House Price** to get an estimated value.")

st.subheader("Enter House Details")

col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Longitude", -125.0, -114.0, -120.0)
    latitude = st.number_input("Latitude", 32.0, 42.0, 37.0)
    housing_median_age = st.slider("Housing Median Age", 1, 52, 20)
    total_rooms = st.number_input("Total Rooms", 1.0, 40000.0, 2000.0)
    total_bedrooms = st.number_input("Total Bedrooms", 1.0, 10000.0, 500.0)

with col2:
    population = st.number_input("Population", 1.0, 50000.0, 1000.0)
    households = st.number_input("Households", 1.0, 10000.0, 500.0)
    median_income = st.slider("Median Income (in 10k USD)", 0.5, 15.0, 4.0)
    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
    )

ocean_dummy = {
    "ocean_proximity_INLAND": 0,
    "ocean_proximity_ISLAND": 0,
    "ocean_proximity_NEAR BAY": 0,
    "ocean_proximity_NEAR OCEAN": 0
}
if ocean_proximity != "<1H OCEAN":
    ocean_dummy[f"ocean_proximity_{ocean_proximity}"] = 1

input_data = pd.DataFrame([[
    longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
    population, households, median_income,
    ocean_dummy["ocean_proximity_INLAND"],
    ocean_dummy["ocean_proximity_ISLAND"],
    ocean_dummy["ocean_proximity_NEAR BAY"],
    ocean_dummy["ocean_proximity_NEAR OCEAN"]
]], columns=[
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    "ocean_proximity_INLAND", "ocean_proximity_ISLAND",
    "ocean_proximity_NEAR BAY", "ocean_proximity_NEAR OCEAN"
])

if st.button("Predict House Price"):
    prediction = model.predict(input_data)[0]

    st.success("Prediction Complete!")

    st.metric(label="Predicted Median House Value", value=f"${prediction:,.2f}")

    st.markdown("---")
   