import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ==========================
# Load Model
# ==========================
model = joblib.load("model/house_price_model.pkl")
model_columns = joblib.load("model/model_columns.pkl")

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 House Price Prediction")
st.write("Enter the property details below to estimate the house price.")

# ==========================
# User Inputs
# ==========================

col1, col2, col3 = st.columns(3)

with col1:
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("Bathrooms", 1, 10, 2, step=1)
    sqft_living = st.number_input("Living Area (sqft)", 500, 15000, 1800)
    sqft_lot = st.number_input("Lot Area (sqft)", 500, 100000, 5000)
    floors = st.number_input("Floors", 1, 5, 1, step=1)

with col2:
    waterfront = st.selectbox("Waterfront", [0, 1])
    view = st.slider("View", 0, 4, 0)
    condition = st.slider("Condition", 1, 5, 3)
    grade = st.slider("Grade", 1, 13, 7)
    sqft_above = st.number_input("Sqft Above", 500, 10000, 1500)

with col3:
    sqft_basement = st.number_input("Sqft Basement", 0, 5000, 0)
    yr_built = st.number_input("Year Built", 1900, 2026, 2000)
    yr_renovated = st.number_input("Year Renovated", 0, 2026, 0)
    zipcode = st.number_input("Zipcode", 98001, 98199, 98001)
    lat = st.number_input("Latitude", value=47.5)
    long = st.number_input("Longitude", value=-122.2)

st.markdown("---")

col4, col5 = st.columns(2)

with col4:
    sqft_living15 = st.number_input("Living Area (Nearest 15)", 500, 15000, 1800)
    sqft_lot15 = st.number_input("Lot Area (Nearest 15)", 500, 100000, 5000)

with col5:
    sale_date = st.date_input("Sale Date", datetime.today())

# ==========================
# Derived Features
# ==========================

house_age = sale_date.year - yr_built

renovated = 1 if yr_renovated > 0 else 0

sale_year = sale_date.year
sale_month = sale_date.month

# ==========================
# Prediction
# ==========================

if st.button("Predict Price"):

    sample = pd.DataFrame({
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "sqft_living": [sqft_living],
        "sqft_lot": [sqft_lot],
        "floors": [floors],
        "waterfront": [waterfront],
        "view": [view],
        "condition": [condition],
        "grade": [grade],
        "sqft_above": [sqft_above],
        "sqft_basement": [sqft_basement],
        "yr_built": [yr_built],
        "yr_renovated": [yr_renovated],
        "zipcode": [zipcode],
        "lat": [lat],
        "long": [long],
        "sqft_living15": [sqft_living15],
        "sqft_lot15": [sqft_lot15],
        "house_age": [house_age],
        "renovated": [renovated],
        "sale_year": [sale_year],
        "sale_month": [sale_month],
    })

    # One-hot encode zipcode
    sample = pd.get_dummies(sample, columns=["zipcode"], drop_first=True)

    # Match training columns
    sample = sample.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(sample)[0]

    st.success(f"### 💰 Predicted Price: ${prediction:,.2f}")