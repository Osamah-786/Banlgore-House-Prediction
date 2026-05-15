import streamlit as st
import pandas as pd
import pickle

# Load cleaned data
data = pd.read_csv("Cleaned_data.csv")

# Load trained model
model = pickle.load(open("RidgeModel.pkl", "rb"))

# Title
st.title("🏠 Bangalore House Price Prediction")

st.write("Enter house details below")

# Location dropdown
locations = sorted(data["location"].unique())

location = st.selectbox("Choose Location", locations)

bhk = st.number_input("BHK", min_value=1, max_value=20, step=1)

bath = st.number_input("Bathrooms", min_value=1, max_value=20, step=1)

sqft = st.number_input("Total Square Feet", min_value=100.0)

# Predict button
if st.button("Predict Price"):

    # Create size column
    size = str(bhk) + " BHK"

    # Create dataframe
    input_data = pd.DataFrame(
        [[location, size, sqft, bath, bhk]],
        columns=["location", "size", "total_sqft", "bath", "bhk"]
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Price: ₹ {round(prediction, 2)} Lakhs")
