import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NYC Airbnb Price Predictor", page_icon="🏙️")


@st.cache_resource
def load_assets():
    model = joblib.load("airbnb_price_model.joblib")
    with open("neighbourhood_lookup.json") as f:
        lookup = json.load(f)
    return model, lookup


model, lookup = load_assets()

st.title("NYC Airbnb Price Predictor")
st.write("Estimate the nightly price (USD) of an Airbnb listing in New York City. "
         "The model was trained on 2019 listings, so treat the result as a rough estimate.")

col1, col2 = st.columns(2)
with col1:
    borough = st.selectbox("Borough", sorted(lookup))
    hood = st.selectbox("Neighbourhood", sorted(lookup[borough]))
    room = st.selectbox("Room type", ["Entire home/apt", "Private room", "Shared room"])
    availability = st.slider("Days available per year", 0, 365, 180)
with col2:
    min_nights = st.number_input("Minimum nights", 1, 365, 3)
    n_reviews = st.number_input("Number of reviews", 0, 1000, 10)
    rpm = st.number_input("Reviews per month", 0.0, 30.0, 1.0, step=0.1)
    host_listings = st.number_input("Host's total listings", 1, 400, 1)

if st.button("Predict price", type="primary"):
    lat, lon = lookup[borough][hood]
    row = pd.DataFrame([{
        "latitude": lat, "longitude": lon, "minimum_nights": min_nights,
        "number_of_reviews": n_reviews, "reviews_per_month": rpm,
        "calculated_host_listings_count": host_listings, "availability_365": availability,
        "neighbourhood_group": borough, "neighbourhood": hood, "room_type": room,
    }])
    price = max(float(model.predict(row)[0]), 0.0)
    st.metric("Estimated price per night", f"${price:,.0f}")
    st.caption("Location uses the typical coordinates of the chosen neighbourhood.")
