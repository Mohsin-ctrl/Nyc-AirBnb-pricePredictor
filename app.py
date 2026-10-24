import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

CV_RMSE = 86

ALGORITHM_NAMES = {
    "HistGradientBoostingRegressor": "Histogram Gradient Boosting",
    "ExtraTreesRegressor": "Extra Trees",
    "MLPRegressor": "MLP Neural Network",
    "Ridge": "Ridge Regression",
    "LinearSVR": "Linear SVR",
}

st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 820px; }
    .hero {
        background: linear-gradient(135deg, #ff5a5f 0%, #d93a63 100%);
        border-radius: 16px;
        padding: 2rem 2rem 1.7rem 2rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(217, 58, 99, 0.25);
    }
    .hero-title {
        color: #ffffff;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        line-height: 1.15;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .hero-sub {
        color: rgba(255, 255, 255, 0.92);
        font-size: 1.02rem;
        margin: 0.6rem 0 0 0;
    }
    .badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        color: #ffffff;
        border-radius: 999px;
        padding: 0.18rem 0.8rem;
        font-size: 0.8rem;
        margin: 0.9rem 0.4rem 0 0;
    }
    .section-title {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #ff5a5f;
        margin: 0 0 0.5rem 0;
    }
    .st-key-predict, .st-key-predict > div, .st-key-predict button { width: 100% !important; }
    .stButton, [data-testid="stButton"] { width: 100%; }
    .stButton > button, [data-testid="stButton"] > button, [data-testid="stBaseButton-primary"] {
        width: 100%;
        padding: 0.7rem 1rem;
        font-size: 1.05rem;
        font-weight: 600;
        border-radius: 10px;
    }
    .result {
        border-left: 6px solid #ff5a5f;
        background: rgba(255, 90, 95, 0.10);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0 0.8rem 0;
    }
    .result-label {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        opacity: 0.75;
    }
    .result-price {
        font-size: 3.4rem;
        font-weight: 800;
        line-height: 1.1;
        color: #ff5a5f;
    }
    .result-for { font-size: 0.98rem; margin-top: 0.2rem; }
    .result-note { font-size: 0.85rem; opacity: 0.75; margin-top: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    model = joblib.load("airbnb_price_model.joblib")
    with open("neighbourhood_lookup.json") as f:
        lookup = json.load(f)
    return model, lookup


model, lookup = load_assets()
prep = model.named_steps["prep"]
estimator_class = type(model.named_steps["model"]).__name__
algorithm = ALGORITHM_NAMES.get(estimator_class, estimator_class)

with st.sidebar:
    st.markdown("### About this model")
    st.write(
        f"Algorithm: **{algorithm}**, chosen as the best of 5 algorithms "
        "using 10-fold cross-validation on the full cleaned dataset."
    )
    st.markdown("### Data")
    st.write(
        "New York City Airbnb Open Data (2019), about 48,600 listings after "
        "cleaning. Prices above $1,000 per night were removed, so the model "
        "does not predict luxury listings."
    )
    st.caption("Machine Learning Fundamentals (AIC354), Assignment 3")

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">NYC Airbnb Price Predictor</div>
        <div class="hero-sub">Estimate the nightly price of a New York City listing
        from its location, room type and activity.</div>
        <span class="badge">{algorithm}</span>
        <span class="badge">10-fold cross-validation</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown('<div class="section-title">Location and room</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        borough = st.selectbox("Borough", sorted(lookup))
    with col2:
        hood = st.selectbox("Neighbourhood", sorted(lookup[borough]))
    room = st.radio(
        "Room type",
        ["Entire home/apt", "Private room", "Shared room"],
        horizontal=True,
    )

with st.container(border=True):
    st.markdown('<div class="section-title">Booking rules</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        min_nights = st.number_input("Minimum nights", 1, 365, 3)
    with col2:
        availability = st.slider("Days available per year", 0, 365, 180)

with st.container(border=True):
    st.markdown('<div class="section-title">Reviews and host</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        n_reviews = st.number_input("Number of reviews", 0, 1000, 10)
    with col2:
        rpm = st.number_input("Reviews per month", 0.0, 30.0, 1.0, step=0.1)
    with col3:
        host_listings = st.number_input("Host's total listings", 1, 400, 1)

if st.button("Predict price", type="primary", key="predict"):
    lat, lon = lookup[borough][hood]
    row = pd.DataFrame(
        [
            {
                "latitude": lat,
                "longitude": lon,
                "minimum_nights": min_nights,
                "number_of_reviews": n_reviews,
                "reviews_per_month": rpm,
                "calculated_host_listings_count": host_listings,
                "availability_365": availability,
                "neighbourhood_group": borough,
                "neighbourhood": hood,
                "room_type": room,
            }
        ]
    )
    price = max(float(model.predict(row)[0]), 0.0)

    st.session_state["result"] = {
        "price": price,
        "row": row,
        "summary": f"{room} in {hood}, {borough}",
    }

result = st.session_state.get("result")

if result:
    st.markdown(
        f"""
        <div class="result">
            <div class="result-label">Estimated price per night</div>
            <div class="result-price">${result['price']:,.0f}</div>
            <div class="result-for">{result['summary']}</div>
            <div class="result-note">Typical model error is about ${CV_RMSE} per night
            (10-fold cross-validation RMSE), so treat this as a rough estimate.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("See the encoded feature vector sent to the model"):
        row = result["row"]

        st.markdown("**1. Input as entered**")
        st.dataframe(row, hide_index=True)

        encoded = prep.transform(row)
        if hasattr(encoded, "toarray"):
            encoded = encoded.toarray()
        raw_names = list(prep.get_feature_names_out())
        encoded_df = pd.DataFrame(encoded, columns=[n.split("__", 1)[1] for n in raw_names])
        numeric_cols = [n.split("__", 1)[1] for n in raw_names if n.startswith("num__")]
        category_cols = [n.split("__", 1)[1] for n in raw_names if n.startswith("cat__")]

        st.markdown("**2. Numeric features**")
        st.dataframe(encoded_df[numeric_cols], hide_index=True)
        unchanged = np.allclose(
            encoded_df[numeric_cols].iloc[0].to_numpy(dtype=float),
            row[numeric_cols].iloc[0].to_numpy(dtype=float),
        )
        st.caption(
            "These values are passed to the model unchanged, because this algorithm "
            "does not need feature scaling."
            if unchanged
            else "These values are standardised (mean 0, standard deviation 1) "
            "before reaching the model, as this algorithm requires."
        )

        active = [c for c in category_cols if encoded_df[c].iloc[0] != 0]
        st.markdown("**3. Categorical features after one-hot encoding (active columns)**")
        st.dataframe(encoded_df[active].astype(int), hide_index=True)
        st.caption(
            f"The model receives {encoded_df.shape[1]} numbers in total: "
            f"{len(numeric_cols)} numeric columns and {len(category_cols)} one-hot "
            f"columns. Only {len(active)} of the one-hot columns are 1; the other "
            f"{len(category_cols) - len(active)} are 0 and are hidden above."
        )

        if st.checkbox("Show the full encoded vector"):
            st.dataframe(encoded_df, hide_index=True)
