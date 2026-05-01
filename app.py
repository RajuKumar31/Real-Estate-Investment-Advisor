import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page Config ──
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide"
)

# ── Load Models ──
@st.cache_resource
def load_models():
    with open("models/rf_classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("models/xgb_regressor.pkl", "rb") as f:
        reg = pickle.load(f)
    with open("models/feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return clf, reg, feature_cols

clf, reg, feature_cols = load_models()

# ── Load Reference Data ──
@st.cache_data
def load_data():
    return pd.read_csv("india_housing_prices.csv")

df_ref = load_data()

# ── Title ──
st.title("🏠 Real Estate Investment Advisor")
st.markdown("Enter property details below to get an **investment recommendation** and **5-year price forecast**.")
st.divider()

# ── Input Form ──
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📍 Location")
    state = st.selectbox("State", sorted(df_ref['State'].unique()))
    city = st.selectbox("City", sorted(df_ref['City'].unique()))
    locality_num = st.slider("Locality (ID)", 1, 500, 100)

with col2:
    st.subheader("🏗️ Property Details")
    property_type = st.selectbox("Property Type", ["Apartment", "Independent House", "Villa"])
    bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])
    size = st.number_input("Size (SqFt)", min_value=500, max_value=5000, value=1500, step=100)
    price = st.number_input("Price (Lakhs)", min_value=10.0, max_value=500.0, value=150.0, step=5.0)
    year_built = st.slider("Year Built", 1990, 2024, 2010)
    floor_no = st.number_input("Floor Number", min_value=0, max_value=50, value=3)
    total_floors = st.number_input("Total Floors", min_value=1, max_value=50, value=10)

with col3:
    st.subheader("✨ Amenities & Features")
    furnished = st.selectbox("Furnished Status", ["Unfurnished", "Semi-furnished", "Furnished"])
    parking = st.selectbox("Parking Space", ["No", "Yes"])
    security = st.selectbox("Security", ["No", "Yes"])
    facing = st.selectbox("Facing", ["North", "South", "East", "West"])
    owner_type = st.selectbox("Owner Type", ["Owner", "Builder", "Broker"])
    availability = st.selectbox("Availability", ["Ready_to_Move", "Under_Construction"])
    nearby_schools = st.slider("Nearby Schools", 1, 10, 5)
    nearby_hospitals = st.slider("Nearby Hospitals", 1, 10, 5)
    amenity_count = st.slider("Number of Amenities", 1, 5, 3)
    transport = st.selectbox("Public Transport", ["Low", "Medium", "High"])

st.divider()

# ── Predict Button ──
if st.button("🔍 Analyse Investment", use_container_width=True):

    # ── Encode Inputs ──
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    # Map categorical inputs
    furnished_map = {"Unfurnished": 0, "Semi-furnished": 1, "Furnished": 2}
    availability_map = {"Under_Construction": 0, "Ready_to_Move": 1}
    transport_map = {"Low": 1, "Medium": 2, "High": 3}
    parking_map = {"No": 0, "Yes": 1}
    security_map = {"No": 0, "Yes": 1}
    property_map = {"Apartment": 0, "Independent House": 1, "Villa": 2}
    facing_map = {"East": 0, "North": 1, "South": 2, "West": 3}
    owner_map = {"Broker": 0, "Builder": 1, "Owner": 2}

    # Encode state and city using reference data
    state_encoded = sorted(df_ref['State'].unique()).index(state)
    city_encoded = sorted(df_ref['City'].unique()).index(city)

    # Derived features
    age_of_property = 2024 - year_built
    price_per_sqft = round(price / size, 4)
    infrastructure_score = nearby_schools + nearby_hospitals
    floor_ratio = round(min(floor_no, total_floors) / total_floors, 4)

    if price <= 150:
        price_segment = 0
    elif price <= 350:
        price_segment = 1
    else:
        price_segment = 2

    if age_of_property <= 5:
        age_group = 0
    elif age_of_property <= 15:
        age_group = 1
    elif age_of_property <= 25:
        age_group = 2
    else:
        age_group = 3

    # Build input row
    input_data = {
        'State': state_encoded,
        'City': city_encoded,
        'Locality': locality_num,
        'Property_Type': property_map[property_type],
        'BHK': bhk,
        'Size_in_SqFt': size,
        'Price_in_Lakhs': price,
        'Price_per_SqFt': price_per_sqft,
        'Furnished_Status': furnished_map[furnished],
        'Floor_No': min(floor_no, total_floors),
        'Total_Floors': total_floors,
        'Age_of_Property': age_of_property,
        'Nearby_Schools': nearby_schools,
        'Nearby_Hospitals': nearby_hospitals,
        'Parking_Space': parking_map[parking],
        'Security': security_map[security],
        'Facing': facing_map[facing],
        'Owner_Type': owner_map[owner_type],
        'Availability_Status': availability_map[availability],
        'Amenity_Count': amenity_count,
        'Infrastructure_Score': infrastructure_score,
        'Floor_Ratio': floor_ratio,
        'Price_Segment': price_segment,
        'Transport_Score': transport_map[transport],
        'Age_Group': age_group
    }

    input_df = pd.DataFrame([input_data])[feature_cols]

    # ── Predictions ──
    investment_pred = clf.predict(input_df)[0]
    investment_prob = clf.predict_proba(input_df)[0][1]
    future_price = reg.predict(input_df)[0]

    # ── Results ──
    st.subheader("📊 Investment Analysis Results")
    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        if investment_pred == 1:
            st.success("✅ GOOD INVESTMENT")
        else:
            st.error("❌ NOT RECOMMENDED")
        st.metric("Investment Score", f"{investment_prob*100:.1f}%")

    with res_col2:
        st.metric("Current Price", f"₹{price:.1f} Lakhs")
        st.metric("Predicted Price (5 Years)", f"₹{future_price:.1f} Lakhs")

    with res_col3:
        gain = future_price - price
        gain_pct = (gain / price) * 100
        st.metric("Expected Gain", f"₹{gain:.1f} Lakhs", f"{gain_pct:.1f}%")
        st.metric("Price per SqFt", f"₹{price_per_sqft:.4f} Lakhs")

    # ── Summary ──
    st.divider()
    st.markdown(f"""
    ### 📝 Summary
    - **Property:** {bhk}BHK {property_type} in {city}, {state}
    - **Size:** {size:,} SqFt | **Age:** {age_of_property} years
    - **Investment Verdict:** {'✅ Good Investment' if investment_pred == 1 else '❌ Not Recommended'}
    - **Confidence:** {investment_prob*100:.1f}%
    - **Current Price:** ₹{price:.1f} Lakhs
    - **5-Year Forecast:** ₹{future_price:.1f} Lakhs (+₹{gain:.1f}L / +{gain_pct:.1f}%)
    """)