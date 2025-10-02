import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Load best model info
# ---------------------------
best_model_info = joblib.load("models/best_model_info.pkl")
model = joblib.load(best_model_info["file_path"])

st.set_page_config(page_title="House Price Prediction", page_icon="🏡", layout="centered")

# ---------------------------
# Enhanced CSS for Modern Dark Theme
# ---------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 100%);
        padding: 40px;
        border-radius: 20px;
        border: 1px solid #2a2a2a;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
    }

    .stApp {
        background: linear-gradient(135deg, #000000 0%, #1a1a2e 50%, #16213e 100%);
    }

    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        text-align: center;
        margin-bottom: 10px;
        padding: 10px;
    }

    h3 {
        color: #e0e0e0 !important;
        font-weight: 500;
        text-align: center;
        margin-bottom: 30px;
        opacity: 0.9;
    }

    .stSelectbox, .stNumberInput {
        background: rgba(30, 30, 30, 0.7) !important;
        border-radius: 12px !important;
        border: 1px solid #333 !important;
        padding: 8px;
        margin-bottom: 15px;
    }

    .stSelectbox div, .stNumberInput div {
        color: white !important;
    }

    .stSelectbox label, .stNumberInput label {
        color: #b0b0b0 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        margin-bottom: 5px !important;
    }

    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 16px;
        font-weight: 600;
        border-radius: 12px;
        padding: 14px 32px;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    .success-message {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 15px rgba(0, 176, 155, 0.2);
    }

    .model-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(255, 107, 107, 0.3);
    }

    .input-section {
        background: rgba(255, 255, 255, 0.05);
        padding: 25px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }

    .section-title {
        color: #667eea;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Page Header
# ---------------------------
st.markdown("<h1>🏡 House Price Prediction System</h1>", unsafe_allow_html=True)
#st.markdown(f'<div class="model-badge">Best Model: {best_model_info["name"]}</div>', unsafe_allow_html=True)
st.markdown("<h3>Enter property details below to get the estimated price</h3>", unsafe_allow_html=True)

# ---------------------------
# Load dataset for dropdowns
# ---------------------------
df = pd.read_csv("house_prices_processed.csv")
location_options = sorted(df["location"].dropna().unique().tolist())
overlooking_options = sorted(df["overlooking"].dropna().unique().tolist())

# ---------------------------
# User Inputs in Organized Sections
# ---------------------------
with st.container():
    st.markdown('<div class="input-section">', unsafe_allow_html=True)

    # Property Basics
    st.markdown('<div class="section-title">📍 Property Basics</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        location = st.selectbox("Location", location_options)
        carpet_area = st.number_input("Carpet Area (sq ft)", min_value=100.0, max_value=10000.0, step=50.0,
                                      value=1000.0)
        status = st.selectbox("Status", ["Ready to Move", "Under Construction"])
        transaction = st.selectbox("Transaction", ["Resale", "New Property"])

    with col2:
        furnishing = st.selectbox("Furnishing", ["Furnished", "Semi-Furnished", "Unfurnished"])
        facing = st.selectbox("Facing Direction", ["East", "West", "North", "South"])
        overlooking = st.selectbox("Overlooking View", overlooking_options)

    st.markdown('</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="input-section">', unsafe_allow_html=True)

    # Property Features
    st.markdown('<div class="section-title">🏠 Property Features</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        bathroom = st.number_input("Bathrooms", min_value=1, max_value=10, step=1, value=2)
        balcony = st.number_input("Balconies", min_value=0, max_value=5, step=1, value=1)

    with col2:
        car_parking = st.number_input("Car Parking", min_value=0, max_value=5, step=1, value=1)
        ownership = st.selectbox("Ownership Type",
                                 ["Freehold", "Leasehold", "Co-operative Society", "Power of Attorney"])

    with col3:
        current_floor = st.number_input("Current Floor", min_value=0, max_value=100, step=1, value=2)
        total_floors = st.number_input("Total Floors", min_value=1, max_value=200, step=1, value=10)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Prepare Input Data
# ---------------------------
input_data = pd.DataFrame({
    "location": [location],
    "Carpet Area": [carpet_area],
    "Status": [status],
    "Transaction": [transaction],
    "Furnishing": [furnishing],
    "facing": [facing],
    "overlooking": [overlooking],
    "Bathroom": [bathroom],
    "Balcony": [balcony],
    "Car Parking": [car_parking],
    "Ownership": [ownership],
    "current_floor": [current_floor],
    "total_floors": [total_floors]
})

# ---------------------------
# Prediction
# ---------------------------
if st.button("🔮 Predict House Price"):
    try:
        prediction = model.predict(input_data)[0]
        st.markdown(
            f"""
            <div class="success-message">
                <h2 style="color: white; margin: 0; font-size: 1.8rem;">💰 Estimated Price</h2>
                <p style="color: white; font-size: 2.2rem; font-weight: 700; margin: 10px 0;">₹ {prediction:,.2f}</p>
                <p style="color: rgba(255,255,255,0.8); margin: 0;">Based on comprehensive property analysis</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error("⚠️ Error: Make sure categorical inputs are encoded the same way as training data.")
        st.exception(e)