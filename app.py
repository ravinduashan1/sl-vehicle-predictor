import streamlit as st
import pandas as pd
import joblib
import urllib.request
import os

# 1. Setup Page
st.set_page_config(page_title="SL Vehicle Predictor", page_icon="🚗")
st.title("🇱🇰 Sri Lanka Vehicle Price Predictor")

# 2. Function to load model from Hugging Face
@st.cache_resource
def load_model():
    model_path = 'vehicle_price_model.pkl'
    url = 'https://huggingface.co/Ravindu92/sl-vehicle-model/resolve/main/vehicle_price_model.pkl'
    
    if not os.path.exists(model_path):
        with st.spinner('Downloading AI Model... Please wait.'):
            urllib.request.urlretrieve(url, model_path)
            
    model = joblib.load(model_path)
    label_encoders = joblib.load('label_encoders.pkl')
    return model, label_encoders

# 3. Load the data
try:
    model, label_encoders = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 4. Create User Inputs
st.subheader("Enter Vehicle Details")
brand = st.selectbox("Brand", label_encoders['Brand'].classes_)
model_name = st.selectbox("Model", label_encoders['Model'].classes_)
year = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
mileage = st.number_input("Mileage (km)", min_value=0, value=50000)

# 5. Prediction Logic (Pay attention to the spaces below!)
if st.button("Predict Price"):
    # This data must be indented to stay "inside" the button click
    input_data = pd.DataFrame([[brand, model_name, year, mileage]], 
                              columns=['Brand', 'Model', 'Year', 'Mileage'])
    
    # Transform text to numbers
    input_data['Brand'] = label_encoders['Brand'].transform(input_data['Brand'])
    input_data['Model'] = label_encoders['Model'].transform(input_data['Model'])
    
    # Run the model
    prediction = model.predict(input_data)
    
    # Show the result
    st.metric("Estimated Price", f"Rs. {prediction[0]:,.2f}")

