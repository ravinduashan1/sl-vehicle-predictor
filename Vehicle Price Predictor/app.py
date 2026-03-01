import streamlit as st
import pandas as pd
import joblib
import datetime

# Load the saved model and encoders
# We use st.cache_resource so the app doesn't reload the model every time you click a button
@st.cache_resource
def load_data():
    model = joblib.load('vehicle_price_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    return model, label_encoders

model, label_encoders = load_data()
