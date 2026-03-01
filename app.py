import streamlit as st
import pandas as pd
import joblib
import urllib.request
import os

@st.cache_resource
def load_model():
    model_path = 'vehicle_price_model.pkl'
    # Paste your NEW Hugging Face direct link here
    url = 'https://huggingface.co/Ravindu92/sl-vehicle-model/resolve/main/vehicle_price_model.pkl'
    
    if not os.path.exists(model_path):
        with st.spinner('Downloading Model from Hugging Face...'):
            urllib.request.urlretrieve(url, model_path)
            
    model = joblib.load(model_path)
    label_encoders = joblib.load('label_encoders.pkl')
    return model, label_encoders

model, label_encoders = load_model()
