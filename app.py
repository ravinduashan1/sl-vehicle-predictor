import streamlit as st
import pandas as pd
import joblib
import urllib.request
import os

# Function to download and load the large model
@st.cache_resource
def load_large_model():
    model_path = 'vehicle_price_model.pkl'
    # The direct link we just created
    url = 'https://drive.google.com/uc?export=download&id=1wBxYwhRWeLdkS3yRsnCRfCzht3LY9pRR'
    
    if not os.path.exists(model_path):
        try:
            with st.spinner('Downloading AI Model from Google Drive (133MB)... Please wait.'):
                urllib.request.urlretrieve(url, model_path)
        except Exception as e:
            st.error(f"Download failed. Error: {e}")
            return None, None
            
    model = joblib.load(model_path)
    label_encoders = joblib.load('label_encoders.pkl')
    return model, label_encoders

# Load everything
model, label_encoders = load_large_model()

if model is None:
    st.stop() # Stop the app if the model didn't load
