import streamlit as st
import pandas as pd
import joblib
import urllib.request
import os

# A function to download the big file from a link if it doesn't exist yet
@st.cache_resource
def load_large_model():
    model_path = 'vehicle_price_model.pkl'
    if not os.path.exists(model_path):
        # REPLACE THIS URL with your direct download link from Google Drive/Dropbox
        url = 'https://drive.google.com/file/d/1wBxYwhRWeLdkS3yRsnCRfCzht3LY9pRR/view?usp=drive_link' 
        with st.spinner('Downloading AI Model (133MB)... this may take a minute.'):
            urllib.request.urlretrieve(url, model_path)
            
    model = joblib.load(model_path)
    label_encoders = joblib.load('label_encoders.pkl')
    return model, label_encoders

model, label_encoders = load_large_model()