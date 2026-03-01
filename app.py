import streamlit as st
import pandas as pd
import joblib
import requests # Make sure 'requests' is in your requirements.txt
import os

@st.cache_resource
def load_large_model():
    model_path = 'vehicle_price_model.pkl'
    file_id = '1wBxYwhRWeLdkS3yRsnCRfCzht3LY9pRR'
    
    if not os.path.exists(model_path):
        with st.spinner('Downloading AI Model (133MB)... Please wait, this takes a moment.'):
            # Professional method to bypass Google Drive's "Large File" warning
            def get_confirm_token(response):
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        return value
                return None

            URL = "https://docs.google.com/uc?export=download"
            session = requests.Session()
            response = session.get(URL, params={'id': file_id}, stream=True)
            token = get_confirm_token(response)

            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(URL, params=params, stream=True)

            # Write the file to the server
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk: 
                        f.write(chunk)
    
    try:
        model = joblib.load(model_path)
        label_encoders = joblib.load('label_encoders.pkl')
        return model, label_encoders
    except Exception as e:
        # If it fails, delete the corrupted file so it retries next time
        if os.path.exists(model_path):
            os.remove(model_path)
        st.error(f"Model loading failed: {e}. The file might have been corrupted during download.")
        return None, None

model, label_encoders = load_large_model()

