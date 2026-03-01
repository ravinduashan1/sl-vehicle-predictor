import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np

# Load the saved model and encoders
model = joblib.load('vehicle_price_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Set up the website title and text
st.title("🇱🇰 Sri Lanka Vehicle Price Predictor")
st.write("Predict future vehicle prices based on local market trends and exchange rates.")

# Create input fields for the user
col1, col2 = st.columns(2)
with col1:
    # We use the keys from your label encoders to populate the dropdowns
    brand = st.selectbox("Vehicle Brand", list(label_encoders['Brand'].classes_))
    year = st.number_input("Manufacture Year", min_value=1980, max_value=2026, value=2015)
    months_ahead = st.slider("Prediction Period (Months ahead)", 1, 12, 2)
with col2:
    model_name = st.selectbox("Vehicle Model", list(label_encoders['Model'].classes_))
    mileage = st.number_input("Mileage (km)", min_value=0, value=50000)
    # Adding exchange rate makes the app interactive and realistic!
    exchange_rate = st.number_input("Expected USD/LKR Exchange Rate", min_value=250.0, max_value=400.0, value=300.0)

# Create a button to run the prediction
if st.button("Predict Price"):
    try:
        # 1. Convert words to numbers for the AI
        brand_encoded = label_encoders['Brand'].transform([brand])[0]
        model_encoded = label_encoders['Model'].transform([model_name])[0]
        
        # 2. Generate future predictions for the graph
        future_prices = []
        months = ["Today"]
        
        # Predict today's price
        base_input = pd.DataFrame({'Brand': [brand_encoded], 'Model': [model_encoded], 'Year': [year], 
                                   'Mileage (km)': [mileage], 'Days_Since_Start': [0], 'Exchange_Rate': [exchange_rate]})
        today_price = model.predict(base_input)[0]
        future_prices.append(today_price)
        
        # Predict future prices month by month
        for i in range(1, months_ahead + 1):
            months.append(f"Month {i}")
            # Add 30 days per month to the 'Days_Since_Start' feature
            future_input = pd.DataFrame({'Brand': [brand_encoded], 'Model': [model_encoded], 'Year': [year], 
                                         'Mileage (km)': [mileage + (i*1000)], # Assume driving 1000km/month
                                         'Days_Since_Start': [i * 30], 'Exchange_Rate': [exchange_rate]})
            future_prices.append(model.predict(future_input)[0])
            
        # 3. Display the final result
        final_price = future_prices[-1]
        st.success(f"### Estimated Price in {months_ahead} Months: Rs. {final_price:,.2f}")
        
        # 4. Draw the Graph
        st.subheader("Predicted Price Trend")
        chart_data = pd.DataFrame({"Price (Rs)": future_prices}, index=months)
        st.line_chart(chart_data)
        
    except ValueError:
        st.error("Error: This specific Brand and Model combination might not be in the training data.")