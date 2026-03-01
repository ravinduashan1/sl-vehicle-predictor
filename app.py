# Transform text to numbers using your encoders
    input_data['Brand'] = label_encoders['Brand'].transform(input_data['Brand'])
    input_data['Model'] = label_encoders['Model'].transform(input_data['Model'])
    
    # Run the prediction
    prediction = model.predict(input_data)
    
    # Display the result
    st.metric("Estimated Price", f"Rs. {prediction[0]:,.2f}")
