import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and encoders
model = joblib.load("fine_tuned_career_model.pkl")
role_encoder = joblib.load("role_encoder.pkl")

# List of features to collect from user
features = [
    "CGPA", "Webdev", "Data_analysis", "Reading_Writing", "Tech_person", "Non_tech_society",
    "Coding_skills", "Mobile_apps", "Communication", "Specialization_security", "Large_databases",
    "Data_science", "English_proficiency", "Event_management", "Technical_blogs", "Marketing_interest",
    "ML_expertise", "Connections", "Live_projects"
]

# Title of the app
st.title("Career Role Prediction")

# User Input Section
st.header("Provide Your Details")

# Collect user inputs for all the features
user_inputs = {}
for feature in features:
    if feature in ['CGPA']:
        user_inputs[feature] = st.number_input(f"{feature} (0 to 10)", min_value=0.0, max_value=10.0, step=0.1)
    else:
        user_inputs[feature] = st.selectbox(f"{feature} (yes/no)", ["yes", "no"])

# Create a DataFrame from the inputs
input_data = pd.DataFrame([user_inputs])

# Encode the features using the saved label encoders
encoded_data = input_data.copy()
for column in input_data.select_dtypes(include=['object']).columns:
    # Use label encoder for each feature column
    le = joblib.load(f"{column}_encoder.pkl")  # Load encoder for the feature
    encoded_data[column] = le.transform(input_data[column])

# Predict the role when the button is pressed
if st.button("Predict Role"):
    # Standardize the input data
    scaler = joblib.load("scaler.pkl")  # Load scaler for feature scaling
    scaled_data = scaler.transform(encoded_data)

    # Make prediction
    prediction = model.predict(scaled_data)

    # Decode the predicted role
    predicted_role = role_encoder.inverse_transform(prediction)
    st.subheader(f"Predicted Role: {predicted_role[0]}")
