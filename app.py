import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and the label encoder for roles
model = joblib.load("fine_tuned_career_model.pkl")
role_encoder = joblib.load("role.pkl")

# List of features to collect from user (assuming your model expects these features)
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
    if feature == 'CGPA':
        user_inputs[feature] = st.number_input(f"{feature} (0 to 10)", min_value=0.0, max_value=10.0, step=0.1)
    else:
        user_inputs[feature] = st.selectbox(f"{feature} (yes/no)", ["yes", "no"])

# Convert the inputs into a pandas DataFrame
input_data = pd.DataFrame([user_inputs])

# Encode "yes" as 1 and "no" as 0 for non-numeric features
input_data_encoded = input_data.copy()
for column in input_data.select_dtypes(include=['object']).columns:
    input_data_encoded[column] = input_data[column].apply(lambda x: 1 if x == "yes" else 0)

# Predict the role when the button is pressed
if st.button("Predict Role"):
    # Make prediction using the model
    prediction = model.predict(input_data_encoded)

    # Decode the predicted role using the role encoder
    predicted_role = role_encoder.inverse_transform(prediction)

    # Display the predicted role
    st.subheader(f"Predicted Role: {predicted_role[0]}")
