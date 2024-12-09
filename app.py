import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model
model_path = "fine_tuned_career_model.pkl"
model = joblib.load(model_path)

# Load label encoders (for decoding the target predictions)
label_encoders = {
    "Role": LabelEncoder()
}

# Title
st.title("Career Prediction Model")

# User Input
st.header("Provide Your Details")

# Input fields
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)
webdev = st.selectbox("Did you do web development during college?", ["yes", "no"])
data_analysis = st.selectbox("Are you good at data analysis?", ["yes", "no"])
reading_writing = st.selectbox("Reading and Writing Skills", ["poor", "medium", "excellent"])
tech_person = st.selectbox("Are you a tech person?", ["yes", "no"])
non_tech_society = st.selectbox("Were you in a non-tech society?", ["yes", "no"])
coding_skills = st.selectbox("Are you good at coding?", ["yes", "no"])
mobile_apps = st.selectbox("Have you developed mobile apps?", ["yes", "no"])
communication = st.selectbox("Are you good at communication?", ["yes", "no"])
specialization_security = st.selectbox("Do you have specialization in security?", ["yes", "no"])
large_databases = st.selectbox("Have you handled large databases?", ["yes", "no"])
data_science = st.selectbox("Do you have knowledge of statistics and data science?", ["yes", "no"])
english_proficiency = st.selectbox("Are you proficient in English?", ["yes", "no"])
event_management = st.selectbox("Have you managed some event?", ["yes", "no"])
technical_blogs = st.selectbox("Do you write technical blogs?", ["yes", "no"])
marketing_interest = st.selectbox("Are you into marketing?", ["yes", "no"])
ml_expertise = st.selectbox("Are you a machine learning expert?", ["yes", "no"])
connections = st.selectbox("Do you have a lot of connections?", ["yes", "no"])
live_projects = st.selectbox("Have you built live projects?", ["yes", "no"])

# Collect inputs
features = pd.DataFrame(
    {
        "CGPA": [cgpa],
        "Did you do webdev during college time ?": [webdev],
        "Are you good at Data analysis ?": [data_analysis],
        "reading and writing skills": [reading_writing],
        "Are you a tech person ?": [tech_person],
        "Were you in a non tech society ?": [non_tech_society],
        "Are you good at coding ?": [coding_skills],
        "Have you developed mobile apps ?": [mobile_apps],
        "Are you good at communication ?": [communication],
        "Do you have specialization in security": [specialization_security],
        "Have you ever handled large databases ?": [large_databases],
        "Do you have knowlege of statistics and data science?": [data_science],
        "Are you proficient in English ?": [english_proficiency],
        "Have you ever managed some event?": [event_management],
        "Do you write technical blogs ?": [technical_blogs],
        "Are you into marketing ?": [marketing_interest],
        "Are you a ML expert ?": [ml_expertise],
        "Do you have a lot of connections ?": [connections],
        "Have you ever built live project ?": [live_projects],
    }
)

# Encode categorical features
for column in features.select_dtypes(include=["object"]).columns:
    le = label_encoders.get(column, LabelEncoder())
    features[column] = le.fit_transform(features[column])

# Standardize numerical features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Predict
if st.button("Predict Role"):
    prediction = model.predict(features)  # Encoded prediction
    # Reverse the encoding to get the original role
    predicted_role = label_encoders['Role'].inverse_transform(prediction)
    st.subheader(f"Predicted Role: {predicted_role[0]}")
