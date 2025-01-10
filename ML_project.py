import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the trained model
model = joblib.load("best_model1.pkl")

st.title("Heart Failure Prediction App")

# Define input fields for features
age = st.slider('Age', min_value=20, max_value=80, value=50, step=1)
sex = st.selectbox('Sex (1 = Male, 0 = Female)', [1, 0])
chestPainType = st.selectbox('Chest Pain Type (0=ASY, 1=ATA, 2=NAP, 3=TA)', [0, 1, 2, 3])
restingBP = st.slider('Resting Blood Pressure', min_value=90, max_value=200, value=120, step=1)
cholesterol = st.slider('Cholesterol', min_value=0, max_value=700, value=200)
fastingBS = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)', [1, 0])
restingECG = st.selectbox('Resting ECG (0=LVH, 1=Normal, 2=ST)', [0, 1, 2])
maxHR = st.slider('Maximum Heart Rate Achieved', min_value=50, max_value=210, value=140, step=1)
exerciseAngina = st.selectbox('Exercise Induced Angina (1 = Yes, 0 = No)', [1, 0])
oldpeak = st.slider('ST Depression', min_value=-3.0, max_value=9.0, value=1.0, step=0.1)
st_Slope = st.selectbox('Slope of ST Segment (0=Down, 1=FLAT, 2=UP)', [0, 1, 2])

#'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
# Create a button for making predictions
if st.button("Predict"):
    # Process input values
    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Sex": [sex],
            "ChestPainType": [chestPainType],
            "RestingBP": [restingBP],
            "Cholesterol": [cholesterol],
            "FastingBS": [fastingBS],
            "RestingECG": [restingECG],
            "MaxHR": [maxHR],
            "ExerciseAngina": [exerciseAngina],
            "Oldpeak": [oldpeak],
            "ST_Slope": [st_Slope]
          
        }
    )

    # Scale input data using the same scaler used during training
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    # Make a prediction using the trained model
    prediction = model.predict(input_data_scaled)

    # Display the prediction
    if prediction[0] == 1:
        st.success("The patient has heart disease.")
    else:
        st.success("The patient does not have heart disease.")