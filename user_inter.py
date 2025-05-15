import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset
lung_data = pd.read_csv(r"C:\Users\Dev Bhatt\OneDrive\Desktop\surveylungcancer.csv")

# Preprocessing
lung_data.GENDER = lung_data.GENDER.map({"M": 1, "F": 2})
lung_data.LUNG_CANCER = lung_data.LUNG_CANCER.map({"YES": 1, "NO": 2})

# Splitting the dataset
x = lung_data.iloc[:, 0:-1]
y = lung_data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# Model training
model = LogisticRegression()
model.fit(x_train, y_train)

# Streamlit application
st.title("Lung Cancer Detection System")
st.sidebar.header("Input Features")
st.write("This application predicts the likelihood of lung cancer based on user inputs.")

# User input
gender = st.sidebar.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
smoking = st.sidebar.radio("Smoking Level", [1, 2], format_func=lambda x: "Low" if x == 1 else "High")
yellow_fingers = st.sidebar.radio("Yellow Fingers", [1, 2], format_func=lambda x: "Low" if x == 1 else "High")
anxiety = st.sidebar.radio("Anxiety", [1, 2], format_func=lambda x: "Low" if x == 1 else "High")
peer_pressure = st.sidebar.radio("Peer Pressure", [1, 2], format_func=lambda x: "Low" if x == 1 else "High")
chronic_disease = st.sidebar.radio("Chronic Disease", [1, 2], format_func=lambda x: "Low" if x == 1 else "High")
fatigue = st.sidebar.radio("Fatigue", [1, 2], format_func=lambda x: "Low" if x == 1 else "High")
allergy = st.sidebar.radio("Allergy", [1, 2], format_func=lambda x: "Low" if x == 1 else "High")
wheezing = st.sidebar.radio("Wheezing", [1, 2], format_func=lambda x: "Low" if x == 1 else "High")
alcohol_consuming = st.sidebar.radio("Alcohol Consuming", [1, 2], format_func=lambda x: "Low" if x == 1 else "High")
coughing = st.sidebar.radio("Coughing", [1, 2], format_func=lambda x: "Low" if x == 1 else "High")
shortness_of_breath = st.sidebar.radio("Shortness Of Breath", [1, 2], format_func=lambda x: "Low" if x == 1 else "High")
swallowing_difficulty = st.sidebar.radio("Swallowing Difficulty", [1, 2], format_func=lambda x: "Low" if x == 1 else "High")
chest_pain = st.sidebar.radio("Chest Pain", [1, 2], format_func=lambda x: "Low" if x == 1 else "High")

# Predictive system
input_data = np.array([gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease,
                       fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath,
                       swallowing_difficulty, chest_pain]).reshape(1, -1)

prediction = model.predict(input_data)

if st.button("Predict"):
    if prediction[0] == 1:
        st.error("Lung Cancer Detected. Consult a doctor immediately.")
    else:
        st.success("No Symptoms of Lung Cancer detected.")

# Display model performance metrics
if st.checkbox("Show Model Metrics"):
    y_pred = model.predict(x_test)
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred))
    st.write("Recall:", recall_score(y_test, y_pred))
    st.write("F1 Score:", f1_score(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    st.pyplot(plt)