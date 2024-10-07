import numpy as np
import pickle
import streamlit as st
from PIL import Image
import base64

# Function to encode the image for CSS background use
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load the saved model
loaded_model = pickle.load(open('/Users/niharika/Downloads/miniproject files/heart_disease_model.sav', 'rb'))

# Create a function for Prediction
def heart_disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return 'The person has heart disease' if prediction[0] == 1 else 'The person does not have heart disease'

def main():
    # Set title
    st.title('Heart Disease Prediction Web App')

    # CSS for background image and styling
    img_base64 = get_base64_encoded_image('Users/niharika/Downloads/miniproject files/heartpic.png')
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{img_base64}");
            background-size: cover;
            background-position: center;
        }}
        .input-container {{
            background: rgba(255, 255, 255, 0.7);
            padding: 15px;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Input fields in a styled container
    with st.container():
        st.markdown("<div class='input-container'>", unsafe_allow_html=True)
        age = st.number_input('Age')
        sex = st.selectbox('Sex (1 = Male, 0 = Female)', [1, 0])
        cp = st.number_input('Chest Pain Type (0-3)')
        trestbps = st.number_input('Resting Blood Pressure')
        chol = st.number_input('Serum Cholesterol in mg/dl')
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)', [1, 0])
        restecg = st.number_input('Resting ECG results (0, 1, or 2)')
        thalach = st.number_input('Maximum Heart Rate Achieved')
        exang = st.selectbox('Exercise-Induced Angina (1 = Yes, 0 = No)', [1, 0])
        oldpeak = st.number_input('ST Depression Induced by Exercise')
        slope = st.number_input('Slope of the Peak Exercise ST Segment (0-2)')
        ca = st.number_input('Number of Major Vessels (0-4)')
        thal = st.number_input('Thalassemia (0-3)')
        
        if st.button('Heart Disease Test Result'):
            diagnosis = heart_disease_prediction([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
            st.success(diagnosis)
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
