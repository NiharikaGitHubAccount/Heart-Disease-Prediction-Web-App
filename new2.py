import numpy as np
import pickle
import streamlit as st
import base64

# Function to encode the image for CSS background use
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load the saved model
loaded_model = pickle.load(open('heart_disease_model.sav', 'rb'))

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
    img_base64 = get_base64_encoded_image('heartpic.png')  # Use your JPEG file here
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
        .footer {{
            text-align: center;
            margin-top: 20px;
            font-size: 14px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Input fields in a styled container
    with st.container():
        st.markdown("<div class='input-container'>", unsafe_allow_html=True)
        
        # Using dropdowns to avoid floating points and +- controls
        age = st.selectbox('Age', options=list(range(1, 201)))
        sex = st.selectbox('Sex', options=[(1, "Male"), (0, "Female")], format_func=lambda x: x[1])[0]
        cp = st.selectbox('Chest Pain Type', options=[(0, "Typical Angina"), (1, "Atypical Angina"), (2, "Non-anginal Pain"), (3, "Asymptomatic")], format_func=lambda x: x[1])[0]
        trestbps = st.selectbox('Resting Blood Pressure (mm Hg)', options=list(range(90, 201)))
        chol = st.selectbox('Serum Cholesterol (mg/dl)', options=list(range(100, 601)))
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[(1, "True"), (0, "False")], format_func=lambda x: x[1])[0]
        restecg = st.selectbox('Resting ECG Results', options=[(0, "Normal"), (1, "ST-T Wave Abnormality"), (2, "Left Ventricular Hypertrophy")], format_func=lambda x: x[1])[0]
        thalach = st.selectbox('Maximum Heart Rate Achieved', options=list(range(60, 211)))
        exang = st.selectbox('Exercise-Induced Angina', options=[(1, "Yes"), (0, "No")], format_func=lambda x: x[1])[0]
        oldpeak = st.selectbox('ST Depression Induced by Exercise', options=[round(x * 0.1, 1) for x in range(0, 101)])  # Ranges from 0.0 to 10.0
        slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[(0, "Upsloping"), (1, "Flat"), (2, "Downsloping")], format_func=lambda x: x[1])[0]
        ca = st.selectbox('Number of Major Vessels (0-4)', options=list(range(5)))
        thal = st.selectbox('Thalassemia', options=[(0, "Normal"), (1, "Fixed Defect"), (2, "Reversible Defect")], format_func=lambda x: x[1])[0]
        
        if st.button('Heart Disease Test Result'):
            diagnosis = heart_disease_prediction([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
            st.success(diagnosis)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Footer with external link and contact info
    st.markdown(
        """
        <div class='footer'>
            <p>For more information on heart disease, visit 
            <a href="https://www.heart.org/en/health-topics/heart-attack">Heart Association</a></p>
            <p>Contact: ryanwand80@gmail.com | Phone: 8217610060</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
