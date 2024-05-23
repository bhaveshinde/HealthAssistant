import os
import joblib
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")


diabetes_model_path = os.path.join("saved_models", "model_diabetes.pkl")
hypertension_model_path = os.path.join("saved_models", "model_BloodPressure.pkl")
obesity_model_path = os.path.join("saved_models", "model_Obesity.pkl")
cholestrol_model_path = os.path.join("saved_models", "model_Cholestrol.pkl")
model = load_model('model_malaria_vgg19.h5')



# loading the saved models
diabetes_model = joblib.load(diabetes_model_path)
hypertension_model = joblib.load(hypertension_model_path)
Obesity_model = joblib.load(obesity_model_path)
cholestrol_model = joblib.load(cholestrol_model_path)

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Blood Pressure Prediction',
                            'Obesity Prediction',
                            'Cholestrol Prediction',
                            'Malaria Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart-pulse', 'person', 'heart'],
                           default_index=0)


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age = st.text_input('Age')

    with col2:
        FBS = st.text_input('FBS')

    with col3:
        RBS = st.text_input('RBS value')

    with col4:
        HbA1c = st.text_input('HbA1c value')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):
        if age.strip() == '' or FBS.strip() == '' or RBS.strip() == '' or HbA1c.strip() == '':
            st.warning("All fields are required!")
        else:
            user_input = [age, FBS, RBS, HbA1c]
            user_input = [float(x) for x in user_input]
            diab_prediction = diabetes_model.predict([user_input])

            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'


    st.success(diab_diagnosis)

# Blood Pressure Prediction Page
if selected == 'Blood Pressure Prediction':

    # page title
    st.title('Blood Pressure Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        person_age = st.text_input("Person's age")

    with col2:
        systolic = st.text_input('Systolic')

    with col3:
        diastolic = st.text_input('Diastolic')

    with col1:
        sodium = st.text_input('Sodium')

    with col2:
        triglyceride = st.text_input('Triglyceride')


    # code for Prediction
    bloodPressure = ''

    # creating a button for Prediction

    if st.button('Blood Pressure Prediction'):
        if person_age.strip() == '' or systolic.strip() == '' or diastolic.strip() == '' or sodium.strip() == '' or triglyceride.strip() == '':
            st.warning("All fields are required!")
        else:
            user_input = [person_age, systolic, diastolic, sodium, triglyceride]
            user_input = [float(x) for x in user_input]
            hypertension_prediction = hypertension_model.predict([user_input])

            if hypertension_prediction[0] == 0:
                bloodPressure = 'Blood Pressure level LOW'
            elif hypertension_prediction[0] == 1:
                bloodPressure = 'Blood Pressure level NORMAL'
            elif hypertension_prediction[0] == 2:
                bloodPressure = 'Blood Pressure level HIGH'

    st.success(bloodPressure)

# Obesity Prediction Page
if selected == "Obesity Prediction":

    # page title
    st.title("Obesity Prediction using ML")

    col1, col2, col3 = st.columns(3)

    with col1:
        patient_age = st.text_input("Patient's age")

    with col2:
        Gender = st.text_input('Gender')

    with col3:
        waist_circum = st.text_input('Waist circumference')

    with col1:
        skin_fold = st.text_input('Skin fold')

    with col2:
        BMI = st.text_input('BMI')


   # Initialize flag for input validation
    valid_input = True

    # Code for Prediction
    Obesity = ''

    # Creating a button for Prediction    
    if st.button("Obesity Test Result"):
        if patient_age.strip() == '' or waist_circum.strip() == '' or skin_fold.strip() == '' or BMI.strip() == '':
            st.warning("All fields are required!")
            valid_input = False
        else:
            if isinstance(Gender, str):
                Gender = Gender.strip().lower()
                if Gender in ['m', 'male']:
                    Gender = 1
                elif Gender in ['f', 'female']:
                    Gender = 0
                else:
                    st.warning("Invalid input for gender. Please enter 'm' or 'male' for male, or 'f' or 'female' for female.")
                    valid_input = False

            if valid_input:
                user_input = [patient_age, Gender, waist_circum, skin_fold, BMI]
                user_input = [float(x) for x in user_input]
                obesity_prediction = Obesity_model.predict([user_input])

                if obesity_prediction[0] == 1:
                    Obesity = "The person is Obese"
                else:
                    Obesity = "The person is not Obese"

    # Display final output only if input is valid
    if valid_input:
        st.success(Obesity)

#Cholestrol prediction
if selected == 'Cholestrol Prediction':

    # page title
    st.title('Cholestrol Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Patient_age = st.text_input("Person's age")

    with col2:
        HDL = st.text_input('HDL')

    with col3:
        LDL = st.text_input('LDL')

    with col1:
        TG = st.text_input('TG')

    with col2:
        totalCholestrol = st.text_input('Total Cholestrol')


    # code for Prediction
    cholestrol = ''

    # creating a button for Prediction

    if st.button('Cholestrol Prediction'):
        if Patient_age.strip() == '' or HDL.strip() == '' or LDL.strip() == '' or TG.strip() == '' or totalCholestrol.strip() == '':
            st.warning("All fields are required!")
        else:
            user_input = [Patient_age, HDL, LDL, TG, totalCholestrol]
            user_input = [float(x) for x in user_input]
            cholestrolPrediction = cholestrol_model.predict([user_input])

            if cholestrolPrediction[0] == 0:
                cholestrol = 'Cholestrol level LOW'
            elif cholestrolPrediction[0] == 1:
                cholestrol = 'Cholestrol level NORMAL'
            elif cholestrolPrediction[0] == 2:
                cholestrol = 'Cholestrol level HIGH'

    st.success(cholestrol)

if selected == 'Malaria Prediction':
    st.title('Malaria Classification')

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type="png")

    if uploaded_file is not None:
        # Display the uploaded image
        image_display = st.image(uploaded_file, caption='Uploaded Image', width=200)

        # Preprocess the uploaded image
        img = image.load_img(uploaded_file, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make prediction
        prediction = model.predict(img_array)
        class_label = 'Person is Malaria Infected' if prediction[0][0] > 0.5 else 'Person is Uninfected'

        # Display prediction result
        st.write(f'Prediction: {class_label}')

        