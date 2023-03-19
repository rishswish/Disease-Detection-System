import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pickle
from streamlit_option_menu import option_menu


parkinson_in = open('parkinson_classifier.pkl', 'rb')
parkinson_model = pickle.load(parkinson_in)

heart_in = open('heart_classifier.pkl', 'rb')
heart_model = pickle.load(heart_in)

diabetes_in = open('diabetes_classifier.pkl', 'rb')
diabetes_model = pickle.load(diabetes_in)

#sidebar for navigate

with st.sidebar:
    selected=option_menu('Multiple Disease Detection System by Rishabh Patil',
                         ['Diabetes Prediction',
                          'Heart Disease Prediction'],
                          icons=['activity','heart'],
                          default_index=0)
    
if selected=='Diabetes Prediction':

    df=pd.read_csv('diabetes.csv')

    X=df.drop(columns='Outcome',axis=1)

    scaler=StandardScaler()
    scaler.fit(X)

    st.title('Diabetes Prediction using ML')

    col1,col2,col3=st.columns(3)

    with col1:
        Glucose=st.number_input('Glucose Level')
        insulin=st.number_input('insulin level')
    with col2:
        bp=st.number_input('Blood Pressure Level')
        BMI=st.number_input('BMI index')
    with col3:
        skinthick=st.number_input('skin thickness')
        DPF=st.number_input('Diabetes Pedigree Function value')

    Pregnancies=st.slider('Pregnancies',0,10,2)
    Age=st.slider('How old are You',0,130,25)

    diab_diag=''
    X = np.array([Pregnancies,Glucose,bp,skinthick,insulin,BMI,DPF,Age]) # here should be your X in np.array format
    X_transformed = scaler.transform(X.reshape(1, -1))

    if st.button('Diabetes Test Result'):

        pred=diabetes_model.predict(X_transformed)

        if(pred[0]==1):
            diab_diag='The Person is Diabetic'
        else:
            diab_diag='The Person is Not Diabetic'
        
    st.success(diab_diag)

    

if selected=='Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    df=pd.read_csv('heart_disease_data.csv')
    X=df.drop(columns='target',axis=1)

    scaler=StandardScaler()
    scaler.fit(X)

    col1,col2,col3=st.columns(3)

    with col1:
        gen = st.radio("What's your gender?",('Male', 'Female'))
        if gen=='Male':
            gender=1
        else:
            gender=0
        
        age=st.slider('How old are You',0,130,25)
        chol=st.slider('serum cholestoral in mg/dl(chol)',120,600,130)
        oldpeak=st.slider('ST depression induced by exercise relative to rest(oldpeak)',0,7,1)
        slope = st.slider("the slope of the peak exercise ST segment",0,2,1,step=1)
        
    with col2:
        fbsv = st.radio("fasting blood sugar > 120 mg/dl(fbs)",('True', 'False'))
        if fbsv=='True':
            fbs=1
        else:
            fbs=0
        
        cp = st.slider("Chest Pain Type?(cp)",0,3,1,step=1)

        restecg= st.slider("resting electrocardiographic results(restecg)",0,3,1,step=1)
        ca = st.slider("number of major vessels (0-3) colored by flourosopy(ca)",0,3,1,step=1)
    with col3:
        exan = st.radio("exercise induced angina(exang)",('Yes', 'No'))
        if exan=='Yes':
            exang=1
        else:
            exang=0
        
        trestbps=st.slider('Resting Blood Pressure(trestbps)',80,200,82)

        thalach=st.slider('maximum heart rate achieved(thalach)',60,250,65)
        thal = st.slider("thal: 0 = normal; 1 = fixed defect; 2 = reversable defect (thal)",0,3,1,step=1)
    
    heart_diag=''
    X = np.array([age,gender,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]) # here should be your X in np.array format
    X_transformed = scaler.transform(X.reshape(1, -1))

    if st.button('Heart Disease Test Result'):

        pred=heart_model.predict(X_transformed)

        if(pred[0]==1):
            heart_diag='The Person has Heart Disease'
        else:
            heart_diag='The Person does not have  Heart Disease'
        
    st.success(heart_diag)



    

    


        
    