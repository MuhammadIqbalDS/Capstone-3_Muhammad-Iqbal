# import library
import streamlit as st
import numpy as np
import pandas as pd
import pickle

#Judul Utama
st.title('Bank Deposit Rate Predictor')
st.text('This web can be used to predict your Deposit rate')



# Menambahkan sidebar
st.sidebar.header("Please input your features")

def create_user_input():
    # Numerical Features
    age = st.sidebar.slider('Age', min_value=18, max_value=95, value=30)
    balance = st.sidebar.number_input('Balance', min_value=-6847, max_value=36935, value=1000)
    campaign = st.sidebar.slider('Campaign Contacts', min_value=1, max_value=63, value=2)
    pdays = st.sidebar.slider('Previous Days Contact', min_value=-1, max_value=854, value=-1)

    # Categorical Features
    job = st.sidebar.selectbox('Job', ['admin', 'self-employed', 'services', 'housemaid', 'management', 
                                       'student', 'technician', 'blue-collar', 'entrepreneur', 
                                       'retired', 'unemployed'])
    housing = st.sidebar.radio('Housing Loan', ['yes', 'no'])
    loan = st.sidebar.radio('Personal Loan', ['yes', 'no'])
    contact = st.sidebar.radio('Contact Type', ['cellular', 'telephone'])
    month = st.sidebar.selectbox('Last Contact Month', 
                                 ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
    poutcome = st.sidebar.selectbox('Previous Campaign Outcome', ['unknown', 'other', 'success', 'failure'])

    # Creating a dictionary with user input
    user_data = {
        'age': age,
        'balance': balance,
        'campaign': campaign,
        'pdays': pdays,
        'job': job,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'poutcome': poutcome
    }

    # Convert the dictionary into a pandas DataFrame (for a single row)
    user_data_df = pd.DataFrame([user_data])
    
    return user_data_df

# Get customer data
data_customer = create_user_input()

# Membuat 2 kontainer
col1, col2 = st.columns(2)

# Kiri
with col1:
    st.subheader("Customer's Features")
    st.write(data_customer.transpose())

# Load model
with open(r'Final Model Bank Campaign.sav', 'rb') as f:
    model_loaded = pickle.load(f)
    
# Predict to data
kelas = model_loaded.predict(data_customer)
probability = model_loaded.predict_proba(data_customer)[0]  # Get the probabilities

# Menampilkan hasil prediksi

# Bagian kanan (col2)
with col2:
    st.subheader('Prediction Result')
    if kelas == 1:
        st.write('Class 1: This customer will deposit')
    else:
        st.write('Class 2: This customer will deposit')
    
    # Displaying the probability of the customer buying
    st.write(f"Probability of Deposit: {probability[1]:.2f}")  # Probability of class 1 (BUY)
