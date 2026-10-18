import numpy as np
import pandas as pd
import streamlit as st
import pickle



# get a data from the user

st.set_page_config(page_title='House Price Prediction', page_icon=':house:', layout='wide')

st.title('House Price Prediction')
st.write('This is a simple web application that predicts the price of a house based on its features. Please enter the details of the house' \
' below and click on the "Predict" button to get the predicted price.')


# Define the features that the user needs to input
features = {}
features['MedInc'] = st.number_input('Median Income (in $1000s)', min_value=0.0, max_value=20.0, value=5.0, step=0.1)
features['HouseAge'] = st.number_input('House Age (in years)', min_value=0.0, max_value=100.0, value=30.0, step=1.0)
features['AveRooms'] = st.number_input('Average Number of Rooms', min_value=0.0, max_value=150.0, value=3.0, step=0.1)
features['AveBedrms'] = st.number_input('Average Number of Bedrooms', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
features['Population'] = st.number_input('Population', min_value=0, max_value=400000, value=1000, step=500)
features['AveOccup'] = st.number_input('Average Occupancy', min_value=0.0, max_value=10.0, value=3.0, step=0.1)
features['Latitude'] = st.number_input('Latitude', min_value=32.0, max_value=42.0, value=37.0, step=0.1)
features['Longitude'] = st.number_input('Longitude', min_value=-124.0, max_value=-114.0, value=-119.0, step=0.1)


# Load the trained model
model = pickle.load(open('california_housing_model.pkl', 'rb'))

# Make the prediction
prediction = model.predict(pd.DataFrame(features, index=[0]))[0]

# Display the prediction
if st.button('Predict'):
    st.write(f'The predicted price of the house is ${prediction * 100000:.2f}')