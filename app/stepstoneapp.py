import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import datetime
from PIL import Image

GB_model_file = os.path.join(os.getcwd(),'assets','models','production','GradientBoostingRegressor_rmse_1.00_r2_0.92.pkl')
RF_model_file = os.path.join(os.getcwd(),'assets','models','production','RandomForestRegressor_rmse_1.91_r2_0.72.pkl')
model = 'Gradient Boosting Regressor'
feat_data = np.asarray([2015,2.52,25000,1,6,1,0,0,1]).reshape(1,-1)

def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model

def load_pretrained_models():
    """
    Load pretrained models
    """
    GB_model = load_model(GB_model_file)
    RF_model = load_model(RF_model_file)
    return GB_model, RF_model

def predict_price(model,X)->float:
    """AI is creating summary for predict_price function

    Args:
        model ([type]): [machine learning model]
        X ([type]): [input vector]

    Returns:
        [float]: [redicted selling price]
    """
    return model.predict(X)[0]

GB_model, RF_model = load_pretrained_models()

html_header="""
<div>
<h1 style="color:black;text-align:center;">Interview Demo</h1> 
</div>"""
st.markdown(html_header,unsafe_allow_html=True)

st.header('Demo of Used **Car Selling Price** prediction')
st.write("""
### Data used for training: [Kaggle Car Dekho Data set](https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho)
""")

image_main = Image.open(os.path.join(os.getcwd(),'assets','images','main.jpg'))

st.image(image_main,caption='copyright: https://autoportal.com/articles/valuation-of-used-cars-how-you-can-do-it-best-1883.html')

st.write('---')
st.sidebar.header('User Data Input')

def user_inputs()->np.ndarray:

    ip_Year = st.sidebar.date_input(label='Year of purchase',value=datetime.date(2015,6,23)).year
    ip_Present_Price = st.sidebar.slider(label='Present Price (in Hundred thousands INR)', min_value=1.0, max_value=35.0, value=7.0, key=None, 
            help='Enter current price of the Car', on_change=None, args=None, kwargs=None)
    ip_Kms_Driven = st.sidebar.slider(label='Odo meter reading (in Km)',min_value=100,max_value=200000,help='Number of kilometres driven',on_change=None,value=15000)
    Owner = st.sidebar.number_input(label="Number of previous owners", min_value=0,max_value=5,on_change=None)
    No_Years = datetime.datetime.now().year - ip_Year
    ip_Fuel_type = st.sidebar.selectbox(label='Fuel Type', options=['Petrol', 'Diesel', 'CNG'],on_change=None)

    if ip_Fuel_type == 'Petrol':
        Fuel_Type_Diesel = 0
        Fuel_Type_Petrol = 1
    elif ip_Fuel_type == 'Diesel':
        Fuel_Type_Diesel = 1
        Fuel_Type_Petrol = 0
    elif ip_Fuel_type == 'CNG':
        Fuel_Type_Diesel = 0
        Fuel_Type_Petrol = 0

    ip_Seller_Type_Individual = st.sidebar.radio(label='Seller Type', options=['Individual', 'Dealer','Organization'],on_change=None) 
    if ip_Seller_Type_Individual == 'Individual':
        Seller_Type_Individual = 1
    else:
        Seller_Type_Individual = 0  

    ip_Transmission = st.sidebar.selectbox(label='Transmission Type', options=['Manual', 'Automatic'],on_change=None)
    if ip_Transmission == 'Manual':
        Transmission = 1
    else:
        Transmission = 0

    input_data = {
        'Year':ip_Year,
        'Present_Price':ip_Present_Price,
        'Kms_Driven':ip_Kms_Driven,
        'Owner':Owner,
        'Fuel_type':ip_Fuel_type,
        'Seller_Type_Individual':ip_Seller_Type_Individual,
        'Transmission':ip_Transmission
    }
    feature_data = {
        'Year':ip_Year,
        'Present_Price':ip_Present_Price,
        'Kms_Driven':ip_Kms_Driven,
        'Owner':Owner,
        'No_Years':No_Years,
        'Fuel_Type_Diesel':Fuel_Type_Diesel,
        'Fuel_Type_Petrol':Fuel_Type_Petrol,
        'Seller_Type_Individual':Seller_Type_Individual,
        'Transmission': Transmission
    }
    input_data = pd.DataFrame(input_data, index=[0])
    feature_data = pd.DataFrame(feature_data, index=[0])
    return input_data, feature_data


# Show input data and feature data
inp_data,feat_data = user_inputs()

st.subheader('User inputs')
st.write(inp_data)
st.subheader('Feature vector')
st.write(feat_data)


#Year	Present_Price	Kms_Driven	Owner	No_Years	Fuel_Type_Diesel	Fuel_Type_Petrol	Seller_Type_Individual	Transmission_Manual
st.write('---')
st.subheader('Prediction Model')
model = st.radio(label='Model',options=['Gradient Boosting Regressor', 'Random Forest Regressor'],key='mlmodel')

if model == 'Gradient Boosting Regressor':
    model = GB_model
elif model == 'Random Forest Regressor':
    model = RF_model

def predict_callback():
    st.write('---')

    st.write("""
    ### Predicted Selling Price : {}""".format(round(predict_price(model, feat_data),4)))
    st.write('---')

if st.button('Predict'):
    predict_callback()

