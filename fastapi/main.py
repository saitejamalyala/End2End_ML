from fastapi import FastAPI
import uvicorn
from SellingPrice import SellingPrice
import numpy as np
import pandas as pd
import datetime
import pickle
import os

app = FastAPI()
model_path = os.path.join('../','assets','models','production','GradientBoostingRegressor_rmse_1.00_r2_0.92.pkl')
with open(model_path, 'rb') as f:
    regressor = pickle.load(f)


@app.get("/")
def index():
    return {"message": "Hello World"}

@app.get("/{name}")
def read_root(name: str = "Stepstone"):
    return {"Hello": f"This is a {name} Interview"}

@app.post("/stepstone/predict")
def predict_selling_price(data: SellingPrice):
    data = data.dict()
    print(data)
    ip_Year = data['ip_Year']
    ip_Present_Price = data['ip_Present_Price']
    ip_Kms_Driven = data['ip_Kms_Driven']
    ip_Owner = data['Owner']
    ip_No_years = datetime.datetime.now().year - ip_Year 
    ip_Fuel_type_Diesel = 1 if data['ip_Fuel_type']=='Diesel' else 0
    ip_Fuel_type_Petrol = 1 if data['ip_Fuel_type']=='Petrol' else 0
    ip_Seller_Type_Individual = 1 if data['ip_Seller_Type_Individual']==1 else 0
    ip_Tranmission_Manual = 1 if data['ip_Transmission']=='Manual' else 0
    print('Data loaded')
    X = np.array([[ip_Year, ip_Present_Price, ip_Kms_Driven, ip_Owner, ip_No_years, ip_Fuel_type_Diesel, ip_Fuel_type_Petrol, ip_Seller_Type_Individual, ip_Tranmission_Manual]])
    X = X.reshape(1,-1)
    print("Predicting.....")
    y_pred = regressor.predict(X)
    return {"Prediction": y_pred[0]}


"""    ip_Year: int = 2015
    ip_Present_Price: float = 8.9
    ip_Kms_Driven: int = 45000
    Owner: int = 1
    ip_Fuel_type: str = "Diesel"
    ip_Seller_Type_Individual: bool = True
    ip_Transmission: str = "Automatic"

"""
"""        'Year':ip_Year,
        'Present_Price':ip_Present_Price,
        'Kms_Driven':ip_Kms_Driven,
        'Owner':Owner,
        'No_Years':No_Years,
        'Fuel_Type_Diesel':Fuel_Type_Diesel,
        'Fuel_Type_Petrol':Fuel_Type_Petrol,
        'Seller_Type_Individual':Seller_Type_Individual,
        'Transmission': Transmission
"""
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.0', port=8000)