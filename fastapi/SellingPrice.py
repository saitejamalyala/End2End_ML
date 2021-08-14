from pydantic import BaseModel

class SellingPrice(BaseModel):
    ip_Year: int = 2015
    ip_Present_Price: float = 8.9
    ip_Kms_Driven: int = 45000
    Owner: int = 1
    ip_Fuel_type: str = "Diesel"
    ip_Seller_Type_Individual: int = 1
    ip_Transmission: str = "Automatic"


"""    input_data = {
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
"""