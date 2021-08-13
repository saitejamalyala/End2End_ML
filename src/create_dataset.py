import pandas as pd 
from pandas import DataFrame
from pathlib import Path
import os
#import sys
#sys.path.append('../')
from End2End_ML import config

#from . import config

#is data set file exists
#add a row with zeros and check if clean)data works

def load_raw_data()->DataFrame:
    """Load Raw Data from CSV file

    Returns:
        DataFrame: pandas dataframe object
    """
    print(os.getcwd())
    data_path = Path(os.path.join(os.getcwd(),*config['raw_data_path']))
    df=pd.read_csv(data_path)
    return df 

def clean_Data(df:DataFrame):
    """Clean the raw data fram, remove unwanted columns and nans

    Args:
        df (DataFrame): Raw data frame

    """
    
    # Drop rows with missing values
    df=df.dropna()
    try:
        assert(
            set(list(df.columns.values))==set(['Car_Name',
                                                'Year',
                                                'Selling_Price',
                                                'Present_Price',
                                                'Kms_Driven',
                                                'Fuel_Type',
                                                'Seller_Type',
                                                'Transmission',
                                                'Owner'])
            )
        
        # Neglect the car name column
        clean_df=df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]

        # Number of years
        clean_df['No_Years'] = config['current_year']-clean_df['Year']

        #Drop year column as we have no need of it
        #clean_df.drop(['Year'],axis=1,inplace=True)  

        #Save the clean data frame
        clean_df.to_csv(os.path.join(os.getcwd(),*config['clean_data_path']),index=False)  
        
    except AssertionError:
        raise AssertionError("Columns in the data frame are not as expected")


def load_clean_data()->DataFrame:
    """Load Clean Data from CSV file

    Returns:
        DataFrame: pandas dataframe object
    """
    data_path = Path(os.path.join(os.getcwd(),*config['clean_data_path']))
    df=pd.read_csv(data_path)
    return df
