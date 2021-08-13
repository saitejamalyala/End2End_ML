import os
from typing import Any, Tuple
import pandas as pd 
from pandas import DataFrame
from pathlib import Path
from seaborn import heatmap
from . import config
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor


def save_feature_importances(X,Y,model: ExtraTreesRegressor):
    """Saves feature importances to a csv file

    Args:
        X ([type]): [Features]
        Y ([type]): [Target labels]
        model (ExtraTreesRegressor): [Extra Trees Regressor model]
    """
    model = ExtraTreesRegressor()
    plt.figure(figsize=(15,8))
    feat_importances = pd.Series(model.fit(X,Y).feature_importances_,index=X.columns)
    feat_importances.plot(kind='barh')
    plt.title("Feature Importances")
    plt.xlabel("Importance score")
    plt.ylabel("Feature")
    plt.savefig(os.path.join(os.getcwd(),*config['feat_importance_plot']))

def create_feat_dataset(clean_dataset: DataFrame)->Tuple[DataFrame,DataFrame]:
    """Create features dataset from the clean dataset

    Args:
        clean_dataset (DataFrame): Cleaned pandas dataframe

    Returns:
        Tuple[DataFrame,DataFrame]: X: Features,Y: target values
    """
    # Convert categorical variable into dummy/indicator variablesconvert
    final_dataset=pd.get_dummies(clean_dataset,drop_first=True)
    #print(final_dataset.head())
    feat_stats=(final_dataset.describe(include='all'))
    try:
        stats_dir = os.path.join(os.getcwd(),*config['feat_data_set_stats'])
        feat_stats.to_csv(stats_dir)
    except:
        print("Error in writing feature data set statistics.")

    try:
        # Drop the target variable from the dataset
        X=final_dataset.drop('Selling_Price',axis=1)
        Y=final_dataset['Selling_Price']
        # Feature importances
        save_feature_importances(X,Y,model=ExtraTreesRegressor())
        X.to_csv(os.path.join(os.getcwd(),*config['ds_features']))
        Y.to_csv(os.path.join(os.getcwd(),*config['ds_target']))
        return X,Y
    except:
        print("Error in writing feature data set.")
        return None,None
    


def load_feature_dataset(path_to_feature_dataset):
    """
    Loads the feature dataset and returns it as a pandas dataframe.
    """
    path_to_feature_dataset = Path(path_to_feature_dataset)
    if not path_to_feature_dataset.exists():
        raise FileNotFoundError(f"{path_to_feature_dataset} does not exist.")
    feature_dataset = pd.read_csv(path_to_feature_dataset)

    return feature_dataset