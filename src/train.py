from pathlib import Path
import sys
import os
sys.path.append('.././')
from End2End_ML.src import create_dataset,create_feat_dataset

from End2End_ML import config,sweep_config

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import wandb


#wandb.init(project="stepstone-demo",config=config)

# Create, clean dataset
raw_data = create_dataset.load_raw_data()

create_dataset.clean_Data(raw_data)

clean_df = create_dataset.load_clean_data()
#create_dataset.clean_Data(create_dataset.load_raw_data())
clean_df.head(2)

# create feature dataset
X,Y = create_feat_dataset.create_feat_dataset(clean_df)


# Split dataset into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=config['random_seed'])

regressor=RandomForestRegressor(n_estimators=config['n_estimators'], 
                                max_depth=config['max_depth'],
                                max_features=config['max_features'],
                                min_samples_split=config['min_samples_split'],
                                min_samples_leaf=config['min_samples_leaf'],
                                random_state=config['random_seed'],
                                warm_start=config['warm_start'],)

"""regressor = GradientBoostingRegressor(n_estimators=3215, learning_rate=0.1,
                                      max_depth=40, max_features='auto',
                                      min_samples_split=20, min_samples_leaf=8,
                                      random_state=config['random_seed'])
"""
regressor.fit(X_train,Y_train)
#wandb.sklearn.plot_regressor(regressor, X_train, X_test, Y_train, Y_test,  model_name='Random Forest Regressor')

preds=regressor.predict(X_test)
rmse = mean_squared_error(Y_test,preds)**0.5
r2 = r2_score(Y_test,preds)

metrics = {"mean_squared_error": rmse,
               "r2_score": r2}
#wandb.log(metrics)

print(f'rmse:{rmse}, r2_score:{r2}')
