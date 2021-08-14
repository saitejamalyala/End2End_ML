from pathlib import Path
import sys
import os
sys.path.append('.././')
from End2End_ML.src import create_dataset,create_feat_dataset

from End2End_ML import config,sweep_config

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge,ElasticNet
import pickle
import wandb

def get_regressor(model_name:str): 
    if model_name == 'RandomForestRegressor':
        regressor=RandomForestRegressor(n_estimators=config['n_estimators'], 
                                    max_depth=config['max_depth'],
                                    max_features=config['max_features'],
                                    min_samples_split=config['min_samples_split'],
                                    min_samples_leaf=config['min_samples_leaf'],
                                    random_state=config['random_seed'],
                                    warm_start=config['warm_start'],)
        return regressor

    elif model_name == 'GradientBoostingRegressor':
        regressor=GradientBoostingRegressor(n_estimators=config['n_estimators'], 
                                    max_depth=config['max_depth'],
                                    max_features=config['max_features'],
                                    min_samples_split=config['min_samples_split'],
                                    min_samples_leaf=config['min_samples_leaf'],
                                    random_state=config['random_seed'],
                                    warm_start=config['warm_start'],
                                    learning_rate=config['learning_rate'],)
        return regressor

    elif model_name == 'AdaBoostRegressor':
        regressor=AdaBoostRegressor(n_estimators=config['n_estimators'], 
                                    learning_rate=config['learning_rate'],
                                    random_state=config['random_seed'],)
        return regressor
    elif model_name == 'DecisionTreeRegressor':
        regressor=DecisionTreeRegressor(max_depth=config['max_depth'],
                                    max_features=config['max_features'],
                                    min_samples_split=config['min_samples_split'],
                                    min_samples_leaf=config['min_samples_leaf'],)
        return regressor
    elif model_name == 'KNeighborsRegressor':
        regressor=KNeighborsRegressor(n_neighbors=config['n_neighbors'])
        return regressor
    elif model_name == 'Ridge':
        regressor=Ridge()
        return regressor
    elif model_name == 'ElasticNet':
        regressor=ElasticNet()
        return regressor

wandb.init(project="stepstone-demo",config=config)

# Create, clean dataset
raw_data = create_dataset.load_raw_data()

create_dataset.clean_Data(raw_data)

clean_df = create_dataset.load_clean_data()


# create feature dataset
X,Y = create_feat_dataset.create_feat_dataset(clean_df)


# Split dataset into train and test and fit regressor
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=config['random_seed'])

regressor = get_regressor(config['model_name'])
regressor.fit(X_train,Y_train)
wandb.sklearn.plot_regressor(regressor, X_train, X_test, Y_train, Y_test,  model_name=config['model_name'])


# predict and log metrics #
preds=regressor.predict(X_test)
rmse = mean_squared_error(Y_test,preds)**0.5
r2 = r2_score(Y_test,preds)

metrics = {"mean_squared_error": rmse,
               "r2_score": r2}
wandb.log(metrics)
print(f'rmse:{rmse}, r2_score:{r2}')


######################################## Model artifacting ##########################################
# save the model to disk
filename = f'{config["model_name"]}_rmse_{rmse:.2f}_r2_{r2:0.2f}.pkl'
filename = os.path.join(os.getcwd(),*config['sweep_model_path'],filename)
with open(filename, 'wb') as f:
    pickle.dump(regressor, f)

artifact = wandb.Artifact('trained_model', type='models')

# Add a file to the artifact's contents
artifact.add_file(filename)

# Save the artifact version to W&B and mark it as the output of this run
wandb.run.log_artifact(artifact)
######################################## ****************** ##########################################
