import sys
sys.path.append('.././')
from End2End_ML.src import create_dataset,create_feat_dataset

from End2End_ML import config
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# Create, clean dataset
raw_data = create_dataset.load_raw_data()

create_dataset.clean_Data(raw_data)

clean_df = create_dataset.load_clean_data()
#create_dataset.clean_Data(create_dataset.load_raw_data())
clean_df.head(2)

# create feature dataset
X,Y = create_feat_dataset.create_feat_dataset(clean_df)
X.head(2)


# Split dataset into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=config['random_seed'])

regressor=RandomForestRegressor(n_estimators=2000, max_depth=10)

regressor.fit(X_train,Y_train)

preds=regressor.predict(X_test)
rmse = mean_squared_error(Y_test,preds)**0.5
r2 = r2_score(Y_test,preds)

print(f'rmse:{rmse}, r2_score:{r2}')
