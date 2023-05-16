import pandas as pd
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold

from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


df = pd.read_csv('../data/pdt-training-set.csv').apply(lambda x: x*1000000).astype('int64')
df.head()
#%%
# This model predicts Beta given a Pendant Drop Profile
X = df.drop('Beta', axis=1)
y = df['Beta']

# Stratified fold includes the same percentage of target values in each fold.
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
# This function takes a list of hyperparameter configs and finds the best one.
def grid_search(params, random=False):
    # Initialize XGB Regressor with objective='reg:squarederror' (MSE)
    xgb = XGBRegressor(booster='gbtree', objective='reg:squarederror',
    random_state=2)
    if random:
        grid = RandomizedSearchCV(xgb, params, cv=kfold, n_iter=20, n_jobs=-1)
    else:
        grid = GridSearchCV(xgb, params, cv=kfold, n_jobs=-1)
    grid.fit(X, y)
    best_params = grid.best_params_
    print("Best params:", best_params)
    best_score = grid.best_score_
    print("Training score: {:.3f}".format(best_score))

    best_params = grid.best_params_

    # Specify the file path
    file_path = "best_params.txt"

    # Open the file in write mode
    with open(file_path, 'w') as f:
        # Write the best parameters to the file
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")

params = {
'learning_rate': [0.1, 0.01, 0.001],    # Learning rate
'n_estimators': [400, 800, 1200],       # Number of trees
'max_depth': [3, 5, 7],                  # Maximum depth of each tree
'min_child_weight': [1, 3, 5],           # Minimum sum of instance weight needed in a child
'subsample': [0.6, 0.8, 1.0],            # Subsample ratio of the training instance
'colsample_bytree': [0.6, 0.8, 1.0],     # Subsample ratio of columns when constructing each tree
'gamma': [0, 0.1, 0.2]
}

grid_search(params=params)