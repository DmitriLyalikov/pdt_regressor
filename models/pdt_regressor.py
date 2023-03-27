"""
Author: Dmitri Lyalikov
Version: 0.1
Date: 3/27/2023

Pendant Drop Tensiometry Extreme Gradient Boosted Regressor
"""

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor

df_bikes = pd.read_csv('../data/census_cleaned.csv')
print(df_bikes.head())
X_bikes = df_bikes.iloc[:,:-1]
y_bikes = df_bikes.iloc[:,-1]

# initialise our XGBRegressor, where the most important hyperparameter
# defaults are explicitly given
# We are doing a regression problem, so our objective is to minimize
# the squared error between predictions and real value
xgb = XGBRegressor(booster='gbtree', objective='reg:squarederror',
                   max_depth=6, learning_rate=0.1, n_estimators=100,
                   random_state=2, n_jobs=-1)


# Fit and score the regressor with cross_val_score
# with cvs, fitting and scoring are done in one step using the model
scores = cross_val_score(xgb, X_bikes, y_bikes, scoring='neg_mean_squared_error', cv=5)

#rmse = np.sqrt(-scores)
#print('RMSE:', np.round(rmse, 3))
#print('RMSE mean: %0.3f' % (rmse.mean()))

# Give the quartiles and general statistics for predictor column
# A score of 63.124 is less than 1 standard deviation


# Stratified fold includes the same percentage of target values in each fold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)

#scores = cross_val_score(xgb, X, y, cv=kfold)
#print('Accuracy:', np.round(scores, 2))

#print('Accuracy mean: %0.2f' % (scores.mean()))

# use the same folds to obtain new scores when fine-tuning hyperparameters.
# with GridSearchCV and RandomizedSearchCV
# GridsearchCV searches all possible combinations in a hyperparameter
# to find the best results. RandomizedSearchCV selects 10 random hyperparameters by default


# define a grid search function with params dictionary as input
def grid_search(params, random=False):
    xgb_model = XGBRegressor(booster='gbtree',objective='reg:squarederror', random_state=2)
    if random:
        grid = RandomizedSearchCV(xgb_model, params, cv=kfold, n_iter=20, n_jobs=-1)
    else:
        grid = GridSearchCV(xgb_model, params, cv=kfold, n_jobs=-1)
    grid.fit(X_bikes, y_bikes)
    best_params = grid.best_params_
    print("Best params:", best_params)
    best_score = grid.best_score_
    print("Training score: {:.3f}".format(best_score))


""" 
Important XGBoost Hyperparamers:
n_estimators: default 100. (1..inf). (number of trees in ensembled)
    - increasing may improve scores with large data
learning_rate: default 0.3 (0..inf). Shrinks the tree weights in each round of boosting
    - Decreasing prevents overfitting
max_depth: default 6 (0..inf). Depth of the tree
    - Decreasing prevents overfitting
"""

# n_estimators provides the number of tress in the ensemble
# initialize a grid search of n_estimators with default of 100
# Then double the numbers of trees through 800:
grid_search(params={'n_estimators': [100, 200, 400, 800]})
# Since we have a small dataset, increasing n_estimators did not produce
# better results

# learning_rate shrinkgs the weights of trees for each round of boosting
# by lowering the learning rate, more trees are required to produce better scores
# This prevents overfitting because the size of the weights carried forward
# is smaller
grid_search(params={'learning_rate':[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]})


# max_depth determines the length of the tree, equivalent to the number
# of rounds of splitting. Limiting max depth prevents overfitting because
# the individual trees can only grow as far as max_depth allows.
grid_search(params={'max_depth':[2, 3, 5, 6, 8]})

# gamma: (lagrange multiplier) provides a threhold that nodes must surpass
# before making further splits according to the loss function
grid_search(params={'gamma':[0, 0.1, 0.5, 1, 2, 5]})

# Minimum child weight refers to the minimum sum of weights required for a node
# to split into a child. reduces overfitting by increasing
grid_search(params={'min_child_weight':[1, 2, 3, 4, 5]})

