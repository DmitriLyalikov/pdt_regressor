# pdt-regressor
An XGBoost Regressor that predicts interfacial tension and pressure from the edge profile of a pendant drop.

This is part of a larger processing pipeline where the input data is derived from the output of the [`pdt-canny-edge-detector`][1].  
profile data will be stored in the /data folder in .csv format. 

## Table of Contents
1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Usage](#usage)
4. [Feature Extraction](#Feature-Extraction)
5. [Hyperparameter Tuning](#Hyperparameter-Tuning)
6. [Results](#Results)
7. [Appendix](#appendix)

## Requirements
The finalized models, feature extraction, and data preparation will be placed in a single pdt_regressor.py application that can be ran as a complete system.
However, for development and understanding, it is often useful to use JupyterNotebooks to visualize our data and step through parameter tuning. For that reason,
it is recommended to use a JupyterNotebook enabled IDE such as [`Pycharm`][3]. Students and researchers are able to access the professional version for free.
## Setup
To use this project, and develop on it, either download the .zip file from the repository or
```
git clone https://github.com/DmitriLyalikov/pdt_regressor.git
```
Open the project in your IDE and run 
```pycon
pip install . e
```
in the PyCharm IDE terminal. This should install all the library dependencies for the project like scikit-learn, xgboost, and pandas.
## Usage
## Feature Extraction
## Hyperparameter Tuning
XGBoost Hyperparameters are used to improve the performance of the model, reduce variance, and minimize overfitting.
Some important HP are learning_rate (eta), max_depth, no_of_iterations, and subsamples. [`Complete list of XGBoost Hyperparameters can be found here`][2]

HP depend on the model, data, and methods of regression, and generally are found empirically. Included in XGBoost.ipynb is grid_search function which will automate the tuning process by finding the best parameter provided in params
```python
grid_search(params={'max_depth': [1, 2, 3, 4, 5, 6]})
```
This will yield the output: 
```pycon
Best params: {'max_depth': 6}
Training score: 951.398
```
For full examples of usage consult XGBoost.ipynb provided. Generally Hyperparameters should be tested together, as one may or may not have an effect on another's score
## Results
## Appendix
[1]: https://github.com/DmitriLyalikov/pdt-canny-edge-detector
[2]: https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster
[3]: https://www.jetbrains.com/pycharm/