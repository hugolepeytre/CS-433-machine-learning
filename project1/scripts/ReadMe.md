TODO : Enlever les .ipynb, les .png, le dossier checkpoints
Don't forget to zip both data and scripts folders
# ML Project 1 documentation

## 1 - Organisation of the folder :

This folder contains several python modules, categorized as follows :

### Machine Learning models :
- gradient_descent.py
- stochastic_gradient_descent.py
- least_squares.py
- ridge_regression.py
- logistic_regression.py
- regularized_logistic_regression.py

### Helper machine learning modules
- costs.py : All the cost functions and gradients
- cross_validation.py : A pipeline for cross-validation and hyper-parameter tuning
- pipeline.py : A pipeline for training an arbitrary model with arbitrary parameters
- data_processing.py : Methods for data processing
- build_polynomial.py : Performs polynomial expansion of the data

### Files needed for submission
- implementations.py : Self-sufficient module containing the machine learning models created in lab sessions
- run.py : Script used to produce our best prediction for the Higgs challenge. Usage is explained below

### General helper functions
- helpers.py : Various helper methods
- __init__.py : Empty file used to have imports work on all systems

## 2 - Final model presentation :

## 3 - run.py script usage :
