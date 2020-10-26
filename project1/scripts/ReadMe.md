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

### Hyperparameters tuning
- fine_tuning.ipynb : Find the best hyper parameters through cross-validation

### Files needed for submission
- implementations.py : Self-sufficient module containing the machine learning models created in lab sessions
- run.py : Script used to produce our best prediction for the Higgs challenge. Usage is explained below

### General helper functions
- helpers.py : Various helper methods
- __init__.py : Empty file used to have imports work on all systems

### Other
- ReadMe.md : This ReadMe file
- ML_Project_1_Report.pdf : The pdf of the report

## 2 - Final model presentation :
### Data Processing
Our data processing pipeline includes the following things :
- Setting -999 values to NaN
- Splitting the set into 4 categories depending on the value of PRI_jet_num feature
- For each split:
- - Removing the columns full of NaN
  - Filter out the rows containing NaNs (or set NaN to 0 in test set)
  - Remove outliers
  - Standardize the data

### Models and hyperparameters
Our run.py includes two different models, since they had very close accuracies in our fine-tuning (see fine-tuning.ipynb) and we wanted to try them both on AICrowd :

#### Model 1 : Ridge regression
This is a ridge regression model on polynomial expansion of the features using :
lambdas = 0.00027, 0.00019, 0.00010, 1.3*10^(-5) (for each category in order)
polynomial expansion degrees = 2, 2, 2, 2 (for each category in order)

#### Model 2 : Logistic regression
This is a logistic regression model on polynomial expansion of the features using :
gammas = 0.1179, 0.0848, 0.1179, 0.0610 (for each category in order)
number of iterations = 1000
polynomial expansion degree = 3, 2, 3, 3 (for each category in order)
## 3 - run.py script usage :
The script should be in the scripts folder with all the other modules from the zip file,
and you should put the test.csv and train.csv files in it.
Then, opening a terminal and running the command
'python run.py ridge_regression' or 'python3 run.py ridge_regression'
should output a submission.csv file in the folder, that represents our prediction for the ridge regression model
while the command
'python run.py logistic_regression' or 'python3 run.py logistic_regression'
should output our prediction for the logistic regression model
To get the predictions for our best submission on AICrowd, use 'logistic_regression'