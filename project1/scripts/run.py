import numpy as np

from proj1_helpers import *
from split_data import *
from data_processing import *
from pipeline import * 
from cross_validation import *
from split_data import *
import sys

DATA_FOLDER = '../data/'
DATA_ZIP = DATA_FOLDER + 'datasets.zip'
DATA_TRAIN_PATH = DATA_FOLDER + 'train.csv'
DATA_TEST_PATH = DATA_FOLDER + 'test.csv'

# Parameters of ridge regression
DEGREES = [5, 15, 5, 9]
LAMBDA_ = 0.0003

# Parameters of logistic regression
GAMMAS = [0.14, 0.08, 0.14, 0.08]
DEGREE = 3
MAX_ITERS = 600


def fit_ridge(Y, X, IDS):
    W = []
    losses = []
    for i in range(len(Y)): 
        y_cat, tX_cat, ids_cat = Y[i], X[i], IDS[i]
        w, loss = model_data(y_cat, tX_cat, 'ridge_regression', poly_exp=DEGREES[i], lambda_=LAMBDA_)
        W.append(w)
        losses.append(loss)
    return W


def fit_logistic(Y, X, IDS):
    W = []
    losses = []
    for i in range(len(Y)):
        y_cat, tX_cat, ids_cat = Y[i], X[i], IDS[i]
        w, loss = model_data(y_cat, tX_cat, 'logistic_regression', poly_exp=DEGREE, gamma=GAMMAS[i], max_iters=MAX_ITERS)
        W.append(w)
        losses.append(loss)
    return W


def prediction_by_cat(model, W, X, IDS):
    y_pred = np.array([])
    ids_pred = np.array([])
    for i, (w, tX, ids) in enumerate(zip(W, X, IDS)):
        if model == 'ridge_regression':
            tX_poly = build_poly_2D(tX, DEGREES[i])
        elif model == 'logistic_regression':
            tX_poly = build_poly_2D(tX, DEGREE)
        else:
            raise ValueError("Wrong arguments : only 'ridge_regression' or 'logistic_regression' are available")
        prediction = predict_labels(w, tX_poly)
        y_pred = np.concatenate((y_pred, prediction)) if len(y_pred) else prediction
        ids_pred = np.concatenate((ids_pred, ids)) if len(ids_pred) else ids
    return y_pred, ids_pred


def create_by_cat_submission(model, Y_train, X_train, IDS_train, X_test, IDS_test):
    if model == 'ridge_regression':
        W = fit_ridge(Y_train, X_train, IDS_train)
    elif model == 'logistic_regression':
        W = fit_logistic(Y_train, X_train, IDS_train)
    else:
        raise ValueError("Wrong arguments : only 'ridge_regression' or 'logistic_regression' are available")
    y_pred, ids_pred = prediction_by_cat(model, W, X_test, IDS_test)
    OUTPUT_PATH = DATA_FOLDER + 'submission.csv'
    create_csv_submission(ids_pred, y_pred, OUTPUT_PATH)


def main():
    if (len(sys.argv) == 2) & ('regression' in str(sys.argv[1])):
        model = str(sys.argv[1])
        print("Loading training data...")
        y_train, tX_train, ids_train = load_csv_data(DATA_TRAIN_PATH)
        print("Loading test data...")
        y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
        print("Cleaning the data...")
        Y_train, X_train, IDS_train, Y_test, X_test, IDS_test = clean_by_cat(y_train, tX_train, ids_train, y_test, tX_test, ids_test)
        print("Predicting the labels...")
        create_by_cat_submission(model, Y_train, X_train, IDS_train, X_test, IDS_test)
    else:
        print("Wrong arguments : only 'ridge_regression' or 'logistic_regression' are available")


if __name__ == "__main__":
    main()
