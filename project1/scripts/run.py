import numpy as np

from proj1_helpers import *
from split_data import *
from data_processing import *
from pipeline import * 
from cross_validation import *
from split_data import *

DATA_FOLDER = '../data/'
DATA_ZIP = DATA_FOLDER + 'datasets.zip'
DATA_TRAIN_PATH = DATA_FOLDER + 'train.csv'
DATA_TEST_PATH = DATA_FOLDER + 'test.csv' 
DEGREE = 2

def fit(Y, X, IDS):
    W = []
    losses = []
    for i in range(len(Y)): 
        y_cat, tX_cat, ids_cat = Y[i], X[i], IDS[i]
        initial_w = np.zeros(tX_cat.shape[1])
        w, loss = model_data(y_cat, tX_cat, 'least_squares', initial_w=initial_w, poly_exp = DEGREE)
        W.append(w)
        losses.append(loss)
    return W

def prediction_by_cat(W, X, IDS):
    y_pred = np.array([])
    ids_pred = np.array([])
    for w, tX, ids in zip(W, X, IDS):
        tX_poly = build_poly_2D(tX, DEGREE)
        prediction = predict_labels(w, tX_poly)
        y_pred = np.concatenate((y_pred, prediction)) if len(y_pred) else prediction
        ids_pred = np.concatenate((ids_pred, ids)) if len(ids_pred) else ids
    return y_pred, ids_pred

def create_by_cat_submission(Y_train, X_train, IDS_train, X_test, IDS_test):
    W = fit(Y_train, X_train, IDS_train)
    y_pred, ids_pred = prediction_by_cat(W, X_test, IDS_test)
    OUTPUT_PATH = DATA_FOLDER + 'submission.csv'
    create_csv_submission(ids_pred, y_pred, OUTPUT_PATH)

def main():
    print("Loading training data...")
    y_train, tX_train, ids_train = load_csv_data(DATA_TRAIN_PATH)
    print("Loading test data...")
    y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    print("Fitting the model...")
    Y_train, X_train, IDS_train, Y_test, X_test, IDS_test = clean_by_cat(y_train, tX_train, ids_train, y_test, tX_test, ids_test)
    print("Predicting the labels...")
    create_by_cat_submission(Y_train, X_train, IDS_train, X_test, IDS_test)

if __name__ == "__main__":
    main()