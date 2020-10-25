from helpers import *
from data_processing import clean_by_cat
from pipeline import *
import sys

DATA_FOLDER = '../data/'
DATA_TRAIN_PATH = DATA_FOLDER + 'train.csv'
DATA_TEST_PATH = DATA_FOLDER + 'test.csv'

# Parameters of ridge regression
DEGREES = [5, 5, 5, 9]
LAMBDA_ = 0.0003

# Parameters of logistic regression
GAMMAS = [0.14, 0.08, 0.14, 0.08]
DEGREE = 3
MAX_ITERS = 600


def fit_ridge(y, x, ids):
    ws = []
    losses = []
    for i in range(len(y)):
        y_cat, tx_cat, ids_cat = y[i], x[i], ids[i]
        w, loss = model_data(y_cat, tx_cat, 'ridge_regression', poly_exp=DEGREES[i], lambda_=LAMBDA_)
        ws.append(w)
        losses.append(loss)
    return ws


def fit_logistic(y, x, ids):
    ws = []
    losses = []
    for i in range(len(y)):
        y_cat, tx_cat, ids_cat = y[i], x[i], ids[i]
        w, loss = model_data(y_cat, tx_cat, 'logistic_regression', poly_exp=DEGREE, gamma=GAMMAS[i], max_iters=MAX_ITERS)
        ws.append(w)
        losses.append(loss)
    return ws


def prediction_by_cat(model, w, x, ids):
    # TODO : change
    y_pred = np.array([])
    ids_pred = np.array([])
    for i, (w_cat, tx_cat, ids_cat) in enumerate(zip(w, x, ids)):
        if model == 'ridge_regression':
            tx_poly = build_poly_2D(tx_cat, DEGREES[i])
            prediction = predict_labels(w, tx_poly)
        elif model == 'logistic_regression':
            tx_poly = build_poly_2D(tx_cat, DEGREE)
            prediction = predict_labels_logistic(w, tx_poly)
        else:
            raise ValueError("Wrong arguments : only 'ridge_regression' or 'logistic_regression' are available")
        y_pred = np.concatenate((y_pred, prediction)) if len(y_pred) else prediction
        ids_pred = np.concatenate((ids_pred, ids)) if len(ids_pred) else ids
    return y_pred, ids_pred


def create_by_cat_submission(model, y_train, x_train, ids_train, x_test, ids_test):
    if model == 'ridge_regression':
        w = fit_ridge(y_train, x_train, ids_train)
    elif model == 'logistic_regression':
        w = fit_logistic(y_train, x_train, ids_train)
    else:
        raise ValueError("Wrong arguments : only 'ridge_regression' or 'logistic_regression' are available")
    y_pred, ids_pred = prediction_by_cat(model, w, x_test, ids_test)
    OUTPUT_PATH = DATA_FOLDER + 'submission.csv'
    create_csv_submission(ids_pred, y_pred, OUTPUT_PATH)


def main():
    if (len(sys.argv) == 2) & ('regression' in str(sys.argv[1])):
        model = str(sys.argv[1])
        print("Loading training data...")
        y_train, tx_train, ids_train = load_csv_data(DATA_TRAIN_PATH)
        print("Loading test data...")
        y_test, tx_test, ids_test = load_csv_data(DATA_TEST_PATH)
        print("Cleaning the data...")
        Y_train, X_train, IDS_train, Y_test, X_test, IDS_test = clean_by_cat(y_train, tx_train, ids_train, y_test, tx_test, ids_test)
        print("Predicting the labels...")
        create_by_cat_submission(model, Y_train, X_train, IDS_train, X_test, IDS_test)
    else:
        print("Wrong arguments : only 'ridge_regression' or 'logistic_regression' are available")


if __name__ == "__main__":
    main()
