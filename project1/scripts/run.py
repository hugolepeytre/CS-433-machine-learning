from helpers import *
from data_processing import clean_by_cat
from pipeline import *
import sys

DATA_FOLDER = '../data/'
DATA_TRAIN_PATH = DATA_FOLDER + 'train.csv'
DATA_TEST_PATH = DATA_FOLDER + 'test.csv'

# Parameters of ridge regression
RIDGE_DEGREES = [5, 5, 5, 9]
RIDGE_LAMBDA_ = 0.0003

# Parameters of logistic regression
LOGISTIC_GAMMAS = [0.14, 0.08, 0.14, 0.08]
LOGISTIC_DEGREE = 3
LOGISTIC_MAX_ITERS = 600


def fit_ridge(y, x, ids):
    """
    Trains the ridge regression model using the hard-coded hyperparameters
    :param y: Labels
    :param x: Feature points
    :param ids: Ids
    :return: The trained weight vector
    """
    ws = []
    losses = []
    for i in range(len(y)):
        y_cat, tx_cat, ids_cat = y[i], x[i], ids[i]
        w, loss = model_data(y_cat, tx_cat, 'ridge_regression', poly_exp=RIDGE_DEGREES[i], lambda_=RIDGE_LAMBDA_)
        ws.append(w)
        losses.append(loss)
    return ws


def fit_logistic(y, x, ids):
    """
    Trains the logistic regression model using the hard-coded hyperparameters
    :param y: Labels
    :param x: Feature points
    :param ids: Ids
    :return: The trained weight vector
    """
    ws = []
    losses = []
    for i in range(len(y)):
        y_cat, tx_cat, ids_cat = y[i], x[i], ids[i]
        w, loss = model_data(y_cat, tx_cat, 'logistic_regression',
                             poly_exp=LOGISTIC_DEGREE, gamma=LOGISTIC_GAMMAS[i], max_iters=LOGISTIC_MAX_ITERS)
        ws.append(w)
        losses.append(loss)
    return ws


def prediction_by_cat(model, w, x, ids):
    """
    Makes prediction using the trained weight vector, and associates them with ids
    :param model: Type of model to train (logistic regression or ridge regression)
    :param w: Trained weight vector
    :param x: Unlabeled feature points
    :param ids: Ids of feature points
    :return: The predictions, with the associated ids
    """
    y_pred = np.empty(0)
    ids_pred = np.empty(0)
    for i, (w_cat, tx_cat, ids_cat) in enumerate(zip(w, x, ids)):
        if model == 'ridge_regression':
            tx_poly = build_poly_2D(tx_cat, RIDGE_DEGREES[i])
            prediction = predict_labels(w_cat, tx_poly)
        elif model == 'logistic_regression':
            tx_poly = build_poly_2D(tx_cat, LOGISTIC_DEGREE)
            prediction = predict_labels_logistic(w_cat, tx_poly)
        else:
            raise ValueError("Wrong arguments : only 'ridge_regression' or 'logistic_regression' are available")
        y_pred = np.r_[y_pred, prediction]
        ids_pred = np.r_[ids_pred, ids_cat]
    return y_pred, ids_pred


def create_by_cat_submission(model, y_train, x_train, ids_train, x_test, ids_test):
    """
    Trains a model on the given data, does the predictions and creates a prediction file as ../data/submission.csv
    :param model: Type of model to train (logistic regression or ridge regression)
    :param y_train: Labels
    :param x_train: Feature points
    :param ids_train: Ids
    :param x_test: Feature points of unlabeled test data
    :param ids_test: Ids of unlabeled test data
    :return: Nothing
    """
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
    """
    Main function. Loads the data, does data processing and makes prediction file.
    Takes the argument given to the script. Only accepts 'ridge_regression' and 'logistic_regression'
    :return: Nothing
    """
    if len(sys.argv) == 2:
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
