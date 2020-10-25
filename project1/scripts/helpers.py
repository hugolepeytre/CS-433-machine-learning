"""Various helper functions, most were given during the labs"""
import numpy as np
import csv


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Returns an iterator on splits of the original x and y (shuffled the same way)
    If num_batches*batch_size is bigger that the number of elements, the last elements of the
    iterator will be empty.
    :param y: First array to shuffle and split
    :param tx: Second array to shuffle and split, must have same length as the first one
    :param batch_size: Size of each split
    :param num_batches: Number of splits
    :param shuffle: If False, array is not shuffled before being split
    :return: An iterator on shuffled splits of the original x and y
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to AICrowd
    :param ids: event ids associated with each prediction
    :param y_pred: Predictions, a list of -1s and 1s
    :param name: string name of .csv output file to be created
    :return: Nothing
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction':int(r2)})


def predict_labels(weights, x):
    """
    Generates class prediction from weights and feature points, for models that use mean square error
    :param weights: Weights
    :param x: Feature points
    :return: Predictions
    """
    y_pred = np.dot(x, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    return y_pred


def predict_labels_logistic(weights, x):
    """
    Generates class prediction from weights and feature points, for models that use negative log-likelihood
    :param weights: Weights
    :param x: Feature points
    :return: Predictions
    """
    y_pred = np.dot(x, weights)
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1
    return y_pred
