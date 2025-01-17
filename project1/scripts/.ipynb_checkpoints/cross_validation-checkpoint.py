from pipeline import *
import matplotlib.pyplot as plt


def find_best_parameter(y, tx, model, param, values, logspace=True, k_fold=4, initial_w=[], max_iters=1000, gamma=0.1,
                        lambda_=0.1, poly_exp=1, seed=1):
    """

    :param y: Labels
    :param tx: Feature vector
    :param model: Which machine learning model to use, argument should be the name as a string
    (available : gradient_descent, stochastic_gradient_descent, least_squares, ridge_regression,
    logistic_regression, regularized_logistic_regression)
    :param param: Which parameter to optimize, argument should be the name as a string
    (available : max_iters, lambda_, gamma, poly_exp)
    :param values: The parameter's values to try
    :param logspace: For visualisation, set to False if values are on a linear scale, false if they are on a log scale
    :param k_fold: Number of subsets in which to divide the data
    :param initial_w: Initial weight vector
    :param max_iters: Number of iterations
    :param gamma: Learning rate
    :param lambda_: Regularization parameter
    :param poly_exp: Maximum degree to which to raise the features
    :param seed: Random seed
    :return: The training and validation losses for this instance of cross-validation
    """
    # split data in k fold
    k_indices = build_k_indices(y.shape[0], k_fold, seed)
    # define lists to store the loss of training data and test data
    accuracies_v = []
    accuracies_t = []

    for v in values:
        accuracy_t_avg = 0
        accuracy_v_avg = 0
        for k in range(k_fold):
            if param == 'max_iters':
                accuracy_t, accuracy_v = cross_validation(y, tx, k_indices, k, model,
                                                          initial_w, v, gamma, lambda_, poly_exp)
            elif param == 'gamma':
                accuracy_t, accuracy_v = cross_validation(y, tx, k_indices, k, model,
                                                          initial_w, max_iters, v, lambda_, poly_exp)
            elif param == 'lambda_':
                accuracy_t, accuracy_v = cross_validation(y, tx, k_indices, k, model,
                                                          initial_w, max_iters, gamma, v, poly_exp)
            elif param == 'poly_exp':
                accuracy_t, accuracy_v = cross_validation(y, tx, k_indices, k, model,
                                                          initial_w, max_iters, gamma, lambda_, v)
            else:
                raise ValueError('Invalid parameter')
            accuracy_t_avg += accuracy_t
            accuracy_v_avg += accuracy_v
        accuracies_t.append(accuracy_t_avg/k_fold)
        accuracies_v.append(accuracy_v_avg/k_fold)

    cross_validation_visualization(values, accuracies_t, accuracies_v, logspace=logspace)
    best_index = np.argmax(accuracies_v)
    return values[best_index], accuracies_v[best_index]


def build_k_indices(N, k_fold, seed=1):
    """
    Creates a set of k splits from a randomly shuffled range
    :param N: Number of elements
    :param k_fold: Number of splits
    :param seed: Random seed
    :return: An array of k arrays of indices, in random order
    """
    interval = int(N / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(N)
    k_indices = np.array([indices[k * interval:(k + 1) * interval]
                          for k in range(k_fold)])
    return k_indices


def cross_validation(y, tx, k_indices, k, model, initial_w=[], max_iters=1000, gamma=0.1, lambda_=0.1, poly_exp=1):
    """
    Computes losses for one split of training data
    :param y: Labels
    :param tx: Feature vector
    :param k_indices: Set of k splits in the data
    :param k: Index of the split which will be the training set
    :param model: Machine learning model to use
    :param initial_w: Initial weight vector
    :param max_iters: Number of iterations
    :param gamma: Learning rate
    :param lambda_: Regularization parameter
    :param poly_exp: Maximum degree to which to raise the features
    :return: The training and validation losses for this instance of cross-validation
    """
    # k-th group is test, the rest is train
    if poly_exp > 1:
        tx = build_poly_2D(tx, poly_exp)
    testing_indices = k_indices[k]
    training_indices = [idx for subgroup in np.vstack((k_indices[:k], k_indices[k+1:])) for idx in subgroup]
    tx_v = tx[testing_indices]
    y_v = y[testing_indices]
    tx_t = tx[training_indices]
    y_t = y[training_indices]
    # Model the data
    w, loss_t = model_data(y_t, tx_t, model, initial_w, max_iters, gamma, lambda_)
    # Compute accuracies
    train_predictions = 2*(((tx_t @ w) > 0) - .5)
    test_predictions = 2*(((tx_v @ w) > 0) - .5)
    # print(len(train_predictions))
    # print(len(test_predictions))
    train_accuracy = np.mean(y_t == train_predictions)
    test_accuracy = np.mean(y_v == test_predictions)
    # print(train_accuracy - test_accuracy)
    return train_accuracy, test_accuracy


def cross_validation_visualization(values, train_results, test_results, logspace=True):
    """
    Visualizes the training and validation error for all tested parameters
    :param values: Tested parameters
    :param train_results: Training losses
    :param test_results: Validation losses
    :param logspace: If false, tested values are on a linear scale. Else they are on a log scale
    :return:
    """
    plt.figure()
    if logspace:
        plt.semilogx(values, train_results, marker=".", color='b', label='Train accuracy')
        plt.semilogx(values, test_results, marker=".", color='r', label='Test accuracy')
    else:
        plt.plot(values, train_results, marker=".", color='b', label='Train accuracy')
        plt.plot(values, test_results, marker=".", color='r', label='Test accuracy')
    plt.xlabel("Values")
    plt.ylabel("Test error")
    #plt.ylim(0.7, 1)
    plt.title("Cross Validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.show()
