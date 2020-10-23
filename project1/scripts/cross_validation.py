from pipeline import *
import matplotlib.pyplot as plt


def find_best_parameter(y, tx, model, param, values, logspace=True, k_fold=4, initial_w=[], max_iters=1000, gamma=0.1,
                        lambda_=0.1, poly_exp=1, seed=1):
    # split data in k fold
    k_indices = build_k_indices(y.shape[0], k_fold, seed)
    # define lists to store the loss of training data and test data
    losses_t = []
    losses_v = []

    for v in values:
        loss_t_avg = 0
        loss_v_avg = 0
        for k in range(k_fold):
            if param == 'max_iters':
                loss_t, loss_v = cross_validation(y, tx, k_indices, k, model, initial_w, v, gamma, lambda_, poly_exp)
            if param == 'gamma':
                loss_t, loss_v = cross_validation(y, tx, k_indices, k, model, initial_w, max_iters, v, lambda_, poly_exp)
            if param == 'lambda_':
                loss_t, loss_v = cross_validation(y, tx, k_indices, k, model, initial_w, max_iters, gamma, v, poly_exp)
            if param == 'poly_exp':
                loss_t, loss_v = cross_validation(y, tx, k_indices, k, model, initial_w, max_iters, gamma, lambda_, v)
            loss_t_avg += loss_t
            loss_v_avg += loss_v
        losses_t.append(loss_t_avg/k_fold)
        losses_v.append(loss_v_avg/k_fold)

    cross_validation_visualization(values, losses_t, losses_v, logspace=logspace)
    best_index = np.argmin(losses_v)
    return values[best_index], losses_v[best_index]


def build_k_indices(N, k_fold, seed=1):
    """build k indices for k-fold."""
    interval = int(N / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(N)
    k_indices = np.array([indices[k * interval:(k + 1) * interval]
                          for k in range(k_fold)])
    return k_indices


def cross_validation(y, tx, k_indices, k, model, initial_w=[], max_iters=1000, gamma=0.1, lambda_=0.1, poly_exp=1):
    """return the loss of ridge regression."""
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
    # Validation loss
    if 'logistic' in model:
        loss_v = get_log_likelihood(y_v, tx_v, w)
    else:
        loss_v = get_loss(y_v, tx_v, w)
    return loss_t, loss_v


def cross_validation_visualization(values, losses_t, losses_v, logspace=True):
    """visualization the curves of mse_tr and mse_te."""
    plt.figure()
    if logspace:
        plt.semilogx(values, losses_t, marker=".", color='b', label='Train error')
        plt.semilogx(values, losses_v, marker=".", color='r', label='Test error')
    else:
        plt.plot(values, losses_t, marker=".", color='b', label='Train error')
        plt.plot(values, losses_v, marker=".", color='r', label='Test error')
    plt.xlabel("Values")
    plt.ylabel("Test error")
    plt.title("Cross Validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.show()
