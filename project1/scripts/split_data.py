import numpy as np

def split_data(x, y, ratio, seed=1):
    """
    Split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    
    N = len(x)
    split = round(ratio * N)
    indices = np.random.permutation(N) # List on suffled indices
    train_idx, test_idx = indices[: split ], indices[split :] #Train and test indices
    x_train, x_test = x[train_idx], x[test_idx] #Extracting the features from the indices
    y_train, y_test = y[train_idx], y[test_idx] #Extracting the features from the indices
    
    return x_train, x_test, y_train, y_test