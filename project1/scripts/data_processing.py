import numpy as np


def set_nan(x):
    """
    Label the -999 values as NaN
    """
    x[x == -999] = np.nan
    return x


def remove_empty_columns(x, threshold=0.4):
    """
    Remove feature columns containing more than the specified threshold proportion of NaN
    """
    # For each column compute the ratio of nan values over the number of rows
    prop_empty_column = (np.isnan(x)).sum(axis=0) / len(x)
    column_mask = prop_empty_column < threshold
    return x[:, column_mask], column_mask


def filter_nan(y, x, ids, remove=True, replace_val=0.0):
    """
    Filter the nan (-999) values, by either removing the rows or replacing by the specified replace_val.
    Remove as well 0 filled columns when remove is set to True
    """
    mask = np.isnan(x)
    
    if remove:
        # Remove the rows containing any NaN
        row_mask = ~mask.any(axis=1)  # Sets to False any rows containing NaN
        x_copy, y_copy, ids_copy = x[row_mask], y[row_mask], ids[row_mask]
        # Remove 0 filled columns
        col_mask = x_copy.sum(axis=0) != 0  # True if the columns is filled with 0
        x_copy = x_copy[:, col_mask] 
        return y_copy, x_copy, ids_copy, col_mask
    else:
        # Replace NaN values by replace_val
        x[mask] = replace_val
        return y, x, ids
    

def remove_outliers(y, x, ids):
    """
    Remove outliers feature points using Interquartile range.
    """
    # Compute first and third quartiles and the Interquartile range
    q1 = np.percentile(x, 25, axis=0)
    q3 = np.percentile(x, 75, axis=0)
    iqr = q3 - q1
    # Set to True any entry outside the Interquartile range
    mask = (x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)

    # Only filter out features with values that are spread over a range bigger than threshold_range
    # i.e. if the difference between the minimum value and the maximum value is bigger than threshold_range
    threshold_range = 10
    # Set to False any feature with range bigger than threshold
    col_mask = (x.max(axis=0) - x.min(axis=0)) < threshold_range
    mask = mask | col_mask
    row_mask = mask.all(axis=1)  # sets to False rows containing any outliers

    return y[row_mask], x[row_mask], ids[row_mask]


def scale(x, method="standard", x_test=None):
    """
    Scale features using the specified method. Possible methods: standard, min-max
    """
    if method == "standard": 
        # Standardize the data
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0] = 1  # No division by 0
        x = (x - mean) / std
        if x_test is not None:
            x_test = (x_test - mean) / std

    else:
        # Apply a min-max normalization to scale data between 0 and 1
        col_min = x.min(axis=0)
        col_max = x.max(axis=0)
        col_range = col_max - col_min
        x = (x - col_min) / col_range
        if x_test is not None:
            x_test = (x_test - col_min) / col_range
   
    return x, x_test


def remove_correlated_features(x, threshold=0.9):
    """
    Compute the correlations between each feature and remove features that have a correlation greater
    than the specified threshold
    """
    x_copy = np.copy(x)
    
    corr_matrix = np.corrcoef(x_copy, rowvar=False)
    # Set to False highly correlated columns
    nb_col = len(corr_matrix)
    columns = np.full((nb_col,), True, dtype=bool)
    for i in range(nb_col):
        for j in range(i+1, nb_col):
            if corr_matrix[i, j] >= threshold:
                if columns[i]:
                    columns[j] = False
    
    # Remove correlated features and concat categorical features
    return x_copy[:, columns], columns


def clean_training(y, x, ids):
    x = set_nan(x)
    x = remove_empty_columns(x, threshold=2)  # Temporary to 2 to not remove any column
    y, x, ids = filter_nan(y, x, ids, remove=False, replace_val=0.0)
    y, x, ids = remove_outliers(y, x, ids)
    x = scale(x, method="standard")[0]
    y, x, ids = remove_outliers(y, x, ids)
    x = np.c_[np.ones(len(x)), x]
    return y, x, ids


def clean_test(y, x, ids, x_train):
    x_nan = set_nan(x)
    x_columns = remove_empty_columns(x_nan, threshold=2)  # Temporary to 2 to not remove any column
    y, x, ids = filter_nan(y, x_columns, ids, remove=False, replace_val=0.0)
    x = scale(x_train[:, 1:], method="standard", x_test=x)[1]  # Scale test set with train set statistics
    x = np.c_[np.ones(len(x)), x]
    return y, x, ids


def group_by_cat(y, x, ids):
    """
    Return subsets (4 in fact) of the datasets, grouped by the category (PRI_jet_num feature).
    One subset will group features with category 0, another one 1 etc.
    """
    NB_CATEGORY = 4 
    # Column index of the categorical feature
    IDX_COL_CAT = np.where((x.max(axis=0) - x.min(axis=0)) == 3)[0][0]
    Y = []
    X = []
    IDS = []
    for i in range(NB_CATEGORY):
        row_idx = np.where(x[:, IDX_COL_CAT] == i)[0]  # index of the rows in category i
        x_cat = np.delete(x[row_idx], IDX_COL_CAT, axis=1)  # Remove category feature
        Y.append(y[row_idx])
        X.append(x_cat) 
        IDS.append(ids[row_idx])
    return Y, X, IDS


def clean_by_cat(y_train, x_train, ids_train, y_test, x_test, ids_test):
    Y_train, X_train, IDS_train = group_by_cat(y_train, x_train, ids_train)
    Y_test, X_test, IDS_test = group_by_cat(y_test, x_test, ids_test)

    for i in range(len(Y_train)):
        # Get the right category
        y_cat_train, x_cat_train, ids_cat_train = Y_train[i], X_train[i], IDS_train[i]
        y_cat_test, x_cat_test, ids_cat_test = Y_test[i], X_test[i], IDS_test[i]

        # Set -999 to NaN
        x_cat_train = set_nan(x_cat_train)
        x_cat_test = set_nan(x_cat_test)

        # Remove the (same) empty columns in the training and test
        x_cat_train, column_mask = remove_empty_columns(x_cat_train, threshold=0.5)
        x_cat_test = x_cat_test[:, column_mask]

        # Process NaN, remove rows in training or set to 0 in test
        y_cat_train, x_cat_train, ids_cat_train, column_mask = filter_nan(y_cat_train, x_cat_train, ids_cat_train)
        y_cat_test, x_cat_test, ids_cat_test = filter_nan(y_cat_test, x_cat_test, ids_cat_test, remove=False)
        x_cat_test = x_cat_test[:, column_mask]  # If 0 filled columns were removed in training, remove the same in test

        # Remove outliers
        y_cat_train, x_cat_train, ids_cat_train = remove_outliers(y_cat_train, x_cat_train, ids_cat_train)

        # Scale training set and test set (using training set statistics)
        x_cat_train, x_cat_test = scale(x_cat_train, method="standard", x_test=x_cat_test)

        # Add bias
        x_cat_train = np.c_[np.ones(len(x_cat_train)), x_cat_train]
        x_cat_test = np.c_[np.ones(len(x_cat_test)), x_cat_test]

        # Assign new values
        Y_train[i], X_train[i], IDS_train[i] = y_cat_train, x_cat_train, ids_cat_train
        Y_test[i], X_test[i], IDS_test[i] = y_cat_test, x_cat_test, ids_cat_test
    
    return Y_train, X_train, IDS_train, Y_test, X_test, IDS_test
