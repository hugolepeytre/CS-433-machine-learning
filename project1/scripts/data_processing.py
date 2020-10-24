import numpy as np

def set_nan(tX):
    """
    Label the -999 values as NaN
    """
    tX_copy = np.copy(tX)
    tX_copy[tX_copy == -999] = np.nan
    return tX_copy


def remove_empty_columns(tX, threshold=0.4):
    """
    Remove feature columns containing more than the specified threshold proportion of NaN
    """
    tX_copy = np.copy(tX)
    # For each column compute the ratio of nan values over the number of rows
    prop_empty_column = (np.isnan(tX_copy)).sum(axis=0) / len(tX_copy)
    column_mask = prop_empty_column < threshold
    return tX_copy[:, column_mask], column_mask


def copy_data(y, tX, ids):
    return np.copy(y), np.copy(tX) , np.copy(ids)


def filter_nan(y, tX, ids, remove=True, replace_val=0):
    """
    Filter the nan (-999) values, by either removing the rows or replacing by the specified replace_val.
    Remove as well 0 filled columns when remove is set to True
    """   
    y_copy, tX_copy, ids_copy = copy_data(y, tX, ids)
    mask = np.isnan(tX_copy)  # True if Nan, False otherwise
    
    if remove:
        # Remove the rows containing any NaN
        row_mask = ~mask.any(axis=1) # sets to False any rows containing NaN
        tX_copy, y_copy, ids_copy = tX_copy[row_mask], y_copy[row_mask], ids_copy[row_mask]
        # Remove 0 filled columns
        col_mask = tX_copy.sum(axis=0) != 0 # True if the columns is filled with 0
        tX_copy = tX_copy[:, col_mask] 
        return y_copy, tX_copy, ids_copy, col_mask
    else:
        # Replace NaN values by replace_val
        tX_copy[mask] = replace_val
        return y_copy, tX_copy, ids_copy
    


def remove_outliers(y, tX, ids):
    """
    Remove outliers feature points using Interquartile range.
    """

    y_copy, tX_copy, ids_copy = copy_data(y, tX, ids)
    # Compute first and third quartiles and the Interquartile range
    Q1 = np.percentile(tX_copy, 25, axis=0) 
    Q3 = np.percentile(tX_copy, 75, axis=0)
    IQR = Q3 - Q1
    # Set to True any entry outside the interquartile range
    mask = (tX_copy >= Q1 - 1.5 * IQR) & (tX_copy <= Q3 + 1.5 * IQR)

    # Only filter out features with values that are spread over a range bigger than threshold_range
    # i.e. if the difference between the minimum value and the maximum value is bigger than threshold_range
    threshold_range = 10
    # Set to False any feature with range bigger than threshold
    col_mask = (tX.max(axis=0) - tX.min(axis=0)) < threshold_range
    mask = mask | col_mask
    row_mask = mask.all(axis=1)  # sets to False rows containing any outliers

    return y_copy[row_mask], tX_copy[row_mask], ids_copy[row_mask]

def scale(tX, method="standard", tX_test=None):
    """
    Scale features using the specified method. Possible methods: standard, min-max
    """
    tX_copy = np.copy(tX)
    tX_test_copy = None
    if tX_test is not None:
        tX_test_copy = np.copy(tX_test)

    if method == "standard": 
        # Standardize the data
        mean = tX.mean(axis=0)
        std = tX.std(axis=0)
        std[std == 0] = 1 # No division by 0
        tX_copy = (tX - mean) / std
        if tX_test is not None:
            tX_test_copy = (tX_test - mean) / std

    else:
        # Apply a min-max normalization to scale data between 0 and 1
        col_min = tX.min(axis=0)
        col_max = tX.max(axis=0)
        col_range = col_max - col_min
        tX_copy = (tX - col_min) / col_range
        if tX_test is not None:
            tX_test_copy = (tX_test - col_min) / col_range
   
    return tX_copy, tX_test_copy


def remove_correlated_features(tX, threshold=0.9):
    """
    Compute the correlations between each feature and remove features that have a correlation greater
    than the specified threshold
    """
    tX_copy = np.copy(tX)
    
    corr_matrix = np.corrcoef(tX_copy, rowvar=False)
    # Set to False highly correlated columns
    nb_col = len(corr_matrix)
    columns = np.full((nb_col,), True, dtype=bool)
    for i in range(nb_col):
        for j in range(i+1, nb_col):
            if corr_matrix[i, j] >= threshold:
                if columns[i]:
                    columns[j] = False
    
    # Remove correlated features and concat categorical features
    return tX_copy[:, columns], columns


def clean_training(y, tX, ids):
    tX = set_nan(tX)
    tX = remove_empty_columns(tX, threshold=2)  # Temporary to 2 to not remove any column
    y, tX, ids = filter_nan(y, tX, ids, remove=False, replace_val=0.0)
    y, tX, ids = remove_outliers(y, tX, ids)
    tX = scale(tX, method="standard")[0]
    y, tX, ids = remove_outliers(y, tX, ids)
    tX = np.c_[np.ones(len(tX)), tX]
    # tX = remove_correlated_features(tX, handpicked=29, threshold=0.9)
    return y, tX, ids


def clean_test(y, tX, ids, tX_train):
    tX_nan = set_nan(tX)
    tX_columns = remove_empty_columns(tX_nan, threshold=2)  # Temporary to 2 to not remove any column
    y, tX, ids = filter_nan(y, tX_columns, ids, remove=False, replace_val=0.0)
    tX = scale(tX_train[:,1:], method="standard", tX_test=tX)[1] # Scale test set with train set statistics
    # tX = remove_correlated_features(tX, handpicked=29, threshold=0.9)
    tX = np.c_[np.ones(len(tX)), tX]
    return y, tX, ids

def group_by_cat(y, tX, ids):
    """
    Return subsets (4 in fact) of the datasets, grouped by the category (PRI_jet_num feature).
    One subset will group features with category 0, another one 1 etc.
    """
    NB_CATEGORY = 4 
    #Column index of the categorical feature
    IDX_COL_CAT = np.where((tX.max(axis=0) - tX.min(axis=0))  == 3)[0][0]
    Y = []
    X = []
    IDS = []
    for i in range(NB_CATEGORY):
        row_idx = np.where(tX[:,IDX_COL_CAT] == i)[0] #index of the rows in category i
        tX_cat = np.delete(tX[row_idx], IDX_COL_CAT, axis=1) #Remove category feature
        Y.append(y[row_idx])
        X.append(tX_cat) 
        IDS.append(ids[row_idx])
    return Y, X, IDS


def clean_by_cat(y_train, tX_train, ids_train, y_test, tX_test, ids_test):
    Y_train, X_train, IDS_train = group_by_cat(y_train, tX_train, ids_train)
    Y_test, X_test, IDS_test = group_by_cat(y_test, tX_test, ids_test)

    for i in range(len(Y_train)):
        # Get the right category
        y_cat_train, tX_cat_train, ids_cat_train = Y_train[i], X_train[i], IDS_train[i]
        y_cat_test, tX_cat_test, ids_cat_test = Y_test[i], X_test[i], IDS_test[i]

        #Set -999 to NaN
        tX_cat_train = set_nan(tX_cat_train)
        tX_cat_test = set_nan(tX_cat_test)

        #Remove the (same) empty columns in the training and test
        tX_cat_train, column_mask = remove_empty_columns(tX_cat_train, threshold=0.5)
        tX_cat_test = tX_cat_test[:, column_mask]

        #Process NaN, remove rows in training or set to 0 in test
        y_cat_train, tX_cat_train, ids_cat_train, column_mask = filter_nan(y_cat_train, tX_cat_train, ids_cat_train)
        y_cat_test, tX_cat_test, ids_cat_test = filter_nan(y_cat_test, tX_cat_test, ids_cat_test, remove=False)
        tX_cat_test = tX_cat_test[:,column_mask] #If 0 filled columns were removed in training, remove the same in test

        #Remove outliers
        y_cat_train, tX_cat_train, ids_cat_train = remove_outliers(y_cat_train, tX_cat_train, ids_cat_train)

        #Scale training set and test set (using training set statistics)
        tX_cat_train, tX_cat_test = scale(tX_cat_train, method="standard", tX_test=tX_cat_test)

        # Add bias
        tX_cat_train = np.c_[np.ones(len(tX_cat_train)), tX_cat_train]
        tX_cat_test = np.c_[np.ones(len(tX_cat_test)), tX_cat_test]

        #Assign new values
        Y_train[i], X_train[i], IDS_train[i] = y_cat_train, tX_cat_train, ids_cat_train
        Y_test[i], X_test[i], IDS_test[i] = y_cat_test, tX_cat_test, ids_cat_test
    
    return Y_train, X_train, IDS_train, Y_test, X_test, IDS_test