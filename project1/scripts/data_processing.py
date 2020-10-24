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
    return tX_copy[:, column_mask]


def copy_data(y, tX, ids):
    return np.copy(y), np.copy(tX) , np.copy(ids)


def filter_nan(y, tX, ids, remove=True, replace_val=0):
    """
    Filter the nan (-999) values, by either removing the rows or replacing by the specified replace_val.
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


def noncategorical_columns(tX):
    """
    Computes the columns with more that 10 unique values
    """
    # count the number of unique values
    nunique_col = (np.diff(np.sort(tX, axis=0), axis=0) != 0).sum(axis=0) + 1 
    noncategorical_col = nunique_col > 10  # set to True columns with more than 10 unique elements
    return noncategorical_col


def scale(tX, method="standard", tX_test=None):
    """
    Scale noncategorical features using the specified method. Possible methods: standard, min-max
    """
    tX_copy = np.copy(tX)
    noncategorical_col = noncategorical_columns(tX_copy)
    tX_noncat = tX_copy[:, noncategorical_col]
    
    tX_test_copy = None
    if tX_test is not None:
        tX_test_copy = np.copy(tX_test)
        noncategorical_col_test = noncategorical_columns(tX_test_copy)

    if method == "standard": 
        # Standardize the data
        tX_copy[:, noncategorical_col] = (tX_noncat - tX_noncat.mean(axis=0)) / tX_noncat.std(axis=0)
        if tX_test is not None:
            tX_test_copy[:, noncategorical_col_test] = (tX_noncat - tX_noncat.mean(axis=0)) / tX_noncat.std(axis=0)

    else:
        # Apply a min-max normalization to scale data between 0 and 1
        col_min = tX_noncat.min(axis=0)
        col_max = tX_noncat.max(axis=0)
        tX_copy[:, noncategorical_col] = (tX_noncat - col_min) / (col_max - col_min)
        if tX_test is not None:
            tX_test_copy[:, noncategorical_col_test] = (tX_noncat - tX_noncat.mean(axis=0)) / tX_noncat.std(axis=0)
   
    return tX_copy, tX_test_copy


def remove_correlated_features(tX, handpicked=-1, threshold=0.9):
    """
    Compute the correlations between each feature and remove features that have a correlation greater
    than the specified threshold
    """
    if handpicked != -1:
        columns = np.full((tX.shape[1],), True, dtype=bool)
        columns[handpicked] = False
        return tX[:, columns]

    tX_copy = np.copy(tX)
    noncategorical_col = noncategorical_columns(tX_copy)
    cat_idx = np.where(~noncategorical_col)[0]  # Index of non categorical features
    tX_noncat = tX_copy[:, noncategorical_col]
    
    corr_matrix = np.corrcoef(tX_noncat, rowvar=False)
    
    # Set to False highly correlated columns
    nb_col = len(corr_matrix)
    columns = np.full((nb_col,), True, dtype=bool)
    for i in range(nb_col):
        for j in range(i+1, nb_col):
            if corr_matrix[i, j] >= threshold:
                if columns[i]:
                    columns[j] = False
    
    # Remove correlated features and concat categorical features
    return np.c_[tX_noncat[:, columns], tX_copy[:, cat_idx]]


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


def clean_test(y, tX, ids):
    tX_nan = set_nan(tX)
    tX_columns = remove_empty_columns(tX_nan, threshold=2)  # Temporary to 2 to not remove any column
    y, tX, ids = filter_nan(y, tX_columns, ids, remove=False, replace_val=0.0)
    tX = scale(tX, method="standard", tX_test=tX)[1]
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
        tX_cat = np.c_[np.ones(len(tX_cat)), tX_cat] #Add bias
        Y.append(y[row_idx])
        X.append(tX_cat) 
        IDS.append(ids[row_idx])
    return Y, X, IDS

def clean_by_cat(y, tX, ids):
    Y, X, IDS = group_by_cat(y, tX, ids)

    for i in range(len(Y)):
        y_cat, tX_cat, ids_cat = Y[i], X[i], IDS[i]
        tX_cat = set_nan(tX_cat)
        tX_cat = remove_empty_columns(tX_cat, threshold=0.4)  #Remove any columns containing r
        y_cat, tX_cat, ids_cat = filter_nan(y_cat, tX_cat, ids_cat, remove=True) #Remove rows containing NaN
        y_cat, tX_cat, ids_cat = remove_outliers(y_cat, tX_cat, ids_cat)
        tX_cat = scale(tX_cat, method="standard")[0]
        # y, tX, ids = remove_outliers(y, tX, ids)
        Y[i], X[i], IDS[i] = y_cat, tX_cat, ids_cat
    return Y, X, IDS