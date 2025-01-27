import os
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import configparser
from pathlib import Path
import shutil
import matplotlib.pyplot as plt

# Check if GPU is available
print('Available GPUs:', tf.config.list_physical_devices('GPU'))

# To use Python `labmda` function in model layer
keras.config.enable_unsafe_deserialization()

# Set up the main function
def main(args):
    # Setup paths
    PATH = Path(args['PATH'])
    DATA_PATH = Path(args['DATA_PATH'])

    best_iteration, best_model, error_test = select_best_model(args, PATH)

    if args['COMPUTE_FI']:
        if args['DATASET'] == 'gnss':
            feature_names, X_test, y_test, test_df = get_gnss_set(args, DATA_PATH)

        elif args['DATASET'] == 'seismicity':
            feature_names, X_test, y_test, test_df = get_seismic_set(args, DATA_PATH)

        model = keras.models.load_model(
            PATH / best_iteration / best_model / 'best_model.keras')

        compute_feature_importance(args, model, feature_names, error_test,
                                   X_test, y_test, PATH, best_iteration,
                                   best_model)

    file_name = (args['DATASET']+'_'+'FI_analysis_seed_'+best_iteration+
                 '_model_'+best_model+'.csv')

    if args['PLOT_FI'] and (os.path.isfile(PATH / file_name)):
        if args['DATASET'] == 'gnss':
            fnames = np.array([
                '3 days gradient DSRG-SNEG', '10 days gradient DSRG-SNEG',
                '30 days gradient DSRG-SNEG', '3 days gradient BOMG-DSRG',
                '10 days gradient BOMG-DSRG', '30 days gradient BOMG-DSRG',
                '3 days gradient BOMG-BORG', '10 days gradient BOMG-BORG',
                '30 days gradient BOMG-BORG', '3 days gradient BORG-SNEG',
                '10 days gradient BORG-SNEG', '30 days gradient BORG-SNEG',
                '3 days gradient BOMG-DERG', '10 days gradient BOMG-DERG',
                '30 days gradient BOMG-DERG', '3 days gradient DERG-SNEG',
                '10 days gradient DERG-SNEG', '30 days gradient DERG-SNEG',
                '3 days gradient BORG-DSRG', '10 days gradient BORG-DSRG',
                '30 days gradient BORG-DSRG', '3 days gradient BORG-DERG',
                '10 days gradient BORG-DERG', '30 days gradient BORG-DERG',
                '3 days gradient DERG-DSRG', '10 days gradient DERG-DSRG',
                '30 days gradient DERG-DSRG', '3 days gradient BOMG-SNEG',
                '10 days gradient BOMG-SNEG', '30 days gradient BOMG-SNEG'])

            plot_feature_importance(args, PATH, file_name,
                                    fnames=fnames,
                                    figsize=(3.22, 6.44),
                                    dpi=200,
                                    limit_proportion=0.02,
                                    bar_width=10,
                                    delta_vert=0.2,
                                    width_vert=1.,
                                    xlims=None, ylims=None)

        elif args['DATASET'] == 'seismicity':
            fnames = np.array(['Seismicity', '6 hours\ngradient',
                               '12 hours\ngradient', '24 hours\ngradient'])

            plot_feature_importance(args, PATH, file_name,
                                    fnames=fnames,
                                    figsize=(3.22, 3.22),
                                    dpi=200,
                                    limit_proportion=0.1,
                                    bar_width=10,
                                    delta_vert=0.2,
                                    width_vert=1.,
                                    xlims=None, ylims=None)

    if args['MAKE_RESUME_TABLE'] and (os.path.isfile(PATH / file_name)):
        make_resume_table(PATH, file_name)

    file_name = (args['DATASET']+'_'+'FI_analysis_seed_'+best_iteration+
                 '_model_'+best_model+'_summary.csv')

    if args['SHOW_CORR_FI'] and (os.path.isfile(PATH / file_name)):
        show_corr_signals(args, PATH, DATA_PATH, file_name)


#=============================================================================
# Data functions
#=============================================================================

def make_gradients(dataf, windows, target, adding=''):
    """
    Function to compute the rolling gradients on a desired column of a
    dataframe.

    Parameters
    ----------
    dataf : pandas.DataFrame
        Pandas dataframe object with the data on whihch the rolling gradients
        will be computed and stored.
    windows : list like iterrable
        Length of the moving windows.
    target : str
        Column name on which you want to compute the rolling gradients.
    adding : str, optional
        Text to add to the created columns. The default is ''.

    Returns
    -------
    None

    """
    for i in range(len(windows)):
        if len(adding) > 0:
            key_i = 'grad_'+str(windows[i])+'_'+adding
        else:
            key_i = 'grad_'+str(windows[i])

        dataf[key_i] = compute_grad(dataf[target].values.astype(float),
                                    windows[i])

def compute_grad(y, window):
    """
    Function to compute the steep of the linear regression between t and
    t-window.

    Parameters
    ----------
    y : np.ndarray
        Vector 1d whose gradient is to be calculated.
    window : int.
        Length of the moving window on which we compute the gradient.

    Returns
    -------
    gradient : np.ndarray
        Vector 1d which is the computed gradient from y.

    """
    sh_y = y.shape
    gradient = np.zeros(sh_y[0])
    x_ = np.arange(0, window, 1, dtype=float)
    y_wind = y[np.arange(sh_y[0]-window+1) +
               np.arange(0, window, 1)[:, np.newaxis]]

    p = np.polyfit(x_, y_wind, 1)
    gradient[window-1:] = p[0]
    return gradient

def load_and_preprocess_gnss_data(data_path):
    """
    Loads data and performs initial preprocessing for gnss data.

    Parameters
    ----------
    data_path : str or Path
        Path to the xlsx file with gnss data

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with features.
    date_time : numpy.ndarray
        1 d time vector.

    """
    df = pd.read_excel(data_path, sheet_name="gnss")
    df = df.dropna().reset_index(drop=True)
    date_time = df.pop('date').to_numpy().astype('datetime64[h]')
    cutoff = np.where(date_time > np.array('2007-12-31T12',
                                        dtype='datetime64[h]'))[0][0]

    df = df.loc[cutoff:].reset_index(drop=True)
    date_time = date_time[cutoff:]

    # Ensure columns are numeric
    df = df.apply(pd.to_numeric, errors='coerce').astype(float) 

    # Generate gradient features
    for col in df.columns:
        make_gradients(df, [3, 10, 30], col, col)
        df = df.drop(columns=col)

    return df, date_time

def load_and_preprocess_seismicity_data(data_path):
    """
    Loads data and performs initial preprocessing for seismicity data.

    Parameters
    ----------
    data_path : str or Path
        Path to the xlsx file with seismicity data

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with features.
    date_time : numpy.ndarray
        1 d time vector.

    """
    df = pd.read_excel(data_path, sheet_name="seismicity")
    df = df.dropna().reset_index(drop=True)
    date_time = df.pop('date').to_numpy().astype('datetime64[h]')

    # Ensure columns are numeric
    df = df.apply(pd.to_numeric, errors='coerce').astype(float) 

    # Generate gradient features
    make_gradients(df, [6, 12, 24], 'seismicity')

    return df, date_time

def split_data(df, date_time):
    """
    Splits the data into training, validation, and test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with the features.
    date_time : numpy.ndarray
         1d vector of 'datetime64[h]'. Time of recordings of the reccords.

    Returns
    -------
    train_df : pandas.DataFrame
        Training dataset with the features.
    valid_df : pandas.DataFrame
        Validation dataset with the features.
    test_df : pandas.DataFrame
        Testing dataset with the features.

    """
    valid_fraction, test_fraction = 0.15, 0.15
    train_fraction = 1 - valid_fraction - test_fraction
    nb_data = len(df)

    cut_train = int(train_fraction * nb_data)
    cut_valid = int((train_fraction + valid_fraction) * nb_data)

    train_df = df[:cut_train]
    valid_df = df[cut_train:cut_valid]
    test_df  = df[cut_valid:]
    return train_df, valid_df, test_df

def standardize_gnss_data(train_df, valid_df, test_df):
    """
    Standardizes gnss datasets using train mean and std deviation.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Tarining dataset with the features.
    valid_df : pandas.DataFrame
        Validation dataset with the features.
    test_df : pandas.DataFrame
        Testing dataset with the features.

    Returns
    -------
    train_df : pandas.DataFrame
        Training dataset with the features after standardization.
    valid_df : pandas.DataFrame
        Validation dataset with the features after standardization.
    test_df : pandas.DataFrame
        Testing dataset with the features after standardization.
    train_mean : pandas.DataFrame
        Training average for standardiztion.
    train_std : pandas.DataFrame
        Training standard deviation for standardiztion.

    """
    train_mean, train_std = train_df.mean(), train_df.std()
    # we only use the standard deviation
    valid_df, test_df = valid_df / train_std, test_df / train_std
    train_df = train_df / train_std

    return train_df, valid_df, test_df, train_mean, train_std

def standardize_seismicity_data(train_df, valid_df, test_df):
    """
    Standardizes seismic datasets using train mean and std deviation.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataset with the features.
    valid_df : pandas.DataFrame
        Validation dataset with the features.
    test_df : pandas.DataFrame
        Testing dataset with the features.

    Returns
    -------
    train_df : pandas.DataFrame
        Training dataset with the features after standardization.
    valid_df : pandas.DataFrame
        Validation dataset with the features after standardization.
    test_df : pandas.DataFrame
        Testing dataset with the features after standardization.
    train_mean : pandas.DataFrame
        Training average for standardiztion.
    train_std : pandas.DataFrame
        Training standard deviation for standardiztion.

    """
    train_mean, train_std = train_df.mean(), train_df.std()

    valid_df = (valid_df - train_mean) / train_std
    test_df  = ( test_df - train_mean) / train_std
    train_df = (train_df - train_mean) / train_std

    return train_df, valid_df, test_df, train_mean, train_std

def create_dataset(df, input_width, out_steps, shuffle=False, threshold=None):
    """
    Creates time series dataset for training/validation/testing with optional
    filtering based on a spike threshold.

    Parameters
    ----------
    df : pandas DataFrame
        Time series data.
    input_width : int
        Number of input time steps in each sequence.
    out_steps : int
        Number of output time steps to predict.
    shuffle : bool
        If the sample are shuffled.
    threshold : float
        Optional threshold to filter sequences that contain spikes.

    Returns
    -------
    X : numpy.ndarray
        Shape (filtered_num_sequences, input_width), input sequences.
    y : numpy.ndarray
        Shape (filtered_num_sequences, out_steps), target sequences.

    """
    # Set appropriate dtype based on the length of df
    dtype = 'int32' if len(df) < 2147483646 else 'int64'
    data_type = 'float64'

    # Generate indices for input and target sequences
    id_indices = np.arange(0, len(df) - input_width - out_steps + 1, dtype=dtype)
    if shuffle:
        np.random.shuffle(id_indices)

    id_indices = id_indices[:, np.newaxis]
    id_X = id_indices + np.arange(0, input_width, dtype=dtype)
    id_y = id_indices + np.arange(input_width, input_width + out_steps, dtype=dtype)

    # Create input (X) and target (y) arrays
    X = df.values.astype(data_type)[id_X]
    y = df.values.astype(data_type)[id_y]

    # Apply threshold filtering if specified
    if threshold is not None:
        # Find sequences containing spikes above the threshold
        mask = np.any(np.abs(X) > threshold, axis=(1, 2))
        X = X[mask]
        y = y[mask]

    return X, y

def get_gnss_set(args, data_path):
    """
    Extract test gnss dataset.

    Parameters
    ----------
    args : dict
        Dictionary with init file informations.
    data_path : pathlib.Path
        Path to access to the data.

    Returns
    -------
    features_names : numpy.ndarray
        1d vector of the feature's name.
    X_test : numpy.ndarray
        3d tensor used as input for the test dataset.
    y_test : numpy.ndarray
        3d tensor used as output for the test dataset.
    test_df : pandas.DataFrame
        Testing dataset with the features after standardization.

    """
    # Load and preprocess data
    df, date_time = load_and_preprocess_gnss_data(data_path)
    feature_names = np.array(list(df.columns))
    to_in_models = feature_names.copy()
    to_out = feature_names.copy()

    # Split data into training, validation, and test sets
    train_df, valid_df, test_df = split_data(df, date_time)

    # Standardize data
    train_df, valid_df, test_df, train_mean, train_std = standardize_gnss_data(train_df, valid_df, test_df)

    # Prepare tet dataset
    X_test, y_test = create_dataset(test_df, args['INPUT_WIDTH'], args['OUT_STEPS'])

    return feature_names, X_test, y_test, test_df

def get_seismic_set(args, data_path):
    """
    Extract test seismic dataset.

    Parameters
    ----------
    args : dict
        Dictionary with init file informations.
    data_path : pathlib.Path
        Path to access to the data.

    Returns
    -------
    features_names : numpy.ndarray
        1d vector of the feature's name.
    X_test : numpy.ndarray
        3d tensor used as input for the test dataset.
    y_test : numpy.ndarray
        3d tensor used as output for the test dataset.
    test_df : pandas.DataFrame
        Testing dataset with the features after standardization.

    """
    # Load and preprocess data
    df, date_time = load_and_preprocess_seismicity_data(data_path)
    feature_names = np.array(list(df.columns))
    to_in_models = feature_names.copy()
    to_out = feature_names.copy()

    # Split data into training, validation, and test sets
    train_df, valid_df, test_df = split_data(df, date_time)

    # Standardize data
    (train_df, valid_df, test_df, train_mean,
     train_std) = standardize_seismicity_data(train_df, valid_df, test_df)

    # Prepare tet dataset
    X_test, y_test = create_dataset(test_df, args['INPUT_WIDTH'],
                                    args['OUT_STEPS'])

    return feature_names, X_test, y_test, test_df

#=============================================================================
# Feature importance functions
#=============================================================================

def select_best_model(args, path):
    """
    Function to found the model with lowest MSE on test dataset.

    Parameters
    ----------
    args : dict
        Dictionary with init file informations.
    path : pathlib.Path
        Path to access the summary .

    Returns
    -------
    best_iteration : str
        String index of the best initialization.
    best_model : str
        Name of the best model from the best initialization.
    error_test : float
        MSE error of the best model through all initialization on the test
        dataset.

    """
    Test_MSE = []
    for i in range(args['NB_RUNS']):
        df = pd.read_csv('./' / path / str(i) / 'performance.csv')
        Test_MSE.append(df['Test_MSE'])

    models = df.values[:, 0]
    Test_MSE = np.array(Test_MSE, dtype=float)
    error_test = np.min(Test_MSE)
    pmin_Test_MSE = np.argwhere(Test_MSE == error_test)[0]
    best_iteration = str(pmin_Test_MSE[0])
    best_model = models[pmin_Test_MSE[1]]
    return best_iteration, best_model, error_test

def estimation_rand(model, X, y, id_i, batch_len=None):
    """
    Function to estimate the error induced from the shuffling.

    Parameters
    ----------
    model : tensorflow.keras model
        Best model to analyze.
    X : numpy.ndarray
        Input of test dataset.
    y : numpy.ndarray
        Output of test dataset.
    id_i : int
        Index to shuffle.
    batch_len : int, optional
        Lenght of the batch to not overflow the memory.

    Returns
    -------
    err_perm : float
        MSE induced from the shuffle.

    """
    # copy the origin input data
    Xp = np.copy(X).astype('float32')
    # Xp shape is => (samples, days, channels)
    channel = np.copy(Xp[:, :, id_i])
    kern = np.argsort(np.random.rand(Xp.shape[0], Xp.shape[1]), axis=1)
    for i in range(Xp.shape[0]):
        Xp[i, :, id_i] = channel[i][kern[i]]

    # If we want to compute by bach to not overflow the memory
    if type(batch_len) == int:
        # initialisation
        err_perm = 0
        for i in range(0, len(Xp)+batch_len, batch_len):
            # compute MSE
            if y_train[i:i+len_batch].shape[0] > 0:
                err_perm += np.sum((y[i:i+batch_len] -
                                    model(Xp[i:i+batch_len]))**2)

        # get the average
        err_perm = err_perm/len(Xp)

    else:
        err_perm = np.mean((y - model(Xp))**2)

    return err_perm

def compute_feature_importance(args, model, feature_names, error_test, X, y,
                               path, best_iteration, best_model):
    """
    Function to compute error induced over multiple shuffles.

    Parameters
    ----------
    args : dict
        Dictionary with init file informations.
    model : tensorflow.keras model
        Best model to analyze.
    feature_names : numpy.ndarray
        1d vector of the feature's name.
    error_test : float
        MSE error of the best model through all initialization on the test
        dataset.
    X : numpy.ndarray
        Input of test dataset.
    y : numpy.ndarray
        Output of test dataset.
    path : pathlib.Path
        Where to save the analyze.
    best_iteration :str
        String index of the best initialization.
    best_model :str
        Name of the best model from the best initialization.
    
    Returns
    -------
    None

    """
    # pd.DataFrame file structure at the output for n shuffles and m features
    # +--------------+------------+ ... +------------+ ... +-------------+
    # |   Shuffles   | feature_0  |     | feature_p  |     |feature_(m-1)|
    # +--------------+------------+ ... +------------+ ... +-------------+
    # | shuffles_-1  | unshuffled |     | unshuffled |     | unshuffled  |
    # |  shuffles_0  |  shuffled  |     |  shuffled  |     |  shuffled   |
    # .....
    # |  shuffles_i  |  shuffled  |     |  shuffled  |     |  shuffled   |
    # .....
    # |shuffles_(n-1)|  shuffled  |     |  shuffled  |     |  shuffled   |
    # +--------------+------------+ ... +------------+ ... +-------------+

    num_features = len(feature_names)
    file = {}
    file['Shuffles'] = 'shuffles_'+np.arange(-1,
                            args['NB_SAMPLING']).astype(str).astype('O')

    for p in range(num_features):
        print(feature_names[p])
        shuff_err = np.zeros(args['NB_SAMPLING']+1)
        shuff_err[0] = error_test
        for shuf in range(1, args['NB_SAMPLING']+1):
            shuff_err[shuf] = estimation_rand(model, X, y, p,
                                              batch_len=args['BATCH'])

        file[feature_names[p]] = np.copy(shuff_err)

    file = pd.DataFrame.from_dict(file)
    file_name = (args['DATASET']+'_'+'FI_analysis_seed_'+best_iteration+
                 '_model_'+best_model+'.csv')

    file.to_csv(path / file_name, index=False)

def make_resume_table(path, file_name):
    """
    Creates a summary table with median and standard deviation of the ratio
    between feature importance (shuffled values) and the test error (MSE).
    The summary is saved as a new CSV file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the directory containing the input file.
    file_name : str
        Save name of the CSV file containing feature importance of the test
        error.

    Returns
    -------
    None

    """
    fi_file = pd.read_csv(path / file_name)
    fnames = np.array(list(fi_file.columns)[1:])

    fi_shuffles = fi_file.values[1:, 1:].astype(float)
    error_test = fi_file.values[0, 1]

    ratio_MSE = fi_shuffles/error_test
    median_ratio_MSE = np.median(ratio_MSE, axis=0)
    std_ratio_MSE = np.std(ratio_MSE, axis=0)

    resume_file = {}
    for i, name in enumerate(fnames):
        resume_file[name+'_median'] = [median_ratio_MSE[i]]
        resume_file[name+'_deviation'] = [std_ratio_MSE[i]]

    resume_file = pd.DataFrame.from_dict(resume_file)
    save_name = file_name[:-4]+'_summary.csv'
    resume_file.to_csv(path / save_name, index=False)

def show_corr_signals(args, path, data_path, file_name):
    """
    Function to print correlation between inputs and features importance.

    Parameters
    ----------
    args : dict
        Dictionary with init file informations.
    path : pathlib.Path
        Path to the directory containing the features importance file.
    data_path : pathlib.Path
        Path to the dataset.
    file_name : str
        Save name of the CSV file containing feature importance of the test
        error.

    Returns
    -------
    None

    """
    df = pd.read_csv(path / file_name)
    if args['DATASET'] == 'gnss':
        feature_names, X_test, y_test, test_df = get_gnss_set(args, data_path)

    if args['DATASET'] == 'seismicity':
        feature_names, X_test, y_test, test_df = get_seismic_set(args,
                                                                 data_path)

    median = df.values[0, ::2]
    rank_fi = np.argsort(median)
    amplitudes = (test_df.max()-test_df.min()).to_numpy().astype(float)
    rank_signal = np.argsort(amplitudes)
    pearson_corr_values = np.corrcoef(median[rank_fi],
                                      amplitudes[rank_fi])[0, 1]

    pearson_corr_rank = np.corrcoef(rank_fi[rank_fi],
                                    rank_signal[rank_fi])[0, 1]

    print('')
    print('With '+args['DATASET']+' dataset:')
    print('Pearson correlation between features amplitudes and MSE ratio '\
          'median: '+str(round(pearson_corr_values, 4)))

    print('Pearson correlation between features amplitudes rank and MSE '\
          'ratio median rank: '+str(round(pearson_corr_rank, 4)))

    if args['DATASET'] == 'gnss':
        names = np.array(['DSRG_SNEG', 'BOMG_DSRG', 'BOMG_BORG', 'BORG_SNEG',
                          'BOMG_DERG', 'DERG_SNEG', 'BORG_DSRG', 'BORG_DERG',
                          'DERG_DSRG', 'BOMG_SNEG'])

        # 1 => baseline cross Dolomieu crater
        # 0 => baseline do not cross Dolomieu crater
        is_crossing = np.array([1, 1, 0, 1, 1, 0, 0, 1, 0, 0])
        shortp = feature_names[rank_fi]
        for i in range(30):
            shortp[i] = shortp[i][-16:-7]

        # Rank of the feature form MSE ratio median (baselines,
        # derived features)
        rank_ord = np.argwhere(shortp == names[:,
                               np.newaxis])[:, 1].reshape((10, 3))

        # baseline index of derived feature, sorted by MSE ratio median
        base_ord = np.argwhere(shortp[:, np.newaxis] == names)[:, 1]

        print('The average rank in feature importance for crossing baselines'\
              ': '+str(round(np.mean(rank_ord[is_crossing == 1]), 4)))

        print('The average rank in feature importance for non-crossing '\
              'baselines: '+str(round(np.mean(rank_ord[is_crossing == 0]),
                                      4)))

        print('The average of the MSE ratio median for crossing baselines'\
              ': '+str(round(np.mean(median[is_crossing[base_ord] == 1]), 4)))

        print('The average of the MSE ratio median for non-crossing '\
              'baselines: '+str(round(
                  np.mean(median[is_crossing[base_ord] == 0]), 4)))

    print('')

#=============================================================================
# Plot functions
#=============================================================================

def plot_feature_importance(args, path, file_name, fnames=None,
                            figsize=(3.22, 3.22), dpi=200,
                            limit_proportion=0.04, bar_width=1,
                            delta_vert=0.2, width_vert=0.05, xlims=None,
                            ylims=None):
    """
    Function to plot the median of the MSE ratio of the shuffle.

    Parameters
    ----------
    args : dict
        Dictionary with init file informations.
    path : pathlib.Path
        Where to save the figure.
    file_name : str
        Name of the file with the summary of the features importance analysis.
    fnames : list, optional
        Features name. The default is None.
    figsize : tuple, optional
        Size of the figure in inches. The default is (3.22, 3.22).
    dpi : int, optional
        Dots per inches. The default is 200.
    limit_proportion : float, optional
        Width proportion for x and y-axis. The default is 0.04.
    bar_width : float, optional
        Width of the bar. The default is 1.
    delta_vert : float, optional
        Vertical offset for the min-max amplitude of the distribution. The
        default is 0.2.
    width_vert : float, optional
        Width or the min-max amplitude of the distribution. The
        default is 0.05.
    xlims : tuple, optional
        Lower and upper limit of x-axis. The default is None.
    ylims : tuple, optional
        Lower and upper limit of y-axis. The default is None.

    Returns
    -------
    None

    """
    fi_file = pd.read_csv(path / file_name)
    if type(fnames) == type(None):
        fnames = np.array(list(fi_file.columns)[1:])

    fi_shuffles = fi_file.values[1:, 1:].astype(float)
    error_test = fi_file.values[0, 1]

    ratio_MSE = fi_shuffles/error_test
    median_ratio_MSE = np.median(ratio_MSE, axis=0)
    rank = np.argsort(median_ratio_MSE)

    positions = np.arange(len(rank))
    median_ratio_MSE = median_ratio_MSE[rank]
    if type(xlims) == type(None):
        delta = (np.max(ratio_MSE)-np.min(ratio_MSE))*limit_proportion
        xlims = (1, np.max(ratio_MSE)+delta)

    if type(ylims) == type(None):
        delta = np.max(positions)*limit_proportion
        ylims = (-delta, np.max(positions)+delta)

    fig_name = file_name[:-3]+'png'

    plt.figure(figsize=figsize, dpi=dpi)
    plt.grid(True, zorder=1)
    plt.hlines(positions, 0, median_ratio_MSE, lw=bar_width, color='g',
               zorder=5)

    plt.vlines(np.min(ratio_MSE, axis=0)[rank], positions-delta_vert,
               positions+delta_vert, lw=width_vert, color='darkorange',
               zorder=6)

    plt.vlines(np.max(ratio_MSE, axis=0)[rank], positions-delta_vert,
               positions+delta_vert, lw=width_vert, color='darkorange',
               zorder=6)

    plt.xlabel(r'$MSE_{permutation}/MSE_{origin}$', fontsize=10)
    plt.yticks(positions, fnames[rank], rotation=0, fontsize=9)

    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])

    plt.savefig(path / fig_name, bbox_inches='tight')

    plt.close()
    plt.show()
    

#=============================================================================
# main
#=============================================================================


if __name__ == "__main__":
    # Load configuration
    config = configparser.ConfigParser()
    config.read('feature_importance.ini')

    BATCH = config['MODEL'].get('BATCH')
    if BATCH == 'None':
        BATCH = None

    # Parse arguments
    args = {
        'DATASET': config['DATA'].get('DATASET'),
        'PATH': config['DATA'].get('PATH'),
        'DATA_PATH': config['DATA'].get('DATA_PATH'),
        'INPUT_WIDTH': config['WINDOW'].getint('INPUT_WIDTH'),
        'OUT_STEPS': config['WINDOW'].getint('OUT_STEPS'),
        'SHIFT': config['WINDOW'].getint('SHIFT'),
        'NB_RUNS': config['MODEL'].getint('NB_RUNS'),
        'COMPUTE_FI': config['MODEL'].getboolean('COMPUTE_FI'),
        'NB_SAMPLING': config['MODEL'].getint('NB_SAMPLING'),
        'BATCH': BATCH,
        'PLOT_FI': config['FIGURE'].getboolean('PLOT_FI'),
        'MAKE_RESUME_TABLE': config['ANALYSIS'].getboolean('MAKE_RESUME_TABLE'),
        'SHOW_CORR_FI': config['ANALYSIS'].getboolean('SHOW_CORR_FI')
    }

    main(args)
