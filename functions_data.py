import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import configparser
from scipy.stats import linregress

#=============================================================================
# Data functions
#=============================================================================

def get_config(config_file_name):
    # Load configuration
    config = configparser.ConfigParser()
    config.read(config_file_name)
    NB_RUNS = config['MODEL'].getint('NB_RUNS')

    # Parse arguments
    args = {
        'DATA_PATH': config['DATA'].get('DATA_PATH'),
        'PATH': config['DATA'].get('PATH'),
        'INPUT_WIDTH': config['WINDOW'].getint('INPUT_WIDTH'),
        'OUT_STEPS': config['WINDOW'].getint('OUT_STEPS'),
        'SHIFT': config['WINDOW'].getint('SHIFT'),
        'MAX_EPOCHS': config['MODEL'].getint('MAX_EPOCHS'),
        'BATCH_SIZE': config['MODEL'].getint('BATCH_SIZE'),

        'huber_delta': config['MODEL'].getfloat('huber_delta'),
        'LR_FACTOR': config['MODEL'].getfloat('LR_FACTOR'),
        'LR_Linear': config['MODEL'].getfloat('LR_Linear'),
        'LR_Dense': config['MODEL'].getfloat('LR_Dense'),
        'LR_Conv': config['MODEL'].getfloat('LR_Conv'),
        'LR_LSTM': config['MODEL'].getfloat('LR_LSTM'),
        'LR_Transformer': config['MODEL'].getfloat('LR_Transformer'),

        'DENSE_UNITS': config['MODEL'].getint('DENSE_UNITS'),
        'CONV_UNITS': config['MODEL'].getint('CONV_UNITS'),
        'CONV_SIZE': config['MODEL'].getint('CONV_SIZE'),
        'LSTM_UNITS': config['MODEL'].getint('LSTM_UNITS'),

        'head_size': config['MODEL'].getint('head_size'),
        'num_heads': config['MODEL'].getint('num_heads'),
        'ff_dim': config['MODEL'].getint('ff_dim'),
        'num_transformer_blocks': config['MODEL'].getint('num_transformer_blocks'),
        'mlp_units': config['MODEL'].get('mlp_units'),
        'mlp_dropout': config['MODEL'].getfloat('mlp_dropout'),
        'norm_type': config['MODEL'].get('norm_type'),

        'PATIENCE': config['MODEL'].getint('PATIENCE'),
        'SEED': config['MODEL'].getint('SEED'),
        'SAVE_REPARTITION': config['METADATA'].getboolean('SAVE_REPARTITION'),
        'SAVE_SHAPE': config['METADATA'].getboolean('SAVE_SHAPE'),
        'SAVE_NAME_FEATURES': config['METADATA'].getboolean('SAVE_NAME_FEATURES'),
        'SAVE_PY_COPY': config['METADATA'].getboolean('SAVE_PY_COPY'),
        'SAVE_REPARTITION_IMG': config['METADATA'].getboolean('SAVE_REPARTITION_IMG'),
        'SAVE_HISTORY_IMG': config['METADATA'].getboolean('SAVE_HISTORY_IMG')
    }

    return args, NB_RUNS

def make_gradients(dataf, windows, target, adding=''):
    """
    Compute rolling gradients for one column of a DataFrame.

    Parameters
    ----------
    dataf : pd.DataFrame
        Data on which rolling gradients will be computed.
    windows : list[int]
        Window lengths.
    target : str
        Column name on which to compute rolling gradients.
    adding : str, optional
        Text to add to created columns.

    Returns
    -------
    None (modifies dataf in place)
    """
    for w in windows:
        key = f"grad_{w}" + (f"_{adding}" if adding else "")
        dataf[key] = compute_grad(dataf[target].values.astype(float), w)
        
def compute_grad(y, window):
    """
    Compute the slope of a rolling linear regression over a given window.

    Parameters
    ----------
    y : np.ndarray (1D)
        Vector whose gradient is to be calculated.
    window : int
        Length of the moving window on which we compute the gradient.

    Returns
    -------
    gradient : np.ndarray (1D)
        Rolling slope of linear regression. First (window-1) values are 0.
    """
    n = len(y)
    gradient = np.zeros(n)
    x = np.arange(window)

    for i in range(window-1, n):
        y_window = y[i-window+1:i+1]
        slope, _, _, _, _ = linregress(x, y_window)
        gradient[i] = slope

    return gradient

def load_and_preprocess_data(data_path, type_data='gnss'):
    """
    Loads data and performs initial preprocessing for gnss data.

    Parameters
    ----------
    data_path : str or Path
        Path to the xlsx file with gnss data

    type_data : str
        Type of data to load, either 'gnss' or 'seismicity'.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with features.
    date_time : numpy.ndarray
        1 d time vector.

    """
    if type_data == 'gnss':
        df = pd.read_excel(data_path, sheet_name="gnss")
    elif type_data == 'seismicity':
        df = pd.read_excel(data_path, sheet_name="seismicity")
    elif type_data == 'both':
        df = pd.read_excel(data_path, sheet_name="both")

    df = df.dropna().reset_index(drop=True)
    date_time = df.pop('date').to_numpy().astype('datetime64[h]')
    cutoff = np.where(date_time > np.array('2007-12-31T12',
                                        dtype='datetime64[h]'))[0][0]

    df = df.loc[cutoff:].reset_index(drop=True)
    date_time = date_time[cutoff:]

    # Ensure columns are numeric
    df = df.apply(pd.to_numeric, errors='coerce').astype(float) 

    # Handle seismicity column if present
    if 'seismicity' in df.columns:
        seismicity_log10 = np.log10(df['seismicity'].replace(0, np.nan))
        seismicity_log10 = seismicity_log10.fillna(0)  # replace -inf with 0
        df['seismicity'] = seismicity_log10

    # Generate gradient features
    drop_cols = []
    for col in df.columns:
        make_gradients(df, [5, 10, 30], col, col)
        if col != 'seismicity':
            drop_cols.append(col)

    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df, date_time

class AsinhAfterZScaler:
    """
    Pipeline:
      1) z-score (mean/std from train)
      2) asinh(x / s) with robust per-feature s
      3) optional re-normalization to unit variance after asinh
    Fully invertible.
    """

    def __init__(self, alpha=1.0, s_mode="mad", renorm_after=True, eps=1e-9):
        """
        alpha: global multiplier for s (try 0.5, 1.0, 2.0)
        s_mode: 'mad' (robust), 'std', or 'p99' (99th abs percentile / 2.326 ~ sigma)
        renorm_after: if True, z-score after asinh
        """
        self.alpha = float(alpha)
        self.s_mode = s_mode
        self.renorm_after = renorm_after
        self.eps = eps

        # learned params
        self.mean1_ = None
        self.std1_  = None
        self.s_     = None
        self.mean2_ = None
        self.std2_  = None

    @staticmethod
    def _mad_sigma(x):
        med = np.median(x, axis=0)
        mad = np.median(np.abs(x - med), axis=0)
        return mad / 0.6745

    @staticmethod
    def _p99_sigma(x):
        # robust scale from 99th percentile of |x|; divide by 2.326 (z at 99%)
        q = np.percentile(np.abs(x), 99, axis=0)
        return q / 2.326

    def fit(self, df: pd.DataFrame):
        X = df.values

        # 1) first z-score params
        self.mean1_ = pd.Series(X.mean(axis=0), index=df.columns)
        self.median1_ = pd.Series(np.median(X, axis=0), index=df.columns)
        self.std1_  = pd.Series(X.std(axis=0), index=df.columns).replace(0, self.eps)

        # standardize train using median and std (not mean)
        X1 = (X - self.median1_.values)  / self.std1_.values

        # 2) compute per-feature s on standardized data
        if self.s_mode == "mad":
            sigma = self._mad_sigma(X1)
        elif self.s_mode == "std":
            sigma = X1.std(axis=0)
        elif self.s_mode == "p99":
            sigma = self._p99_sigma(X1)
        else:
            raise ValueError("s_mode must be 'mad', 'std', or 'p99'.")

        sigma = np.where(np.isfinite(sigma), sigma, 1.0)
        sigma = np.maximum(sigma, self.eps)
        self.s_ = pd.Series(self.alpha * sigma, index=df.columns)

        # 3) compute post-asinh z-score params (optional)
        Z = np.arcsinh(X1 / self.s_.values)
        if self.renorm_after:
            self.mean2_ = pd.Series(Z.mean(axis=0), index=df.columns)
            self.std2_  = pd.Series(Z.std(axis=0), index=df.columns).replace(0, self.eps)
        else:
            self.mean2_ = pd.Series(np.zeros(Z.shape[1]), index=df.columns)
            self.std2_  = pd.Series(np.ones(Z.shape[1]), index=df.columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.values
        X1 = (X - self.mean1_.values) / self.std1_.values
        Z  = np.arcsinh(X1 / self.s_.values)
        Z2 = (Z - self.mean2_.values) / self.std2_.values
        return pd.DataFrame(Z2, index=df.index, columns=df.columns)

    def inverse_transform(self, data):
        """
        Inverse transform.
        If array_mode=True, expects numpy array of shape (..., n_features)
        Supports 2D or 3D numpy arrays.
        """
        if isinstance(data, pd.DataFrame):
            # assume DataFrame
            Z2 = data.values
            Z  = Z2 * self.std2_.values + self.mean2_.values
            X1 = np.sinh(Z) * self.s_.values
            X  = X1 * self.std1_.values + self.mean1_.values
            return pd.DataFrame(X, index=data.index, columns=data.columns)

        else:
            Z2 = np.asarray(data)
            orig_shape = Z2.shape
            if Z2.ndim == 3:
                Z2 = Z2.reshape(-1, orig_shape[-1])

            Z  = Z2 * self.std2_.values + self.mean2_.values
            X1 = np.sinh(Z) * self.s_.values
            X  = X1 * self.std1_.values + self.mean1_.values

            if len(orig_shape) == 3:
                X = X.reshape(orig_shape)
            return X

class SimpleScaler:
    def __init__(self, eps=1e-9):
        self.eps = eps
        self.median_ = None
        self.mean_ = None
        self.std_ = None

    def fit(self, df: pd.DataFrame):
        self.mean_ = df.mean()
        self.median_ = df.median()
        self.std_ = df.std().replace(0, self.eps)
        return self

    def transform(self, data):
        if isinstance(data, pd.DataFrame):
            return (data - self.median_) / self.std_
        else:
            X = np.asarray(data)
            orig_shape = X.shape
            if X.ndim == 3:
                X = X.reshape(-1, orig_shape[-1])
            X = (X - self.median_.values) / self.std_.values
            if len(orig_shape) == 3:
                X = X.reshape(orig_shape)
            return X

    def inverse_transform(self, data):
        if isinstance(data, pd.DataFrame):
            return data * self.std_ + self.median_
        else:
            X = np.asarray(data)
            orig_shape = X.shape
            if X.ndim == 3:
                X = X.reshape(-1, orig_shape[-1])
            X = (X * self.std_.values) + self.median_.values
            if len(orig_shape) == 3:
                X = X.reshape(orig_shape)
            return X


def standardize_data(train_df, valid_df, test_df,
                     use_gnss_asinh=False, alpha=1.0, s_mode="mad", renorm_after=True):
    if use_gnss_asinh:
        scaler = AsinhAfterZScaler(alpha=alpha, s_mode=s_mode, renorm_after=renorm_after)
    else:
        # simple z-score fallback
        scaler = SimpleScaler()

    scaler.fit(train_df)
    return (scaler.transform(train_df),
            scaler.transform(valid_df),
            scaler.transform(test_df),
            scaler)

def create_dataset(df, input_width, out_steps, dates=None, shuffle=False, threshold=None):
    """
    Creates a time series dataset for training, validation, or testing with
    optional filtering based on a spike threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        The time series data to be used for creating the dataset.
    input_width : int
        Number of input time steps in each sequence.
    out_steps : int
        Number of output time steps to predict.
    dates : pandas.Series or numpy.ndarray, optional
        Datetime vector aligned with df. If provided, will return the date
        corresponding to the last day of each input window (x_dates)
        and the first day of each target window (y_dates).
    shuffle : bool
        Whether to shuffle the dataset.
    threshold : float, optional
        A threshold value to filter out sequences that contain spikes.

    Returns
    -------
    X : numpy.ndarray
        Input sequences with shape (filtered_num_sequences, input_width, num_features).
    y : numpy.ndarray
        Target sequences with shape (filtered_num_sequences, out_steps, num_features).
    x_dates : numpy.ndarray, optional
        Array of datetimes for the last input timestamp of each X sequence.
    y_dates : numpy.ndarray, optional
        Array of datetimes for the first output timestamp of each y sequence.
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

    # Dates
    x_dates, y_dates = None, None
    if dates is not None:
        dates = np.asarray(dates)
        x_dates = dates[id_indices.flatten() + input_width - 1]   # last of X
        y_dates = dates[id_indices.flatten() + input_width]       # first of y

    # Apply threshold filtering if specified
    if threshold is not None:
        mask = np.any(np.abs(X) > threshold, axis=(1, 2))  # Find sequences containing spikes above the threshold
        X = X[mask]
        y = y[mask]
        if x_dates is not None:
            x_dates = x_dates[mask]
            y_dates = y_dates[mask]

    if dates is not None:
        return X, y, x_dates, y_dates
    else:
        return X, y


def split_data(df, date_time, output_path, generate_figure=True, type_data="gnss"):
    """
    Splits the data into training, validation, and test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with the features.
    date_time : numpy.ndarray
         1d vector of 'datetime64[h]'. Time of recordings of the reccords.
    output_path : pathlib.Path
        Where to save the the figure.
    generate_figure : bool, optional
        If the repartition figure is plot. The default is True.
    type_data : str
        "gnss", "seismicity" or "both". Default = "gnss".

    Returns
    -------
    train_df : pandas.DataFrame
        Training dataset with the features.
    valid_df : pandas.DataFrame
        Validation dataset with the features.
    test_df : pandas.DataFrame
        Testing dataset with the features.
    split_infos : dict
        Dictionary with informations on how the dataset is splitted.

    """
    split_infos = {}

    valid_fraction, test_fraction = 0.15, 0.15
    train_fraction = 1 - valid_fraction - test_fraction
    nb_data = len(df)

    cut_train = int(train_fraction * nb_data)
    cut_valid = int((train_fraction + valid_fraction) * nb_data)

    train_df = df[:cut_train]
    valid_df = df[cut_train:cut_valid]
    test_df  = df[cut_valid:]

    split_infos['index_train'] = str((0, cut_train))
    split_infos['index_valid'] = str((cut_train, cut_valid))
    split_infos['index_test'] = str((cut_valid, len(df)))
    split_infos['train_times'] = str((str(date_time[0]),
                                      str(date_time[cut_train])))

    split_infos['valid_times'] = str((str(date_time[cut_train]),
                                      str(date_time[cut_valid])))

    split_infos['test_times'] = str((str(date_time[cut_valid]),
                                     str(date_time[len(df)-1])))

    # MAKE THE FIGURE
    if generate_figure:
        serie_to_plot = df.to_numpy()[:,0]

        plt.figure(figsize=(22, 8))
        ax1 = plt.subplot()
        plt.grid(True)
        plt.fill_between(date_time[:cut_train], -1e9, 1e9,
                         alpha=0.2, color='steelblue', label='train')
    
        plt.fill_between(date_time[cut_train:cut_valid], -1e9, 1e9,
                         alpha=0.2, color='red', label='valid')
    
        plt.fill_between(date_time[cut_valid:], -1e9, 1e9,
                         alpha=0.2, color='orange', label='test')
    
        plt.xlim(date_time[0], date_time[-1])
    
        ax1.plot(date_time, serie_to_plot, 'k', lw=1)
        plt.ylim(np.min(serie_to_plot)-0.1*np.min(serie_to_plot),
                 np.max(serie_to_plot)+0.1*np.max(serie_to_plot))
    
        if type_data == "gnss":
            ax1.set_ylabel("GNSS gradient baseline", fontsize=16)
        elif type_data == "seismicity":
            ax1.set_ylabel("VT events per day", fontsize=16)
        elif type_data == "both":
            ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
            ax2.plot(date_time, df.to_numpy()[:,-1])

            ax2.set_ylabel("VT events per day", fontsize=16)
            ax1.set_ylabel("GNSS gradient baseline", fontsize=16)

        plt.xlabel('Time', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(title='Datasets', fontsize=16, title_fontsize=16)
        plt.savefig(output_path / 'sets_repartition.png', bbox_inches='tight')

        plt.close()
        plt.show()

    return train_df, valid_df, test_df, split_infos

def saving_infos(args, datasets, to_in_models, split_infos, output_path):
    """
    Saves metadata information about the dataset splits, feature names, and
    shapes to a CSV file.

    Parameters
    ----------
    args : dict
        Configuration dictionary containing flags to determine which metadata
        to save. Keys include:
        - 'SAVE_REPARTITION': bool, whether to save dataset split information.
        - 'SAVE_SHAPE': bool, whether to save the shapes of the datasets.
        - 'SAVE_NAME_FEATURES': bool, whether to save the feature names.
    datasets : tuple
        Contains the datasets in the following order:
        (X_train, y_train, X_valid, y_valid, X_test, y_test).
        Each element is a numpy.ndarray or compatible object.
    to_in_models : list of str
        List of feature names to be saved if 'SAVE_NAME_FEATURES' is enabled.
    split_infos : dict
        Dictionary containing information about dataset splits (e.g.,
        train/validation/test ratios).
    output_path : pathlib.Path
        Directory where the metadata CSV file will be saved.

    Returns
    -------
    None
        Saves a CSV file named "metadata.csv" in the specified `output_path`.

    """
    df_metadata = pd.DataFrame()
    if args['SAVE_REPARTITION']:
        for key in split_infos:
            df_metadata[key] = [split_infos[key]]

    if args['SAVE_SHAPE']:
        df_metadata['X_train_shape'] = [str(datasets[0].shape)]
        df_metadata['y_train_shape'] = [str(datasets[1].shape)]
        df_metadata['X_valid_shape'] = [str(datasets[2].shape)]
        df_metadata['y_valid_shape'] = [str(datasets[3].shape)]
        df_metadata['X_test_shape'] = [str(datasets[4].shape)]
        df_metadata['y_test_shape'] = [str(datasets[5].shape)]

    if args['SAVE_NAME_FEATURES']:
        for i, key in enumerate(to_in_models):
            df_metadata['FEATURE_'+str(i)] = [key]

    df_metadata = df_metadata.transpose()
    df_metadata.to_csv(output_path / "metadata.csv")