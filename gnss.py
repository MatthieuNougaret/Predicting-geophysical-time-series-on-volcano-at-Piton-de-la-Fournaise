import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
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
    DATA_PATH = Path(args['DATA_PATH'])
    OUTPUT_PATH = Path(args['PATH']) / str(args['SEED'])
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Set parameters
    SEED = args['SEED']
    np.random.seed(SEED)
    tf.config.experimental.enable_op_determinism()
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    # Load and preprocess data
    df, date_time = load_and_preprocess_data(DATA_PATH)
    feature_names = np.array(list(df.columns))
    to_in_models = feature_names.copy()
    to_out = feature_names.copy()

    # Split data into training, validation, and test sets
    train_df, valid_df, test_df, split_infos = split_data(df, date_time, OUTPUT_PATH, args['SAVE_REPARTITION_IMG'])

    # Standardize data
    train_df, valid_df, test_df, train_mean, train_std = standardize_data(train_df, valid_df, test_df)

    # Prepare datasets
    X_train, y_train = create_dataset(train_df, args['INPUT_WIDTH'], args['OUT_STEPS'], shuffle=True)
    X_valid, y_valid = create_dataset(valid_df, args['INPUT_WIDTH'], args['OUT_STEPS'])
    X_test, y_test = create_dataset(test_df, args['INPUT_WIDTH'], args['OUT_STEPS'])

    # saving some hyperparameters
    saving_infos(args, [X_train, y_train, X_valid, y_valid, X_test, y_test],
                 to_in_models, split_infos, OUTPUT_PATH)

    # Compile and evaluate models
    performance = evaluate_models(
        X_train, y_train, X_valid, y_valid, X_test, y_test, args, to_out, OUTPUT_PATH)

    # Save performance metrics
    performance.to_csv(OUTPUT_PATH / "performance.csv", index=False)

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
    
def load_and_preprocess_data(data_path):
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

def split_data(df, date_time, output_path, generate_figure=True):
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
        plt.grid(True)
        plt.fill_between(date_time[:cut_train], -1e9, 1e9,
                         alpha=0.2, color='steelblue', label='train')
    
        plt.fill_between(date_time[cut_train:cut_valid], -1e9, 1e9,
                         alpha=0.2, color='red', label='valid')
    
        plt.fill_between(date_time[cut_valid:], -1e9, 1e9,
                         alpha=0.2, color='orange', label='test')
    
        plt.xlim(date_time[0], date_time[-1])
    
        plt.plot(date_time, serie_to_plot, 'k', lw=1)
        plt.ylim(np.min(serie_to_plot)-0.1*np.min(serie_to_plot),
                 np.max(serie_to_plot)+0.1*np.max(serie_to_plot))
    
        plt.ylabel("GNSS gradient baseline", fontsize=16)
        plt.xlabel('Time', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(title='Datasets', fontsize=16, title_fontsize=16)
        plt.savefig(output_path / 'sets_repartition.png', bbox_inches='tight')

        plt.close()
        plt.show()

    return train_df, valid_df, test_df, split_infos

def standardize_data(train_df, valid_df, test_df):
    """
    Standardizes gnss datasets using train standard deviation.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataset without standardization.

    Returns
    -------
    train_df : pandas.DataFrame
        Standardized training dataset.

    """
    train_mean, train_std = train_df.mean(), train_df.std()
    # we only use the standard deviation
    valid_df, test_df = valid_df / train_std, test_df / train_std
    train_df = train_df / train_std
    return train_df, valid_df, test_df, train_mean, train_std

def create_dataset(df, input_width, out_steps, shuffle=False, threshold=None):
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
    threshold : float, optional
        A threshold value to filter out sequences that contain spikes.

    Returns
    -------
    X : numpy.ndarray
        Input sequences with shape (filtered_num_sequences, input_width).
    y : numpy.ndarray
        Target sequences with shape (filtered_num_sequences, out_steps).

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
        mask = np.any(np.abs(X) > threshold, axis=(1, 2))  # Find sequences containing spikes above the threshold
        X = X[mask]
        y = y[mask]

    return X, y

#=============================================================================
# Model functions
#=============================================================================

def evaluate_models(X_train, y_train, X_valid, y_valid, X_test, y_test, args,
                    to_out, output_path):
    """
    Function to evaluates models and saves training history.

    Parameters
    ----------
    X_train : numpy.ndarray
        3d tensor of the training dataset input.
    y_train : numpy.ndarray
        3d tensor of the training dataset output.
    X_valid : numpy.ndarray
        3d tensor of the validation dataset input.
    y_valid : numpy.ndarray
        3d tensor of the validation dataset output.
    X_test : numpy.ndarray
        3d tensor of the testing dataset input.
    y_test : numpy.ndarray
        3d tensor of the testing dataset output.
    args : dict
        Dictionary with init file informations.
    to_out : numpy.ndarray
        1d vector with all the features.
    output_path : pathlib.Path
        Where to save the models.

    Returns
    -------
    performance_df : pandas.DataFrame
        Data Frame with the models score results.

    """
    num_features = len(to_out)
    performance_df = pd.DataFrame(['Last', 'Linear', 'Dense', 'Conv', 'LSTM',
                                   'Transformer'], columns=["Model"])

    performance_df['Valid_MSE'], performance_df['Valid_MAE'] = 0.0, 0.0
    performance_df['Test_MSE'], performance_df['Test_MAE'] = 0.0, 0.0
    performance_df['Valid_R2'], performance_df['Test_R2'] = 0.0, 0.0
    for i in range(num_features):
        performance_df['Valid_R2_'+to_out[i]] = 0.0
        performance_df['Test_R2_'+to_out[i]] = 0.0

    for i in range(args['OUT_STEPS']):
        performance_df['Valid_R2_t+'+str(i+1)] = 0.0
        performance_df['Test_R2_t+'+str(i+1)] = 0.0

    # Define models
    models = {
        'Last': Baseline(args['OUT_STEPS']),
        'Linear': build_linear_model(args, num_features),
        'Dense': build_dense_model(args, num_features),
        'Conv1D': build_conv_model(args, num_features),
        'LSTM': build_lstm_model(args, num_features),
        'Transformer': build_transformer_model((args['INPUT_WIDTH'], num_features), # Shape of input (timesteps, features),
                                               args['INPUT_WIDTH'], # head_size
                                               4, # heads
                                               21, #ff_dim
                                               4, # num_transformer_blocks
                                               [7], # mlp_units
                                               args['OUT_STEPS'], #output_steps
                                               0.0, # mlp_dropout
                                               0.0, # dropout
                                              )

    }

    learning_rates = {
        'Last': 0.01,
        'Linear': 0.01,
        'Dense': 0.0001,
        'Conv1D': 0.0001,
        'LSTM': 0.0006,
        'Transformer': 0.0002
    }

    for idx, (name, model) in enumerate(models.items()):
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
        )

        # Train and evaluate
        if name != 'Last':
            history = compile_and_fit(
                model, (X_train, y_train), (X_valid, y_valid),
                patience=args['PATIENCE'], save_folder=output_path / name,
                max_epoch=args['MAX_EPOCHS'], batch_size=args['BATCH_SIZE'],
                start_lr=learning_rates[name], lr_factor=args['LR_FACTOR']
            )

            plot_learning(history.history, y_scale='log', save_p=output_path / name)

            model = keras.models.load_model(output_path / name / 'best_model.keras')

        valid_performance = model.evaluate(X_valid, y_valid)
        test_performance = model.evaluate(X_test, y_test)

        # Compute R2 score
        r2_valid, r2_features_valid, r2_times_valid = batch_R2(model, X_valid, y_valid, 256)
        r2_test, r2_features_test, r2_times_test = batch_R2(model, X_test, y_test, 256)

        performance_df.loc[idx, 'Valid_MSE'], performance_df.loc[idx, 'Valid_MAE'] = valid_performance[1], valid_performance[2]
        performance_df.loc[idx, 'Test_MSE'], performance_df.loc[idx, 'Test_MAE'] = test_performance[1], test_performance[2]
        performance_df.loc[idx, 'Valid_R2'], performance_df.loc[idx, 'Test_R2'] = r2_valid, r2_test
        for i in range(num_features):
            performance_df.loc[idx, 'Valid_R2_'+to_out[i]] = r2_features_valid[i]
            performance_df.loc[idx, 'Test_R2_'+to_out[i]] = r2_features_test[i]

        for i in range(args['OUT_STEPS']):
            performance_df.loc[idx, 'Valid_R2_t+'+str(i+1)] = r2_times_valid[i]
            performance_df.loc[idx, 'Test_R2_t+'+str(i+1)] = r2_times_test[i]

    return performance_df

class Baseline(tf.keras.Model):
    """
    Build Last method. It repeat the last seen value.
    """

    def __init__(self, OUT_STEPS):
        """
        Initialise Last method.

        Parameters
        ----------
        OUT_STEPS : int
            Number of times the last value is repeated.

        Returns
        -------
        None

        """
        super(Baseline, self).__init__()
        self.OUT_STEPS = OUT_STEPS
    
    def call(self, inputs):
        """
        Prediction method.

        Parameters
        ----------
        inputs : numpy.ndarray
            3d tensor, inpunt data.

        Returns
        -------
        tf.tile
            Model prediction.

        """
        return tf.tile(inputs[:, -1:, :], [1, self.OUT_STEPS, 1])

def build_linear_model(args, num_features):
    """
    Build a Linear model.

    Parameters
    ----------
    args : dict
        Dictionary of parameters.
    num_features : int
        Number of features.

    Returns
    -------
    tensorflow.keras model
        Linear model.

    """
    return tf.keras.Sequential([
        #tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(args['OUT_STEPS'] * num_features, kernel_initializer='glorot_uniform'),
        tf.keras.layers.Reshape([args['OUT_STEPS'], num_features])
    ])


def build_dense_model(args, num_features):
    """
    Build a Dense model.

    Parameters
    ----------
    args : dict
        Dictionary of parameters.
    num_features : int
        Number of features.

    Returns
    -------
    tensorflow.keras model
        Dense model.

    """
    return tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        tf.keras.layers.Dense(512, activation='swish', kernel_initializer='glorot_uniform'),
        tf.keras.layers.Dense(args['OUT_STEPS'] * num_features, kernel_initializer='glorot_uniform'),
        tf.keras.layers.Reshape([args['OUT_STEPS'], num_features])
    ])


def build_conv_model(args, num_features):
    """
    Build a 1d Convolutional model.

    Parameters
    ----------
    args : dict
        Dictionary of parameters.
    num_features : int
        Number of features.

    Returns
    -------
    tensorflow.keras model
        1d Convolutional model.

    """
    return tf.keras.Sequential([
        tf.keras.layers.Conv1D(512, activation='relu', kernel_size=3, padding="same", kernel_initializer='glorot_uniform'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(args['OUT_STEPS'] * num_features, kernel_initializer='glorot_uniform'),
        tf.keras.layers.Reshape([args['OUT_STEPS'], num_features])
    ])


def build_lstm_model(args, num_features):
    """
    Build a LSTM model.

    Parameters
    ----------
    args : dict
        Dictionary of parameters.
    num_features : int
        Number of features.

    Returns
    -------
    tensorflow.keras model
        LSTM model.

    """
    return tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=False, activation='swish', kernel_initializer='glorot_uniform'),
        tf.keras.layers.Dense(args['OUT_STEPS'] * num_features, kernel_initializer='glorot_uniform'),
        tf.keras.layers.Reshape([args['OUT_STEPS'], num_features])
    ])


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0):
    """
    Creates a transformer encoder block with multi-head attention,
    feed-forward layers, and residual connections.

    Parameters
    ----------
    inputs : tf.Tensor
        Input tensor to the transformer encoder block with shape (batch_size,
        sequence_length, feature_dim).
    head_size : int
        Dimensionality of each attention head.
    num_heads : int
        Number of attention heads in the multi-head attention mechanism.
    ff_dim : int
        Dimensionality of the feed-forward network.
    dropout : float, optional
        Dropout rate applied to attention and feed-forward layers. The default
        is 0.0.

    Returns
    -------
    tf.Tensor
        Output tensor of the transformer encoder block with the same shape as
        the input tensor.

    """
    # Attention and Normalization
    x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout,
                                           kernel_initializer='glorot_uniform')(inputs, inputs)

    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu", kernel_initializer='glorot_uniform')(res)

    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1, kernel_initializer='glorot_uniform')(x)

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_transformer_model(
            input_shape, head_size,
            num_heads, ff_dim,
            num_transformer_blocks,
            mlp_units,
            out_steps,
            dropout=0.0,
            mlp_dropout=0.0):
    """
    Builds a Transformer-based model for sequence-to-sequence prediction.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input data, excluding the batch size. Typically
        (sequence_length, feature_dim).
    head_size : int
        Dimensionality of each attention head in the transformer encoder.
    num_heads : int
        Number of attention heads in each multi-head attention layer.
    ff_dim : int
        Dimensionality of the feed-forward network in the transformer encoder.
    num_transformer_blocks : int
        Number of transformer encoder blocks to include in the model.
    mlp_units : list of int
        List of integers specifying the number of units in each dense layer
        of the MLP.
    out_steps : int
        Number of time steps to predict in the output sequence.
    dropout : float, optional
        Dropout rate applied within the transformer encoder blocks. The
        default is 0.0.
    mlp_dropout : float, optional
        Dropout rate applied within the MLP layers. The default is 0.0.

    Returns
    -------
    keras.Model
        A compiled Keras model that maps input sequences to output sequences
        with the specified architecture.

    """
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = tf.keras.layers.Dense(dim, activation="swish", kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Dropout(mlp_dropout)(x)

    x = tf.keras.layers.Dense(out_steps*input_shape[-1], kernel_initializer='glorot_uniform')(x)

    # Shape => [batch, out_steps, features].
    outputs = tf.keras.layers.Reshape([out_steps, input_shape[-1]])(x)
    return keras.Model(inputs, outputs)


def compile_and_fit(model, train_set, valid_set, patience=15,
                    save_folder='./', max_epoch=150, batch_size=512,
                    start_lr=1e-4, lr_factor=0.75):
    """
    Function to compile and train models.

    Parameters
    ----------
    model : keras.Model
        model to train.
    train_set : tuple of numpy.ndarray
        Training dataset (Xtrain, Ytrain).
    valid_set : tuple of numpy.ndarray
        Validation dataset (Xvalid, Yvalid).
    patience : int, optional
        Number of iteration before stop training if loss doesn't get lower.
        The default is 15.
    save_folder : pathlib.Path, optional
        Where to save the best model. The default is './'.
    max_epoch : int, optional
        Maximum number of train epochs. The default is 150.
    batch_size : int, optional
        Batch size. The default is 512.
    start_lr : float, optional
        Learning rate starting value. The default is 1e-4.
    lr_factor : float, optional
        Learning rate decrese factor. The default is 0.75.

    Returns
    -------
    history : keras.callbacks.History
        The training history object containing details about the training and
        validation metrics and losses.

    """

    r_lro = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                              factor=lr_factor, patience=2,
                                              min_lr=1e-6, cooldown=1,
                                              verbose=1)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min', verbose=1,
                                                    # restore_best_weights is only activated when patience is reach,
                                                    # if not, it will return the last weights values
                                                    restore_best_weights=False)

    saving = keras.callbacks.ModelCheckpoint(str(save_folder / "best_model.keras"),
                                             save_best_only=True,
                                             monitor="val_loss", verbose=0)

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=start_lr),
                  metrics=['mean_squared_error','mean_absolute_error'])

    history = model.fit(train_set[0], train_set[1], epochs=max_epoch,
                        validation_data=valid_set, callbacks=[r_lro, saving,
                        early_stopping], verbose=1, batch_size=batch_size)

    return history

#=============================================================================
# Metric functions
#=============================================================================

def batch_R2(model, X, y, batch_size):
    """
    Computes the coefficient of determination (R2) for multivariate data,
    both overall and per feature or time step, using batched model
    predictions.

    Parameters
    ----------
    model : keras.Model
        The trained model used to make predictions.
    X : numpy.ndarray
        Input data of shape (num_samples, time_steps, num_features).
    y : numpy.ndarray
        Target data of shape (num_samples, time_steps, num_features).
    batch_size : int
        Number of samples per batch for generating predictions.

    Returns
    -------
    metric : float
        Overall R2 score, considering all features and time steps.
    metric_features : numpy.ndarray
        R2 scores for each feature, averaged across samples and time steps. 
        Shape: (num_features,).
    metric_times : numpy.ndarray
        R2 scores for each time step, averaged across samples and features.
        Shape: (time_steps,).

    """
    preds_f = np.zeros(y.shape)
    for i in range(0, len(X)+batch_size, batch_size):
        if y[i:i+batch_size].shape[0] > 0:
            preds_f[i:i+batch_size] = model(X[i:i+batch_size]).numpy().astype('float64')

    y_f64 = y.astype(np.float64)
    average_y = np.mean(y_f64, axis=(0, 1))

    # All features together
    metric = 1 - np.sum((y_f64 - preds_f)**2)/np.sum((
                            y_f64-average_y[np.newaxis, np.newaxis])**2)

    # Metric per features
    metric_features = 1 - np.sum((y_f64 - preds_f)**2, axis=(0, 1))/np.sum((
                            y_f64-average_y[np.newaxis, np.newaxis])**2, axis=(0, 1))

    # Metric per time step
    metric_times = 1 - np.sum((y_f64 - preds_f)**2, axis=(0, 2))/np.sum((
                            y_f64-average_y[np.newaxis, np.newaxis])**2, axis=(0, 2))

    return metric, metric_features, metric_times

#=============================================================================
# Plot functions
#=============================================================================

def plot_learning(history, y_scale='log', save_p="./", figsize=(16, 4)):
    """
    Function to plot the loss history of train & validation set.

    history : dictionary
        Out of model.fit.
    save_p : Patlib path, optional
        Path for saving the plot. No name needed.
    y_scale : str, optional
        If you want log y scale or an other type. The default is 'log'.
    figsize : tuple, optional
        Size of the figure. The default is (16, 4).

    Returns
    -------
    None

    """
    nb_iter = np.arange(1, len(history['loss'])+1)
    keys = list(history.keys())
    for i in range(len(keys)):
        plt.figure(figsize=figsize)
        plt.grid(True, zorder=1)
        plt.plot(history[keys[i]], label=keys[i], zorder=3)
        if ('lr' in keys[i])|('rate' in keys[i]):
            plt.ylabel('Learning rate', fontsize=14)

        elif 'val' not in keys[i]:
            plt.plot(history['val_'+keys[i]], label='val_'+keys[i], zorder=3)
            plt.ylabel(keys[i], fontsize=14)

        plt.xlabel('Epochs', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        if type(y_scale) == str:
            plt.yscale(y_scale)

        plt.savefig(str(save_p)+'/'+keys[i]+'.png', bbox_inches='tight')
        plt.close()
        plt.show()


#=============================================================================
# Meta-data functions
#=============================================================================

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

#=============================================================================
# main
#=============================================================================


if __name__ == "__main__":
    # Load configuration
    config = configparser.ConfigParser()
    config.read('gnss.ini')
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
        'LR_FACTOR': config['MODEL'].getfloat('LR_FACTOR'),
        'PATIENCE': config['MODEL'].getint('PATIENCE'),
        'SEED': config['MODEL'].getint('SEED'),
        'SAVE_REPARTITION': config['METADATA'].getboolean('SAVE_REPARTITION'),
        'SAVE_SHAPE': config['METADATA'].getboolean('SAVE_SHAPE'),
        'SAVE_NAME_FEATURES': config['METADATA'].getboolean('SAVE_NAME_FEATURES'),
        'SAVE_PY_COPY': config['METADATA'].getboolean('SAVE_PY_COPY'),
        'SAVE_REPARTITION_IMG': config['METADATA'].getboolean('SAVE_REPARTITION_IMG'),
        'SAVE_HISTORY_IMG': config['METADATA'].getboolean('SAVE_HISTORY_IMG')
    }

    # Copy the configuration file for the record in the working directory
    ROOT_PATH = Path(os.path.dirname('./'))
    DEST_PATH = ROOT_PATH / args['PATH']
    DEST_PATH.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT_PATH / 'gnss.ini', DEST_PATH / 'gnss.ini')

    if config['METADATA'].getboolean('SAVE_PY_COPY'):
        shutil.copyfile(ROOT_PATH / 'gnss.py', DEST_PATH / 'gnss.py')

    # Run multiple configurations
    for run in range(NB_RUNS):
        args['SEED'] = run
        main(args)
