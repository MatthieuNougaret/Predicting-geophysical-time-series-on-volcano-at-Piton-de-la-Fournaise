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

# Set up the main function
def main(args):
    # Setup paths
    DATA_PATH = Path(args['DATA_PATH'])
    OUTPUT_PATH = Path(args['PATH']) / str(args['SEED'])
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Set parameters
    SEED = args['SEED']
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    # Load and preprocess data
    df, date_time = load_and_preprocess_data(DATA_PATH)
    feature_names = np.array(list(df.columns))
    to_in_models = feature_names.copy()
    to_out = feature_names.copy()

    # Split data into training, validation, and test sets
    train_df, valid_df, test_df = split_data(df, date_time, OUTPUT_PATH)

    # Standardize data
    train_df, valid_df, test_df, train_mean, train_std = standardize_data(train_df, valid_df, test_df)

    # Prepare datasets
    X_train, y_train = create_dataset(train_df, args['INPUT_WIDTH'], args['OUT_STEPS'], threshold=3.0)
    X_valid, y_valid = create_dataset(valid_df, args['INPUT_WIDTH'], args['OUT_STEPS'])
    X_test, y_test = create_dataset(test_df, args['INPUT_WIDTH'], args['OUT_STEPS'])

    print(len(train_df))
    print(X_train.shape)
    # Compile and evaluate models
    performance = evaluate_models(
        X_train, y_train, X_valid, y_valid, X_test, y_test, args, to_out, OUTPUT_PATH)

    # Save performance metrics
    performance.to_csv(OUTPUT_PATH / "performance.csv")

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

    Returns
    -------
    None

    """
    for i in range(len(windows)):
        key_i = 'grad_'+adding+str(windows[i])
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
    """Loads data and performs initial preprocessing."""
    df = pd.read_excel(data_path, sheet_name="all_merged").dropna().reset_index(drop=True)
    date_time = pd.to_datetime(df.pop('date'), format='datetime64[D]')
    cutoff = np.where(date_time > '2007-12-31')[0][0]
    df, date_time = df.loc[cutoff:].reset_index(drop=True), date_time[cutoff:].reset_index(drop=True).to_numpy()

    # Ensure columns are numeric
    df = df.apply(pd.to_numeric, errors='coerce').astype(float)

    # Generate gradient features
    for col in df.columns:
        make_gradients(df, [3, 10, 30], col, col)
        df = df.drop(columns=col)

    return df, date_time

def split_data(df, date_time, OUTPUT_PATH, generate_figure=True):
    """Splits the data into training, validation, and test sets."""
    valid_fraction, test_fraction = 0.15, 0.15
    train_fraction = 1 - valid_fraction - test_fraction
    nb_data = len(df)

    cut_train = int(train_fraction * nb_data)
    cut_valid = int((train_fraction + valid_fraction) * nb_data)

    train_df = df[:cut_train]
    valid_df = df[cut_train:cut_valid]
    test_df = df[cut_valid:]

    # MAKE THE FIGURE
    if generate_figure == True:

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
    
        plt.ylabel("GNSS baseline", fontsize=16)
        plt.xlabel('Time', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(title='Datasets', fontsize=16, title_fontsize=16)
        plt.savefig(OUTPUT_PATH / 'sets_repartition.png', bbox_inches='tight')

    plt.close()
    plt.show()

    return train_df, valid_df, test_df


def standardize_data(train_df, valid_df, test_df):
    """Standardizes datasets using train mean and std deviation."""
    train_mean, train_std = train_df.mean(), train_df.std()
    # we only use the standard deviation
    train_df, valid_df, test_df = train_df / train_std, valid_df / train_std, test_df / train_std
    return train_df, valid_df, test_df, train_mean, train_std

def create_dataset(df, input_width, out_steps, threshold=None):
    """
    Creates time series dataset for training/validation/testing with optional filtering 
    based on a spike threshold.
    
    Parameters:
    - df: pandas DataFrame, time series data
    - input_width: int, number of input time steps in each sequence
    - out_steps: int, number of output time steps to predict
    - threshold: float, optional threshold to filter sequences that contain spikes
    
    Returns:
    - X: numpy array of shape (filtered_num_sequences, input_width), input sequences
    - y: numpy array of shape (filtered_num_sequences, out_steps), target sequences
    """
    # Set appropriate dtype based on the length of df
    dtype = 'int32' if len(df) < 2147483646 else 'int64'
    data_type = 'float64'

    # Generate indices for input and target sequences
    id_indices = np.arange(0, len(df) - input_width - out_steps + 1, dtype=dtype)[:, np.newaxis]
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

def evaluate_models(X_train, y_train, X_valid, y_valid, X_test, y_test, args, to_out, output_path):
    """Evaluates models and saves training history."""
    num_features = len(to_out)
    performance_df = pd.DataFrame(['Last', 'Linear', 'Dense', 'Conv', 'LSTM', 'Transformer'], columns=["Model"])
    performance_df['Valid_MSE'], performance_df['Valid_MAE'] = 0.0, 0.0
    performance_df['Test_MSE'], performance_df['Test_MAE'] = 0.0, 0.0

    # Define models
    models = {
        'Last': Baseline(args['OUT_STEPS']),
        'Linear': build_linear_model(args, num_features),
        'Dense': build_dense_model(args, num_features),
        'Conv1D': build_conv_model(args, num_features),
        'LSTM': build_lstm_model(args, num_features),
        'Transformer': build_attention_model((args['INPUT_WIDTH'], num_features), # Shape of input (timesteps, features)
                      1, # heads
                      16, #ff_dim
                      2, #num_transformer_blocks
                      args['OUT_STEPS'], #output_steps
                      num_features, #num_features
                      )

        #f.build_transformer_model(
        #    (X_train.shape[1], X_train.shape[2]),
        #    head_size=args['INPUT_WIDTH'], num_heads=1, ff_dim=16, num_transformer_blocks=2,
        #    mlp_units=[args['INPUT_WIDTH']], SEED=args['SEED'], mlp_dropout=0.0, dropout=0.0,
        #    num_features=num_features, OUT_STEPS=args['OUT_STEPS'])
    }

    learning_rates = {
        'Last': 0.001,
        'Linear': 0.001,
        'Dense': 0.0003,
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

        valid_performance = model.evaluate(X_valid, y_valid)
        test_performance = model.evaluate(X_test, y_test)
        
        performance_df.loc[idx, 'Valid_MSE'], performance_df.loc[idx, 'Valid_MAE'] = valid_performance[1], valid_performance[2]
        performance_df.loc[idx, 'Test_MSE'], performance_df.loc[idx, 'Test_MAE'] = test_performance[1], test_performance[2]

    return performance_df

class Baseline(tf.keras.Model):
    """
    Repeat the last seen value
    """

    def __init__(self, OUT_STEPS):
        super(Baseline, self).__init__()
        self.OUT_STEPS = OUT_STEPS
    
    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, self.OUT_STEPS, 1])

def build_linear_model(args, num_features):
    return tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        tf.keras.layers.Dense(args['OUT_STEPS'] * num_features, kernel_initializer='glorot_uniform'),
        tf.keras.layers.Reshape([args['OUT_STEPS'], num_features])
    ])


def build_dense_model(args, num_features):
    return tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        tf.keras.layers.Dense(512, activation='swish', kernel_initializer='glorot_uniform'),
        tf.keras.layers.Dense(args['OUT_STEPS'] * num_features, kernel_initializer='glorot_uniform'),
        tf.keras.layers.Reshape([args['OUT_STEPS'], num_features])
    ])


def build_conv_model(args, num_features):
    return tf.keras.Sequential([
        tf.keras.layers.Conv1D(512, activation='relu', kernel_size=args['CONV_WIDTH'], padding="same", kernel_initializer='glorot_uniform'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(args['OUT_STEPS'] * num_features, kernel_initializer='glorot_uniform'),
        tf.keras.layers.Reshape([args['OUT_STEPS'], num_features])
    ])


def build_lstm_model(args, num_features):
    return tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=False, activation='swish', kernel_initializer='glorot_uniform'),
        tf.keras.layers.Dense(args['OUT_STEPS'] * num_features, kernel_initializer='glorot_uniform'),
        tf.keras.layers.Reshape([args['OUT_STEPS'], num_features])
    ])

def build_attention_model(input_shape, 
                          num_heads, 
                          ff_dim, 
                          num_transformer_blocks, 
                          output_steps, 
                          num_features):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Positional Encoding: Adds spatial information to the time series data
    x = inputs + tf.range(input_shape[0], dtype=tf.float32)[tf.newaxis, :, tf.newaxis]
    
    # Transformer Blocks
    for _ in range(num_transformer_blocks):
        # Multi-head Attention Layer
        x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1], dropout=0.1)(x, x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-Forward Layer
        x_ff = tf.keras.layers.Dense(ff_dim, activation='relu')(x) #
        x_ff = tf.keras.layers.Dense(input_shape[-1])(x_ff)
        x = tf.keras.layers.Add()([x, x_ff])  # Residual connection
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Dense Layer for Output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(output_steps * num_features)(x)
    outputs = tf.keras.layers.Reshape((output_steps, num_features))(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def compile_and_fit(model, train_set, valid_set, patience=15,
                    save_folder='./', max_epoch=150, batch_size=512,
                    start_lr=1e-4, lr_factor=0.75):
    """
    
    Parameters
    ----------
    model : model to train.
    train_set : (Xtrain, Ytrain).
    valid_set : (Xvalid, Yvalid).
    patience : n-iter before stop training if loss doesn't get lower.
    save_folder : Posix path to save the best model.
    max_epoch : max number of train epochs.
    batch_size : batch size.
    start_lr : Learning rate starting value.
    lr_factor : Learning rate decrese factor.

    Returns
    -------
    history

    """

    r_lro = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                              factor=lr_factor, patience=2,
                                              min_lr=1e-6, cooldown=1,
                                              verbose=1)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min', verbose=1,
                                                    restore_best_weights=True)

    saving = keras.callbacks.ModelCheckpoint(str(save_folder / "best_model.keras"),
                                             save_best_only=True,
                                             monitor="val_loss", verbose=0)

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  #loss=lambda y_true, y_pred: combined_spike_loss(y_true, y_pred, threshold=1.0, spike_weight=10.0),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=start_lr),
                  metrics=['mean_squared_error','mean_absolute_error'])

    history = model.fit(train_set[0], train_set[1], epochs=max_epoch,
                        validation_data=valid_set, callbacks=[r_lro, saving,
                        early_stopping], verbose=1, batch_size=batch_size)

    return history

#=============================================================================
# Plot functions
#=============================================================================

def plot_learning(history, y_scale='log', save_p="./"):
    """
    Function to plot the loss history of train & validation set.

    history : dictionary
        Out of model.fit.
    save_p : Patlib path
        Path for saving the plot. No name needed.
    y_scale : str, optional
        If you want log y scale or an other type. The default is None.

    Returns
    -------
    None

    """
    
    plt.figure(figsize=(16, 4))
    plt.grid(True, zorder=1)
    nb_iter = np.arange(1, len(history['loss'])+1)
    plt.plot(nb_iter, history['loss'], label="MSE Loss, train")
    plt.plot(nb_iter, history['val_loss'], label="MSE Loss, valid")
    plt.legend(fontsize=12)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.yscale(y_scale)
    plt.savefig(save_p / 'loss_MSE.png', bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    # Load configuration
    config = configparser.ConfigParser()
    config.read('gnss.ini')
    NB_RUNS = config['MODELE'].getint('NB_RUNS')

    # Parse arguments
    args = {
        'DATA_PATH': config['DATA'].get('DATA_PATH'),
        'PATH': config['DATA'].get('PATH'),
        'INPUT_WIDTH': config['WINDOW'].getint('INPUT_WIDTH'),
        'OUT_STEPS': config['WINDOW'].getint('OUT_STEPS'),
        'SHIFT': config['WINDOW'].getint('SHIFT'),
        'MAX_EPOCHS': config['MODELE'].getint('MAX_EPOCHS'),
        'BATCH_SIZE': config['MODELE'].getint('BATCH_SIZE'),
        'LR_FACTOR': config['MODELE'].getfloat('LR_FACTOR'),
        'CONV_WIDTH': config['MODELE'].getint('CONV_WIDTH'),
        'PATIENCE': config['MODELE'].getint('PATIENCE'),
        'SEED': config['MODELE'].getint('SEED')
    }

    # Copy the configuration file for the record in the working directory
    ROOT_PATH = Path(os.path.dirname('./'))
    DEST_PATH = ROOT_PATH / args['PATH']
    DEST_PATH.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT_PATH / 'gnss.ini', DEST_PATH / 'gnss.ini')

    # Run multiple configurations
    for run in range(NB_RUNS):
        args['SEED'] = run
        main(args)
