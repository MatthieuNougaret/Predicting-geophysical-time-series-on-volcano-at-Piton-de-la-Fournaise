import os
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import configparser
from pathlib import Path
import shutil
import matplotlib as mpl
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

    if args['DATASET'] == 'gnss':
        df_data_raw, df_data_norm, date_time = get_gnss_df(args, PATH, DATA_PATH)

    elif args['DATASET'] == 'seismicity':
        df_data_raw, df_data_norm, date_time = get_seismic_df(args, PATH, DATA_PATH)

    feature_names = np.array(list(df_data_raw.columns))
    path_model = PATH / args['SEED'] / args['MODEL'] / 'best_model.keras'
    model = keras.models.load_model(path_model)

    plot_predictions(args,
                     df_data_raw,
                     df_data_norm,
                     date_time,
                     model,
                     PATH,
                     figsize=(6.44, 3.22),
                     dpi=200,
                     limit_proportion=0.02,
                     ylims=None,
                     xticks_sz=13,
                     yticks_sz=13,
                     chanel_lw=2,
                     chanel_ms=6,
                     size_label=16,
                     lw_grid=0.6,
                     alpha_grid=0.7,
                     lw_pred=3,
                     legend_sz=8,
                     legend_mksc=1.5)

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

def split_data_train(args, df, path):
    """
    Splits the data into training, validation, and test sets from
    the saved meta-data.
    """
    df_meta = pd.read_csv(path / args['SEED'] / 'metadata.csv')
    split_index = df_meta.loc[[0, 1, 2], '0'].to_list()
    for i in range(3):
        split_index[i] = split_index[i][1:-1].split(', ')

    split_index = np.array(split_index, dtype=int)

    train_df = df.copy().loc[split_index[0, 0]:split_index[0, 1]].reset_index(drop=True)

    return train_df

def standardize_gnss_df(df_raw, train_df):
    """
    Standardizes datasets using train mean and std deviation.
    """
    train_std = train_df.std()

    # we only use the standard deviation
    df_norm = df_raw.copy() / train_std
    return df_norm

def standardize_seismicity_df(df_raw, train_df):
    """
    Standardizes datasets using train mean and std deviation.
    """
    train_mean, train_std = train_df.mean(), train_df.std()

    df_norm = (df_raw.copy() - train_mean) / train_std
    return df_norm

def get_gnss_df(args, path, data_path):
    # Load and preprocess data
    df, date_time = load_and_preprocess_gnss_data(data_path)

    # Split data to get training sets for normalisation
    train_df = split_data_train(args, df, path)

    # Standardize data
    df_norm = standardize_gnss_df(df, train_df)

    return df, df_norm, date_time

def get_seismic_df(args, path, data_path):
    # Load and preprocess data
    df, date_time = load_and_preprocess_seismicity_data(data_path)

    # Split data to get training sets for normalisation
    train_df = split_data_train(args, df, path)

    # Standardize data
    df_norm = standardize_seismicity_df(df, train_df)

    return df, df_norm, date_time

#=============================================================================
# Plot functions
#=============================================================================

def xlim_from_args(args):
    xlims = np.array(args['TIME_LIMITS'], dtype='datetime64[h]')
    return xlims

def time_to_index(args, date_time):
    plot_index = np.argwhere(date_time == np.array(args['PLOT_TIMES'],
                             dtype='datetime64[h]')[:, np.newaxis])[:, 1]

    return plot_index

def new_line_str(text):
    array_c = np.array(list(text)).astype(object)
    if '\\' in array_c:
        pos_spe = np.argwhere(array_c == '\\')
        mask_n = array_c[pos_spe+1] == 'n'
        array_c[pos_spe[mask_n]] = '\n'
        np.delete(array_c, pos_spe+1)
        new_text = np.sum(array_c)
    else:
        new_text = text

    return new_text

def compute_prediction(args, index, dataf_n, date_time, model):
    """
    Function to give model prediction and time reference of this predition.

    Parameters
    ----------
    args : dict
        Dictionary of parameters.
    index : int
        Index to slice the standardized DataFrame to make prediction.
    dataf_n : pandas.DataFrame
        Standardized data.
    date_time : numpy.ndarray
        1D array with time.
    model : tensorflow / keras model
        model we want to use for prediction.

    Returns
    -------
    predictions : numpy.ndarray
        model prediction.
    ref_time : numpy.ndarray
        time for the model prediction.

    """
    # [index-IN, index-1] <= X
    st_x, ed_x = index-args['INPUT_WIDTH'], index

    # [index, index+OUT] <= y
    st_y, ed_y = (index, index+args['OUT_STEPS'])

    features = np.array(list(dataf_n.columns))
    nfeatures = len(dataf_n.columns)
    if args['FEATURE'] not in features:
        raise ValueError('The model was not trained to predict'\
                         ' '+args['FEATURE'])

    id_y = np.where(features == args['FEATURE'])[0][0]
    input_n = dataf_n.loc[st_x:ed_x-1].to_numpy()

    # the 0 is for the batch index
    prediction = model(input_n[np.newaxis]).numpy()[0, :, id_y]
    ref_time = date_time[st_y:ed_y]
    return prediction, ref_time

def plot_predictions(args, df_raw, df_norm, date_time, model, path,
                     figsize=(3.22, 3.22), dpi=200,
                     limit_proportion=0.02, ylims=None,
                     xticks_sz=13, yticks_sz=13,
                     chanel_lw=2, chanel_ms=6,
                     size_label=16,
                     lw_grid=0.6, alpha_grid=0.7,
                     lw_pred=3,
                     legend_sz=8, legend_mksc=1.5):

    xlims = xlim_from_args(args)
    time_split_id = np.argwhere(date_time == xlims[:, np.newaxis])[:, 1]
    timing = date_time[time_split_id[0]:time_split_id[1]+1]
    raw_to_plot = df_raw.loc[time_split_id[0]:time_split_id[1],
                             args['FEATURE']].to_numpy().astype(float)

    norm_to_plot = df_norm.loc[time_split_id[0]:time_split_id[1],
                               args['FEATURE']].to_numpy().astype(float)

    if type(ylims) == type(None):
        prop_lim_r = (np.max(raw_to_plot)-
                      np.min(raw_to_plot))*limit_proportion
        ylims_r = (np.min(raw_to_plot)-prop_lim_r,
                   np.max(raw_to_plot)+prop_lim_r)

        prop_lim_n = (np.max(norm_to_plot)-
                      np.min(norm_to_plot))*limit_proportion
        ylims_n = (np.min(norm_to_plot)-prop_lim_n,
                   np.max(norm_to_plot)+prop_lim_n)

    else:
        ylims_r = ylims[0]
        ylims_n = ylims[1]

    x_ticks_p = timing[args['START_X_TICKS']::args['FREQ_X_TICKS']]
    x_ticks_t = x_ticks_p.astype(str).tolist()

    for i in range(len(x_ticks_t)):
        x_ticks_t[i] = (x_ticks_t[i][5:7]+'/'+x_ticks_t[i][8:10]+'\n'+
                        x_ticks_t[i][11:]+'h')

    vlines = np.arange(xlims[0], xlims[1], args['STEP_X_GRID'],
                       dtype='datetime64[h]')

    hlines = np.arange(args['START_Y_TICKS'], ylims_r[1]+1,
                       args['STEP_Y_GRID'])

    plot_index = time_to_index(args, date_time)

    args['Y_LABEL'] = new_line_str(args['Y_LABEL'])

    mpl.rcParams['xtick.labelsize'] = xticks_sz
    mpl.rcParams['ytick.labelsize'] = yticks_sz

    fig, ax1 = plt.subplots()
    fig.set_size_inches(figsize)
    fig.set_dpi(dpi)
    ax2 = ax1.twinx()

    ax1.set(ylim=(ylims_r[0], ylims_r[1]))
    ax2.set(ylim=(ylims_n[0], ylims_n[1]))
    ax1.set(xlim=(xlims[0], xlims[1]))
    ax2.set(xlim=(xlims[0], xlims[1]))
    
    ax1.plot(timing, raw_to_plot, '.-', lw=chanel_lw, color='k',
             markerfacecolor='none', label=args['FEATURE_LABEL'],
             zorder=10, ms=chanel_ms)

    make_leg_er = True
    make_leg_in = True
    for i in range(len(args['EVENTS_TYPE'])):
        if args['EVENTS_TYPE'][i] == 'Eruption' and make_leg_er:
            make_leg_er = False
            ax1.fill_between(args['EVENTS_LIMITS'][i], -1e9, 1e9,
                             color='m', alpha=0.2, label='Eruption',
                             zorder=2, lw=0)

        elif args['EVENTS_TYPE'][i] == 'Eruption':
            ax1.fill_between(args['EVENTS_LIMITS'][i], -1e9, 1e9,
                             color='m', alpha=0.2, zorder=2, lw=0)

        if args['EVENTS_TYPE'][i] == 'Intrusion' and make_leg_in:
            make_leg_in = False
            ax1.fill_between(args['EVENTS_LIMITS'][i], -1e9, 1e9,
                             color='b', alpha=0.2, label='Intrusion',
                             zorder=2, lw=0)

        elif args['EVENTS_TYPE'][i] == 'Intrusion':
            ax1.fill_between(args['EVENTS_LIMITS'][i], -1e9, 1e9,
                             color='b', alpha=0.2, zorder=2, lw=0)

    for i, idx in enumerate(plot_index):
        predi, predi_t = compute_prediction(args, idx, df_norm, date_time, model)
        ax2.plot(predi_t, predi, '--', lw=lw_pred, color=args['COLORS_PREDS'][i],
                         markerfacecolor='none', label=' ')

    ax1.vlines(vlines, -1e9, 1e9, color='grey', lw=lw_grid, alpha=alpha_grid)
    ax1.hlines(hlines, -1e9, 1e9, color='grey', lw=lw_grid, alpha=alpha_grid)
    ax1.legend(loc='upper right', fontsize=legend_sz, markerscale=legend_mksc,
               handlelength=3.5)

    ax2.legend(loc='center right', fontsize=legend_sz, columnspacing=0.5,
               markerscale=legend_mksc, handletextpad=0.,
               handlelength=3.5, title='Model predictions', ncols=2,
               title_fontsize='x-small')

    ax1.set_xticks(x_ticks_p, x_ticks_t)
    ax2.set_yticks([], [])
    ax1.set_xlabel('Time', fontsize=size_label)
    ax1.set_ylabel(args['Y_LABEL'], fontsize=size_label)

    fig_name = (args['DATASET'] + '_seed_' + args['SEED'] + '_' +
                args['MODEL'] + '_predictions_' + args['FEATURE'] + '.png')

    plt.savefig(path / fig_name, bbox_inches='tight')
    plt.close()
    plt.show()

#=============================================================================
# main
#=============================================================================


if __name__ == "__main__":
    # Load configuration
    config = configparser.ConfigParser()
    config.read('plot_prediction.ini')

    PLOT_TIMES = config['FIGURE'].get('PLOT_TIMES').split(',')
    COLORS_PREDS = config['FIGURE'].get('COLORS_PREDS').split(',')
    TIME_LIMITS = config['FIGURE'].get('TIME_LIMITS').split(',')
    EVENTS_LIMITS = config['FIGURE'].get('EVENTS_LIMITS').split(',')
    EVENTS_LIMITS = np.array(EVENTS_LIMITS,
             dtype='datetime64[m]').reshape((len(EVENTS_LIMITS)//2, 2))

    EVENTS_TYPE = config['FIGURE'].get('EVENTS_TYPE').split(',')

    # Parse arguments
    args = {
        'DATASET': config['DATA'].get('DATASET'),
        'PATH': config['DATA'].get('PATH'),
        'DATA_PATH': config['DATA'].get('DATA_PATH'),
        'INPUT_WIDTH': config['WINDOW'].getint('INPUT_WIDTH'),
        'OUT_STEPS': config['WINDOW'].getint('OUT_STEPS'),
        'SHIFT': config['WINDOW'].getint('SHIFT'),
        'SEED': config['MODEL'].get('SEED'),
        'MODEL': config['MODEL'].get('MODEL'),
        'PLOT_TIMES': PLOT_TIMES,
        'COLORS_PREDS': COLORS_PREDS,
        'FEATURE': config['FIGURE'].get('FEATURE'),
        'TIME_LIMITS': TIME_LIMITS,
        'EVENTS_LIMITS': EVENTS_LIMITS,
        'EVENTS_TYPE': EVENTS_TYPE,
        'FEATURE_LABEL': config['FIGURE'].get('FEATURE_LABEL'),
        'STEP_X_GRID': config['FIGURE'].getint('STEP_X_GRID'),
        'FREQ_X_TICKS': config['FIGURE'].getint('FREQ_X_TICKS'),
        'START_X_TICKS': config['FIGURE'].getint('START_X_TICKS'),
        'START_Y_TICKS': config['FIGURE'].getfloat('START_Y_TICKS'),
        'STEP_Y_GRID': config['FIGURE'].getfloat('STEP_Y_GRID'),
        'Y_LABEL': config['FIGURE'].get('Y_LABEL'),
    }

    main(args)


