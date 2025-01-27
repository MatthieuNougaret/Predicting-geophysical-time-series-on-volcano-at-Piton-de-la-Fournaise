import os
import numpy as np
import pandas as pd
import configparser
from pathlib import Path
import shutil
import matplotlib.pyplot as plt

# Set up the main function
def main(args):
    # Setup paths
    PATH = Path(args['PATH'])
    if args['COMPILE_RUNS']:
        results = []
        for i in range(args['NB_RUNS']):
            df = pd.read_csv('./' / PATH / str(i) / 'performance.csv')
            results.append(df.values[:, 1:])

        metrics = np.array(list(df.columns)[1:])
        models = df.values[:, 0]
        results = np.array(results, dtype=float)

        df_compile = compute_compile(results, models, metrics)

        # Save performance metrics
        df_compile.to_csv(PATH / "performance_compiled.csv", index=False)

    if (len(args['METRICS_TO_PLOT']) > 0) and (os.path.isfile(PATH / "performance_compiled.csv")):
        df_compile = pd.read_csv(PATH / 'performance_compiled.csv')
        for key in args['METRICS_TO_PLOT']:
            if len(key) > 0:
                plot_bar(args,
                         df_compile,
                         key,
                         PATH,
                         figsize=(6.44, 6.44),
                         dpi=200,
                         ylabel='Mean Squared Error median',
                         xlims=(-0.4, 5.4),
                         ylims=(1.00, 3.25),
                         limit_proportion=0.05,
                         model_names=('Last', 'Linear', 'Dense', 'CNN 1D',
                                      'LSTM', 'Transformer'))

    if args['PLOT_R2_TIME'] and (os.path.isfile(PATH / "performance_compiled.csv")):
        df_compile = pd.read_csv(PATH / 'performance_compiled.csv')
        if args['DATASET'] == 'gnss':
            plot_R2_vs_time(args,
                            df_compile,
                            PATH,
                            figsize=(6.44, 6.44),
                            dpi=200,
                            tick_sz=10,
                            label_sz=12,
                            xlims=None,
                            ylims=None,
                            limit_proportion=0.02,
                            xticks_freq=None,
                            xticks_start=1)

        if args['DATASET'] == 'seismicity':
            plot_R2_vs_time(args,
                            df_compile,
                            PATH,
                            figsize=(9.66, 6.44),
                            dpi=200,
                            tick_sz=10,
                            label_sz=12,
                            xlims=None,
                            ylims=None,
                            limit_proportion=0.02,
                            xticks_freq=3,
                            xticks_start=1)

    if args['PLOT_R2_FEATURE'] and (os.path.isfile(PATH / "performance_compiled.csv")):
        df_compile = pd.read_csv(PATH / 'performance_compiled.csv')
        if args['DATASET'] == 'gnss':
            fnames = np.array([
                        '3 days DSRG-SNEG', '10 days DSRG-SNEG',
                        '30 days DSRG-SNEG', '3 days BOMG-DSRG',
                        '10 days BOMG-DSRG', '30 days BOMG-DSRG',
                        '3 days BOMG-BORG', '10 days BOMG-BORG',
                        '30 days BOMG-BORG', '3 days BORG-SNEG',
                        '10 days BORG-SNEG', '30 days BORG-SNEG',
                        '3 days BOMG-DERG', '10 days BOMG-DERG',
                        '30 days BOMG-DERG', '3 days DERG-SNEG',
                        '10 days DERG-SNEG', '30 days DERG-SNEG',
                        '3 days BORG-DSRG', '10 days BORG-DSRG',
                        '30 days BORG-DSRG', '3 days BORG-DERG',
                        '10 days BORG-DERG', '30 days BORG-DERG',
                        '3 days DERG-DSRG', '10 days DERG-DSRG',
                        '30 days DERG-DSRG', '3 days BOMG-SNEG',
                        '10 days BOMG-SNEG', '30 days BOMG-SNEG'])

            plot_R2_vs_features(args, df_compile, PATH,
                                figsize=(9.66, 3.22),
                                dpi=200, 
                                delta_p=0.05,
                                ylabel='R2 score',
                                xlims=None,
                                ylims=None,
                                limit_proportion=0.01,
                                model_names=None,
                                x_ticks=fnames)

        if args['DATASET'] == 'seismicity':
            fnames = np.array([
                        'Hourly seismicity', '6 hours gradient',
                        '12 hours gradient', '24 hours gradient'])

            plot_R2_vs_features(args, df_compile, PATH,
                                figsize=(6.44, 3.22),
                                dpi=200, 
                                delta_p=0.15,
                                ylabel='R2 score',
                                xlims=None,
                                ylims=None,
                                limit_proportion=0.04,
                                model_names=None,
                                x_ticks=fnames)

#=============================================================================
# Data functions
#=============================================================================

def compute_compile(results, models, metrics):
    """
    Function to make a summary of the n-th runs of the models.

    Parameters
    ----------
    results : numpy.ndarray
        3d array of float. Consists of metric score of the models for test
        and valid datasets. 
    models : numpy.ndarray
        1d vector of string. List of the tested models/methods.
    metrics : numpy.ndarray
        1d vector of string. List of the metrics computed.

    Returns
    -------
    df_compile : pandas.DataFrame
        Data Frame with minimum, median average, maximum and deviation of
        the metrics for the models over the n-th initializations.

    """
    df_compile = {}
    df_compile['models'] = models
    for i in range(len(metrics)):
        df_compile[metrics[i]+'_minimum'] = np.min(results[:, :, i], axis=0)
        df_compile[metrics[i]+'_median'] = np.median(results[:, :, i], axis=0)
        df_compile[metrics[i]+'_average'] = np.mean(results[:, :, i], axis=0)
        df_compile[metrics[i]+'_maximum'] = np.max(results[:, :, i], axis=0)
        df_compile[metrics[i]+'_deviation'] = np.std(results[:, :, i], axis=0)

    df_compile = pd.DataFrame.from_dict(df_compile)
    return df_compile

#=============================================================================
# Plot functions
#=============================================================================

def plot_bar(args, df_compile, metric_name, save_path, figsize=(3.22, 3.22),
             dpi=200, ylabel=None, xlims=None, ylims=None,
             limit_proportion=0.02, model_names=None):
    """
    Function to plot median of a metric with bar plot.

    Parameters
    ----------
    args : dict
        Dictionary with init file informations.
    df_compile : pandas.DataFrame
        Data Frame with minimum, median average, maximum and deviation of
        the metrics for the models over the n-th initializations.
    metric_name : string
        Name of the metric to plot.
    save_path : string or Pathlib.Path
        Where to save the figure.
    figsize : tuple, optional
        Size of the figure in inches. The default is (3.22, 3.22).
    dpi : integer, optional
        Dots per inches. The default is 200.
    ylabel : string or None, optional
        Y-label string. The default is None.
    xlims : tuple or None, optional
        Lower and upper limit of x-axis. The default is None.
    ylims : tuple or None, optional
        Lower and upper limit of y-axis. The default is None.
    limit_proportion : floating, optional
        Width proportion for x and y-axis. The default is 0.02.
    model_names : tuple or None, optional
        1d vector listing the model's name. The default is None.

    Returns
    -------
    None.

    """
    data = df_compile[metric_name].to_numpy().astype(float)
    if type(model_names) == type(None):
        model_names = df_compile['models'].to_numpy().astype(str)

    n_models = len(model_names)
    positions = np.arange(n_models)
    if type(ylabel) == type(None):
        ylabel = metric_name

    if type(xlims) == type(None):
        delta = n_models*limit_proportion
        xlims = (-delta, n_models-1+delta)

    if type(ylims) == type(None):
        delta = (np.max(data)-np.min(data))*limit_proportion
        ylims = (np.min(data)-delta, np.max(data)+delta)

    plt.figure(figsize=figsize, dpi=dpi)
    plt.grid(True, zorder=1)
    plt.bar(positions, data, bottom=0.0, color='green', zorder=2,
            width=0.5)

    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(positions, model_names, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])

    plt.savefig(str(save_path)+'/'+args['DATASET']+'_'+metric_name+'.png',
                bbox_inches='tight')

    plt.close()
    plt.show()

def plot_R2_vs_time(args, df_compile, path, figsize=(3.22, 3.22), dpi=200,
                    tick_sz=12, label_sz=12, xlims=None, ylims=None,
                    limit_proportion=0.02, xticks_freq=None, xticks_start=0):
    """
    Function to plot median of a metric

    Parameters
    ----------
    args : dict
        Dictionary with init file informations
    df_compile : pandas.DataFrame
        Data Frame with minimum, median average, maximum and deviation of
    	the metrics for the models over the n-th initializations.
    path : string or Pathlib.Path
        Where to save the figure.
    figsize : tuple, optional
        Size of the figure in inches. The default is (3.22, 3.22).
    dpi : integer, optional
        Dots per inches. The default is 200.
    tick_sz : float, optional
        . The default is 12.
    label_sz : float, optional
        . The default is 12.
    xlims : tuple or None, optional
        Lower and upper limit of x-axis. The default is None.
    ylims : tuple or None, optional
        Lower and upper limit of y-axis. The default is None.
    limit_proportion : floating, optional
        Width proportion for x and y-axis. The default is 0.02.
    xticks_freq : int, optional
        Frequency of the sampling for the x-axis. The default is None.
    xticks_start : int, optional
        Index at which the sampling start. The default is 0.

    Returns
    -------
    None.

    """
    models_nm = np.array(list(df_compile['models']))
    col_l = list(df_compile.columns)
    using = []
    x_ticks = []
    c = 1
    for i in range(len(col_l)):
        if col_l[i][:10] == 'Test_R2_t+':
            if col_l[i][-7:] == '_median':
                using.append(col_l[i])
                x_ticks.append('t+'+str(c)+' '+args['TIME_STEP'])
                c += 1

    kernel = np.arange(1, len(using)+1, 1)
    compile_used = df_compile.loc[:, using].to_numpy()
    if type(xlims) == type(None):
        delta = (len(using))*limit_proportion
        xlims = (1-delta, len(using)+delta)

    if type(ylims) == type(None):
        delta = (np.max(compile_used)-np.min(compile_used))*limit_proportion
        ylims = (np.min(compile_used)-delta, np.max(compile_used)+delta)

    plt.figure(figsize=figsize, dpi=dpi)
    plt.grid(True, zorder=1)
    for i in range(len(models_nm)):
        line = np.argwhere(models_nm[i] == models_nm)[0, 0]
        plt.plot(kernel, compile_used[line], '.--', label=models_nm[i],
                 zorder=2)

    plt.xlabel('Prediction time step', fontsize=label_sz)
    plt.ylabel('R2 score', fontsize=label_sz)
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    if type(xticks_freq) == int:
        plt.xticks(kernel[xticks_start::xticks_freq],
                   x_ticks[xticks_start::xticks_freq],
                   fontsize=tick_sz)
    else:
        plt.xticks(kernel, x_ticks, fontsize=tick_sz)

    plt.yticks(fontsize=tick_sz)
    plt.legend()

    plt.savefig(str(path)+'/'+args['DATASET']+'_R2_future.png',
                bbox_inches='tight')

    plt.close()
    plt.show()

def plot_R2_vs_features(args, df_compile, path, figsize=(3.22, 3.22), dpi=200,
                        delta_p=0.2, ylabel=None, xlims=None, ylims=None,
                        limit_proportion=0.02, model_names=None,
                        x_ticks=None):
    """
    Function to plot median of R2 vs features for the methods.

    Parameters
    ----------
    args : dict
        Dictionary with init file informations
    df_compile : pandas.DataFrame
        Data Frame with minimum, median average, maximum and deviation of
    	the metrics for the models over the n-th initializations.
    path : str or Pathlib.Path
        Where to save the figure.
    figsize : tuple, optional
        Size of the figure in inches. The default is (3.22, 3.22).
    dpi : int, optional
        Dots per inches. The default is 200.
    ylabel : string or None, optional
        Y-label string. The default is None.
    xlims : tuple or None, optional
        Lower and upper limit of x-axis. The default is None.
    ylims : tuple or None, optional
        Lower and upper limit of y-axis. The default is None.
    limit_proportion : floating, optional
        Width proportion for x and y-axis. The default is 0.02.
    model_names : tuple or None, optional
        1d vector listing the model's name. The default is None.
    x_ticks : None or list-like iterable
        1d vector listing the ticks of x-axis. The default is None.

    Returns
    -------
    None.

    """
    models_nm = np.array(list(df_compile['models']))
    col_l = list(df_compile.columns)
    using = []
    if type(x_ticks) == type(None):
        x_ticks_r = []

    c = 1
    for i in range(len(col_l)):
        if (col_l[i][:8] == 'Test_R2_')&(col_l[i] != 'Test_R2_median'):
            if (col_l[i][8:10] != 't+')&(col_l[i][-7:] == '_median'):
                using.append(col_l[i])
                if type(x_ticks) == type(None):
                    x_ticks_r.append(col_l[i][8:-7])

                c += 1

    if type(x_ticks) == type(None):
        x_ticks = x_ticks_r

    n_feats = len(using)
    kernel = np.arange(1, n_feats+1, 1)
    data = df_compile.loc[:, using].to_numpy()
    if type(model_names) == type(None):
        model_names = df_compile['models'].to_numpy().astype(str)

    n_models = len(model_names)
    model_p_d = np.linspace(-0.5+delta_p, 0.5-delta_p, n_models)
    positions = np.arange(n_feats)
    if type(xlims) == type(None):
        delta = (n_feats-1+model_p_d[-1]+model_p_d[0])*limit_proportion
        xlims = (model_p_d[0]-delta, n_feats-1+model_p_d[-1]+delta)

    if type(ylims) == type(None):
        delta = (np.max(data)-np.min(data))*limit_proportion
        ylims = (np.min(data)-delta, np.max(data)+delta)

    plt.figure(figsize=figsize, dpi=dpi)
    plt.grid(True, zorder=1)
    for i in range(n_models):
        plt.bar(positions+model_p_d[i], data[i], zorder=2, width=0.1,
                label=model_names[i])

    plt.legend()
    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(positions, x_ticks, fontsize=10, rotation=90)
    plt.yticks(fontsize=14)
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    plt.savefig(str(path)+'/'+args['DATASET']+'_R2_features.png',
                bbox_inches='tight')

    plt.close()
    plt.show()

#=============================================================================
# main
#=============================================================================

if __name__ == "__main__":
    # Load configuration
    config = configparser.ConfigParser()
    config.read('compile_runs.ini')

    # Parse arguments
    args = {
        'PATH': config['DATA'].get('PATH'),
        'DATASET': config['DATA'].get('DATASET'),
        'NB_RUNS': config['MODELE'].getint('NB_RUNS'),
        'COMPILE_RUNS': config['COMPILATION'].getboolean('COMPILE_RUNS'),
        'METRICS_TO_PLOT': config['COMPILATION'].get('METRICS_TO_PLOT').split(','),
        'PLOT_R2_TIME': config['COMPILATION'].getboolean('PLOT_R2_TIME'),
        'TIME_STEP': config['COMPILATION'].get('TIME_STEP'),
        'PLOT_R2_FEATURE': config['COMPILATION'].getboolean('PLOT_R2_FEATURE')
    }

    main(args)
