import matplotlib.pyplot as plt
import numpy as np

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

# -------------------- Plot Utilities -------------------- #

def _apply_axis_limits(data, n, limit_proportion, xlims, ylims):
    """Helper to compute axis limits if not given."""
    if xlims is None:
        delta = n * limit_proportion
        xlims = (-delta, n - 1 + delta)

    if ylims is None:
        delta = (np.max(data) - np.min(data)) * limit_proportion
        ylims = (np.min(data) - delta, np.max(data) + delta)

    return xlims, ylims


def _finalize_and_save(path, dataset, name, figsize, dpi):
    """Helper to save and close matplotlib figure."""
    plt.savefig(path / f"{dataset}_{name}.pdf", bbox_inches='tight', dpi=dpi)
    plt.close()


# -------------------- Plot Functions -------------------- #

def plot_bar(args, df_compile, metric_name, path,
             figsize=(3.22, 3.22), dpi=200, ylabel=None,
             xlims=None, ylims=None, limit_proportion=0.02,
             model_names=None):
    """Bar plot for a given metric."""
    data = df_compile[metric_name].to_numpy(dtype=float)
    model_names = model_names or df_compile['models'].astype(str).to_numpy()

    n_models = len(model_names)
    xlims, ylims = _apply_axis_limits(data, n_models, limit_proportion, xlims, ylims)

    plt.figure(figsize=figsize, dpi=dpi)
    plt.grid(True, zorder=1)
    plt.bar(np.arange(n_models), data, color='green', width=0.5, zorder=2)

    plt.ylabel(ylabel or metric_name, fontsize=16)
    plt.xticks(np.arange(n_models), model_names, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(*xlims)

    _finalize_and_save(path, args['DATASET'], metric_name, figsize, dpi)


def plot_R2_vs_time(args, 
                    df_compile, 
                    path, 
                    figsize=(3.22, 3.22), 
                    dpi=200,
                    tick_sz=12, 
                    label_sz=12,):
    """Line plot of R² vs prediction time step."""
    models_nm = df_compile['models'].to_numpy()
    cols = [c for c in df_compile.columns if c.startswith("Test_R2_t+") and c.endswith("_median")]
    x_ticks = [f"t+{i+1} {args['TIME_STEP']}" for i in range(len(cols))]

    data = df_compile[cols].to_numpy()
    kernel = np.arange(1, len(cols) + 1)

    plt.figure(figsize=figsize, dpi=dpi)
    plt.grid(True, zorder=1)
    for row, model in zip(data, models_nm):
        plt.plot(kernel, row, '.--', label=model, zorder=2)

    plt.xlabel('Prediction time step', fontsize=label_sz)
    plt.ylabel('R$^{2}$ score', fontsize=label_sz)
    #plt.ylim(*ylims)

    plt.xticks(kernel, x_ticks, fontsize=tick_sz)

    plt.yticks(fontsize=tick_sz)
    plt.legend(loc="best", ncol=2)

    _finalize_and_save(path, args['DATASET'], "R2_future", figsize, dpi)


def plot_R2_vs_features(args, df_compile, path, figsize=(3.22, 3.22), dpi=200,
                        delta_p=0.2, ylabel=None, ylims=None,
                        limit_proportion=0.02, x_ticks=None):
    """Bar plot of R² vs features."""
    models_nm = df_compile['models'].astype(str).to_numpy()
    cols = [c for c in df_compile.columns
            if c.startswith("Test_R2_") and c.endswith("_median") and not c.startswith("Test_R2_t+")
            and not c == "Test_R2_median"]
    if x_ticks is None:
        x_ticks = [c[8:-7] for c in cols]

    data = df_compile[cols].to_numpy()
    n_feats, n_models = len(cols), len(models_nm)

    offsets = np.linspace(-0.5 + delta_p, 0.5 - delta_p, n_models)
    _, ylims = _apply_axis_limits(data, n_feats, limit_proportion, None, ylims)

    plt.figure(figsize=figsize, dpi=dpi)
    plt.grid(True, zorder=1)
    for i, model in enumerate(models_nm):
        plt.bar(np.arange(n_feats) + offsets[i], data[i], width=0.1, label=model, zorder=2)

    plt.legend(loc="best", ncol=2)
    plt.ylabel(ylabel or "R$^2$ score", fontsize=16)
    plt.xticks(np.arange(len(x_ticks)), x_ticks, fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    plt.xlim([-0.5, n_feats-0.5])
    plt.ylim(*ylims)

    _finalize_and_save(path, args['DATASET'], "R2_features", figsize, dpi)
