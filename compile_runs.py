import os
import numpy as np
import pandas as pd
import configparser
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from functions_plot import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile ML results into plots and tables"
    )

    parser.add_argument(
        "--DATASET", type=str, required=True,
        help="Dataset name (used in file naming)"
    )

    parser.add_argument(
        "--PATH", type=Path, required=True,
        help="Directory where results are stored"
    )

    return parser.parse_args()

def main(args):
    path = Path(args['PATH'])
    compiled_file = path / "performance_compiled.csv"

    # Compile runs
    results = []
    for i in range(args['NB_RUNS']):
        df = pd.read_csv(path / str(i) / 'performance.csv')
        results.append(df.values[:, 1:])

    metrics = df.columns[1:].to_numpy()
    models = df.values[:, 0]
    results = np.array(results, dtype=float)

    df_compile = compute_compile(results, models, metrics)
    df_compile.to_csv(compiled_file, index=False)

    # Save subset for paper table
    cols = ["Train_pRMSE_average", "Valid_pRMSE_average", "Test_pRMSE_average",
            "Train_MASE_average", "Valid_MASE_average", "Test_MASE_average",
            "Train_R2_average", "Valid_R2_average", "Test_R2_average"]
    df_compile[cols].to_csv(path / "performance_compiled_paper.csv", index=False)

    # Plot metrics if requested
    for metric in filter(None, args['METRICS_TO_PLOT']):
        plot_bar(args, df_compile, metric, path,
                 figsize=(6.44, 6.44), dpi=200,
                 ylabel='Mean Squared Error',
                 xlims=(-0.4, 5.4), ylims=(1.00, 3.25),
                 limit_proportion=0.05,
                 model_names=('Last', 'Linear', 'Dense', 'CNN 1D',
                              'LSTM', 'Transformer'))

    # Plot R2 vs time if requested
    if args['PLOT_R2_TIME']:
        dataset_params = {
            'gnss': dict(figsize=(6.44, 6.44)),
            'seismicity': dict(figsize=(6.44, 6.44)),
            'both': dict(figsize=(6.44, 6.44))
        }
        if args['DATASET'] in dataset_params:
            plot_R2_vs_time(args, df_compile, path, dpi=200,
                            tick_sz=10, label_sz=12,
                            **dataset_params[args['DATASET']])

    # Plot R2 vs features if requested
    if args['PLOT_R2_FEATURE']:
        fnames_map = {
            'gnss': [
                '5 days DSRG-SNEG', '10 days DSRG-SNEG', '30 days DSRG-SNEG',
                '5 days BOMG-DSRG', '10 days BOMG-DSRG', '30 days BOMG-DSRG',
                '5 days BOMG-BORG', '10 days BOMG-BORG', '30 days BOMG-BORG',
                '5 days BORG-SNEG', '10 days BORG-SNEG', '30 days BORG-SNEG',
                '5 days BOMG-DERG', '10 days BOMG-DERG', '30 days BOMG-DERG',
                '5 days DERG-SNEG', '10 days DERG-SNEG', '30 days DERG-SNEG',
                '5 days BORG-DSRG', '10 days BORG-DSRG', '30 days BORG-DSRG',
                '5 days BORG-DERG', '10 days BORG-DERG', '30 days BORG-DERG',
                '5 days DERG-DSRG', '10 days DERG-DSRG', '30 days DERG-DSRG',
                '5 days BOMG-SNEG', '10 days BOMG-SNEG', '30 days BOMG-SNEG'
            ],
            'seismicity': [
                'Daily seismicity', '5 days grad.',
                '10 days grad.', '30 days grad.'
            ],
            'both': [
                '5 days DSRG-SNEG', '10 days DSRG-SNEG', '30 days DSRG-SNEG',
                '5 days BOMG-DSRG', '10 days BOMG-DSRG', '30 days BOMG-DSRG',
                '5 days BOMG-BORG', '10 days BOMG-BORG', '30 days BOMG-BORG',
                '5 days BORG-SNEG', '10 days BORG-SNEG', '30 days BORG-SNEG',
                '5 days BOMG-DERG', '10 days BOMG-DERG', '30 days BOMG-DERG',
                '5 days DERG-SNEG', '10 days DERG-SNEG', '30 days DERG-SNEG',
                '5 days BORG-DSRG', '10 days BORG-DSRG', '30 days BORG-DSRG',
                '5 days BORG-DERG', '10 days BORG-DERG', '30 days BORG-DERG',
                '5 days DERG-DSRG', '10 days DERG-DSRG', '30 days DERG-DSRG',
                '5 days BOMG-SNEG', '10 days BOMG-SNEG', '30 days BOMG-SNEG',
                '5 day grad. daily seismicity',
                '10 day grad. daily seismicity',
                '30 day grad. daily seismicity',
                'daily seismicity'
            ]
        }
        if args['DATASET'] in fnames_map:
            plot_R2_vs_features(args, df_compile, path,
                                figsize=(9.66, 3.22) if args['DATASET'] != 'seismicity' else (6.44, 3.22),
                                dpi=200,
                                delta_p=0.05 if args['DATASET'] != 'seismicity' else 0.15,
                                ylabel='R$^2$ score',
                                limit_proportion=0.02 if args['DATASET'] != 'seismicity' else 0.04,
                                x_ticks=fnames_map[args['DATASET']])


# -------------------- Helper functions -------------------- #

def compute_compile(results, models, metrics):
    """Summarize runs into min/median/avg/max/std per model."""
    return pd.DataFrame({
        'models': models,
        **{f"{m}_{stat}": func(results[:, :, i], axis=0)
           for i, m in enumerate(metrics)
           for stat, func in {
               'minimum': np.min, 'median': np.median,
               'average': np.mean, 'maximum': np.max,
               'deviation': np.std}.items()}
    })


# -------------------- Main -------------------- #

if __name__ == "__main__":

    args = parse_args()
    print(f"Dataset: {args.DATASET}")
    print(f"Path: {args.PATH}")

    config = configparser.ConfigParser()
    config.read('compile_runs.ini')

    config_run = configparser.ConfigParser()
    config_run.read(Path(args.PATH) / f'{args.DATASET}.ini')

    args_for_main = {
        'PATH': args.PATH,
        'DATASET': args.DATASET,
        'NB_RUNS': config_run['MODEL'].getint('NB_RUNS'),
        'METRICS_TO_PLOT': config['COMPILATION'].get('METRICS_TO_PLOT').split(','),
        'PLOT_R2_TIME': config['COMPILATION'].getboolean('PLOT_R2_TIME'),
        'TIME_STEP': config['COMPILATION'].get('TIME_STEP'),
        'PLOT_R2_FEATURE': config['COMPILATION'].getboolean('PLOT_R2_FEATURE')
    }

    main(args_for_main)
