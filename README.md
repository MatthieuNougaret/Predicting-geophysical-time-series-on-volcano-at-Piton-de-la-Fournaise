# ML-forecast-PitonFournaise

This is the GitHub repository of the paper: "Predicting geophysical time series on volcano at Piton de la Fournaise".

# Training models

## The configuration files

The files seismicity.ini, gnss.ini and both.ini contain the settings for data paths, windowing, model hyperparameters,
learning rates, and metadata saving options for experiments using the GNSS and seismicity data
from Piton de la Fournaise volcano.

## Training script : `train.py`

To train the models, change the configuration files and then run the `train.py` script via the command line:

    $ python train.py --PATH OUTPUT_PATH --DATASET dataset_type

where `OUTPUT_PATH` is the path where you want the models to be saved, and `dataset_type` is the type of dataset you want to treat:
- gnss: GNSS dataset
- seismicity: the daily number of VT events
- both: the combination of the GNSS and seismicity dataset

## Tuning model hyperparameters : `optuna_hyperopt.py`

The script `optuna_hyperopt.py` uses [Optuna](https://optuna.readthedocs.io/en/stable/index.html#) to tune the model hyperparameters. This automatic tuning is only done for the autoregressive Transformer model. Hyperparameter of other models were tuned by hand. If wanted, it is easy to change this script to tune the hyperparameters of another model, just replace the creation of the Transformer model by that of another model available in the source code (see in particular the code in `functions_model.py`).

To launch the search for hyperparameters, do in a terminal:

    $ python optuna_hyperopt.py --DATASET dataset_type --NTRIALS 50

with `dataset_type` as previously defined. You can increase `NTRIALS` to search for more solutions. Pruning is activated, so not-promising trials will be stopped early. To look at the results, we use optuna-dashboard, see the [documentation of Optuna](https://optuna.readthedocs.io/en/stable/index.html).

## Result compilation

By default, several runs are performed to have access to the variability of the results inherent to the stochasticity of artificial neural network training. This is set by the `NB_RUNS` variable in the .ini files.

To combine results from those runs, please run the command

    $ python compile_runs.py --PATH OUTPUT_PATH --DATASET dataset_type

with `OUTPUT_PATH` and `dataset_type` as previously defined for training.

## Observing predictions

The
