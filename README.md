# ML-forecast-PitonFournaise

This is the GitHub repository of the paper: "Predicting geophysical time series on volcano at Piton de la Fournaise".

This repository contains machine learning models for forecasting volcanic activity using GNSS baseline gradients and seismic data from Piton de la Fournaise volcano. The models predict multi-step time series up to 5 days ahead using various architectures including Linear, Dense, CNN, LSTM, and Transformer models.

## Data

The input data (`data/data_in.xlsx`) contains:
- **GNSS data**: Baseline gradient measurements between stations (DSRG-SNEG, BOMG-DSRG, etc.)  
- **Seismicity data**: Daily volcano-tectonic (VT) event counts (log₁₀ transformed)
- **Combined data**: Joint GNSS and seismicity features

Features are processed using rolling gradient computations over 5, 10, and 30-day windows to capture temporal trends.

## Training models

### The configuration files

The files [`seismicity.ini`](seismicity.ini), [`gnss.ini`](gnss.ini) and [`both.ini`](both.ini) contain the settings for data paths, windowing, model hyperparameters, learning rates, and metadata saving options for experiments using the GNSS and seismicity data from Piton de la Fournaise volcano.

### Training script : `train.py`

To train the models, change the configuration files and then run the [`train.py`](train.py) script via the command line:

```bash
$ python train.py --PATH OUTPUT_PATH --DATASET dataset_type
```

where `OUTPUT_PATH` is the path where you want the models to be saved, and `dataset_type` is the type of dataset you want to treat:
- `gnss`: GNSS dataset
- `seismicity`: the daily number of VT events  
- `both`: the combination of the GNSS and seismicity dataset

### Tuning model hyperparameters : `optuna_hyperopt.py`

The script [`optuna_hyperopt.py`](optuna_hyperopt.py) uses [Optuna](https://optuna.readthedocs.io/en/stable/index.html#) to tune the model hyperparameters. This automatic tuning is only done for the autoregressive Transformer model. Hyperparameter of other models were tuned by hand. If wanted, it is easy to change this script to tune the hyperparameters of another model, just replace the creation of the Transformer model by that of another model available in the source code (see in particular the code in [`functions_model.py`](functions_model.py)).

To launch the search for hyperparameters, do in a terminal:

```bash
$ python optuna_hyperopt.py --DATASET dataset_type --NTRIALS 50
```

with `dataset_type` as previously defined. You can increase `NTRIALS` to search for more solutions. Pruning is activated, so not-promising trials will be stopped early. To look at the results, we use optuna-dashboard, see the [documentation of Optuna](https://optuna.readthedocs.io/en/stable/index.html).

### Result compilation

By default, several runs are performed to have access to the variability of the results inherent to the stochasticity of artificial neural network training. This is set by the `NB_RUNS` variable in the .ini files.

To combine results from those runs, please run the command

```bash
$ python compile_runs.py --PATH OUTPUT_PATH --DATASET dataset_type
```

with `OUTPUT_PATH` and `dataset_type` as previously defined for training.

### Observing predictions

The [`plot_prediction.ipynb`](plot_prediction.ipynb) notebook contains visualization examples showing model predictions for different datasets. It demonstrates how to:
- Load trained models and select the best performing run
- Generate multi-step forecasts on test data  
- Plot observed vs predicted time series with forecast horizons
- Compare performance across different feature types

## Model Architectures

The repository implements six different forecasting models:

1. **Last (Baseline)**: Repeats the last observed value
2. **Linear**: Simple linear regression on flattened input
3. **Dense**: Multi-layer perceptron with hidden layers
4. **Conv1D**: 1D convolutional neural network
5. **LSTM**: Long Short-Term Memory recurrent network
6. **Transformer**: Autoregressive transformer with positional encoding

## Evaluation Metrics

Models are evaluated using:
- **R² score**: Coefficient of determination for prediction accuracy
- **MASE**: Mean Absolute Scaled Error for scale-independent comparison
- **Peak-weighted RMSE**: Emphasizes errors during high-activity periods

## Key Files

- [`train.py`](train.py): Main training script
- [`functions_data.py`](functions_data.py): Data loading, preprocessing and dataset creation
- [`functions_model.py`](functions_model.py): Model architectures and training/evaluation
- [`functions_metrics.py`](functions_metrics.py): Custom evaluation metrics
- [`functions_plot.py`](functions_plot.py): Visualization utilities
- [`compile_runs.py`](compile_runs.py): Aggregates results across multiple runs
- [`optuna_hyperopt.py`](optuna_hyperopt.py): Hyperparameter optimization

## Usage Example

```bash
# Train models on GNSS data
python train.py --PATH ./models_gnss --DATASET gnss

# Compile results from multiple runs  
python compile_runs.py --PATH ./models_gnss --DATASET gnss
```

## Requirements

- Python 3.8+
- TensorFlow/Keras 2.10+
- scikit-learn
- pandas, numpy
- matplotlib
- optuna (for hyperparameter tuning)
- openpyxl (for Excel file reading)

## License

This project is licensed under the MIT License.