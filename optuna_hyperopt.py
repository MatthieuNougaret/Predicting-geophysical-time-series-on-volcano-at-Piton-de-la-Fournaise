import os, random

# --- FORCE CPU + deterministic ops BEFORE TF IMPORT ---
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

# Optional: make execution single-threaded
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import numpy as np
import tensorflow as tf
import keras

# threading determinism (sometimes env vars don’t take effect)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Enable deterministic ops (TF 2.10+)
tf.config.experimental.enable_op_determinism()

# Set parameters
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

from sklearn.metrics import mean_squared_error, r2_score

import logging
import sys
import argparse

# Check if GPU is available
print('Available GPUs:', tf.config.list_physical_devices('GPU'))

# To use Python `lambda` function in model layer
keras.config.enable_unsafe_deserialization()

# For hyperparameter tuning
import optuna

from functions_data import *
from functions_model import *
from functions_plot import *
from functions_metrics import *
from train import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="OPTUNA HYPERPARAMETER OPT"
    )

    parser.add_argument(
        "--DATASET", type=str, required=True,
        help="Dataset name (used in file naming): gnss, seismicity or both"
    )

    parser.add_argument(
        "--NTRIALS", type=str, required=True,
        help="Number of trials"
    )

    return parser.parse_args()

# ==============================
# 1. Define Objective
# ==============================
def objective(trial):
    # Hyperparams
    input_width = trial.suggest_int("window_past", 1, 30)
    out_steps = 5

    # Create datasets (per trial, since input_width changes)
    X_train, y_train = create_dataset(train_df, input_width, out_steps, shuffle=True)
    X_valid, y_valid = create_dataset(valid_df, input_width, out_steps)

    num_features = X_train.shape[2]

    model = build_autoregressive_transformer(
        input_shape=(input_width, num_features),
        head_size=trial.suggest_int("head_size", 4, 80, step=4),
        num_heads=trial.suggest_int("num_heads", 1, 16, step=2),
        ff_dim=trial.suggest_int("ff_dim", 8, 256, step=32),
        num_transformer_blocks=trial.suggest_int("num_transformer_blocks", 1, 3),
        mlp_units=[trial.suggest_int("mlp_units", 64, 512, step=64)],
        out_steps=out_steps,
        mlp_dropout=trial.suggest_float("mlp_dropout", 0.0, 0.5, step=0.05),
        norm_type=trial.suggest_categorical("norm_type", ["post", "pre"]),
    )

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    lr_decay = trial.suggest_float("lr_decay", 0.5, 0.95)
    batch_size = trial.suggest_int("batch_size", 32, 512, step=32)

    loss_huber = tf.keras.losses.Huber(
        delta=trial.suggest_float("huber_delta", 0.5, 3.0, step=0.5)
    )

    # Compile
    model.compile(
        loss=loss_huber,
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=["mse", "mae"],
    )

    # Callbacks
    r_lro = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=lr_decay, patience=5, min_lr=1e-6, verbose=0
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=0
    )

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=500,
        batch_size=batch_size,
        verbose=0,
        callbacks=[r_lro, early_stopping],
    )

    # Predict
    y_pred = model(X_valid, training=False).numpy()

    # Return mase and peak_weight_mse using available functions
    return mase(y_valid, y_pred), peak_weighted_rmse(y_valid, y_pred)  # multi-objective: directions=["minimize", "minimize"]

# Set up the main script

if __name__ == "__main__":

    # ==============================
    # Get command line arguments
    # ==============================
    args_cl = parse_args()
    print(f"Dataset: {args_cl.DATASET}")

    # ==============================
    # Load & preprocess
    # ==============================
    df, date_time = load_and_preprocess_data("./data/data_in.xlsx", type_data=args_cl.DATASET)
    feature_names = np.array(list(df.columns))

    train_df, valid_df, test_df, _ = split_data(df, date_time, "./", False, type_data=args_cl.DATASET)

    train_df, valid_df, test_df, scaler = standardize_data(
        train_df, valid_df, test_df, use_gnss_asinh=False
    )

    # keep scaler for later inverse-transform
    target_scaler = scaler   # if you only want to inverse-transform target cols, fit separately here!

    # ==============================
    # Create the study
    # ==============================
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "transformer-{}-study".format(args_cl.DATASET)  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(directions=['minimize', 'minimize'], 
                                sampler=optuna.samplers.TPESampler(),
                                study_name=study_name, 
                                storage=storage_name, 
                                load_if_exists=True)

    # ===================================
    # Optimize the objective function.
    # ===================================
    study.optimize(objective, n_trials=int(args_cl.NTRIALS), n_jobs=1, show_progress_bar=True)