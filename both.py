import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from pathlib import Path
import shutil

from functions_data import *
from functions_model import *
from functions_plot import *
from functions_metrics import *

# Check if GPU is available
print('Available GPUs:', tf.config.list_physical_devices('GPU'))

# To use Python `lambda` function in model layer
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

    # Load and preprocess seismic data
    df, date_time = load_and_preprocess_data(DATA_PATH, type_data="both")
    feature_names = np.array(list(df.columns))
    to_in_models = feature_names.copy()
    to_out = feature_names.copy()

    # Split data into training, validation, and test sets
    train_df, valid_df, test_df, split_infos = split_data(df, date_time, OUTPUT_PATH, args['SAVE_REPARTITION_IMG'], type_data="both")

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

if __name__ == "__main__":
    
    # load configuration
    args, NB_RUNS = get_config('both.ini')

    # Copy the configuration file for the record in the working directory
    ROOT_PATH = Path(os.path.dirname('./'))
    DEST_PATH = ROOT_PATH / args['PATH']
    DEST_PATH.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT_PATH / 'both.ini', DEST_PATH / 'both.ini')

    # Run multiple configurations
    for run in range(NB_RUNS):
        args['SEED'] = run
        main(args)
