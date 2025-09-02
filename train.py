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

import argparse
from pathlib import Path
import shutil
import joblib
from functions_data import *
from functions_model import *
from functions_plot import *
from functions_metrics import *

# Check if GPU is available
print('Available GPUs:', tf.config.list_physical_devices('GPU'))

# To use Python `lambda` function in model layer
keras.config.enable_unsafe_deserialization()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the datasets and train the models"
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


def generate_dataset(args):
    """
    Generates and preprocesses datasets for training, validation, and testing from geophysical time series data.
    This function performs the following steps:
    1. Loads and preprocesses the raw data from the specified path.
    2. Splits the data into training, validation, and test sets.
    3. Standardizes the data according to the dataset type.
    4. Creates input/output datasets for model training and evaluation.
    5. Saves relevant information, including hyperparameters, scaler, and datasets, to disk.
    Args:
        args (dict): Dictionary containing configuration parameters. Expected keys include:
            - 'DATA_PATH': Path to the raw data directory.
            - 'PATH': Output directory for saving processed data and artifacts.
            - 'DATASET': Type of dataset ('gnss', 'seismicity', or 'both').
            - 'SAVE_REPARTITION_IMG': Boolean flag to save data split images.
            - 'INPUT_WIDTH': Number of input time steps for the model.
            - 'OUT_STEPS': Number of output time steps for the model.
    Returns:
        dict: Dictionary containing processed datasets and feature names:
            - 'X_train': Training input data.
            - 'y_train': Training output data.
            - 'X_valid': Validation input data.
            - 'y_valid': Validation output data.
            - 'X_test': Test input data.
            - 'y_test': Test output data.
            - 'Features': Array of feature names.
    """

    DATA_PATH = Path(args['DATA_PATH'])
    OUTPUT_PATH = Path(args['PATH'])
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    df, date_time = load_and_preprocess_data(DATA_PATH, 
                                             type_data=args['DATASET'])
    feature_names = np.array(list(df.columns))

    # Split data into training, validation, and test sets
    train_df, valid_df, test_df, split_infos = split_data(df, date_time, 
                                                          OUTPUT_PATH, args['SAVE_REPARTITION_IMG'], 
                                                          type_data=args['DATASET'])

    # data time of subsets
    # train_date_time = date_time[train_df.index]
    # valid_date_time = date_time[valid_df.index]
    test_date_time = date_time[test_df.index]

    # Standardize data 
    # this is now all to the same values but was used to test asinh() transformation and scaling.
    # keeping the code as is if we want to test further this option in the future.
    if args['DATASET'] == "gnss" or args['DATASET'] == "both":
        train_df, valid_df, test_df, scaler = standardize_data(train_df, valid_df, test_df,
                                                               use_gnss_asinh=False)
    elif args['DATASET'] == "seismicity":
        train_df, valid_df, test_df, scaler = standardize_data(train_df, valid_df, test_df,
                                                               use_gnss_asinh=False)
    else:
        raise ValueError("DATASET should be set to gnss, seismicity or both")

    # Create datasets
    X_train, y_train = create_dataset(train_df, args['INPUT_WIDTH'], args['OUT_STEPS'], shuffle=True)
    X_valid, y_valid = create_dataset(valid_df, args['INPUT_WIDTH'], args['OUT_STEPS'])
    X_test, y_test, X_test_dates, y_test_dates = create_dataset(test_df, args['INPUT_WIDTH'], args['OUT_STEPS'], dates=test_date_time)

    # saving some hyperparameters
    saving_infos(args, [X_train, y_train, X_valid, y_valid, X_test, y_test],
                 feature_names, split_infos, OUTPUT_PATH)
    
    # Save data and save the scaler
    joblib.dump(scaler, OUTPUT_PATH / "scaler.pkl")
    np.savez_compressed(
        OUTPUT_PATH / "datasets.npz",
        X_train=X_train, y_train=y_train,
        X_valid=X_valid, y_valid=y_valid,
        X_test=X_test, y_test=y_test,
        X_test_dates=X_test_dates,  y_test_dates=y_test_dates,
        Features=feature_names
    )

    return dict(X_train=X_train, y_train=y_train,
                X_valid=X_valid, y_valid=y_valid,
                X_test=X_test, y_test=y_test,
                Features=feature_names)
    
def train(args, data_dict):
    """Trains and evaluates machine learning models using provided data and arguments.

    Args:
        args (dict): Training configuration parameters such as 'PATH' and 'SEED'.
        data_dict (dict): Dictionary containing datasets:
            'X_train': Training features.
            'y_train': Training targets.
            'X_valid': Validation features.
            'y_valid': Validation targets.
            'X_test': Test features.
            'y_test': Test targets.
            'Features': List of feature names.

    Side Effects:
        - Creates output directory for results.
        - Sets random seeds for reproducibility.
        - Trains models and saves them to disk.
        - Evaluates models and saves performance metrics as 'performance.csv' in the output directory.
    """
    
    OUTPUT_PATH = Path(args['PATH']) / str(args['SEED'])
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Set parameters
    SEED = args['SEED']
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    # train
    # train_models(data_dict['X_train'],
    #              data_dict['y_train'],
    #              data_dict['X_valid'],
    #              data_dict['y_valid'], 
    #              args,
    #              data_dict['Features'], 
    #              OUTPUT_PATH)

    # evaluate models
    performance = evaluate_models(data_dict['X_train'],
                                  data_dict['y_train'],
                                  data_dict['X_valid'], 
                                  data_dict['y_valid'],
                                  data_dict['X_test'],
                                  data_dict['y_test'],
                                  args,
                                  data_dict['Features'],
                                  OUTPUT_PATH)

    # Save performance metrics
    performance.to_csv(OUTPUT_PATH / "performance.csv", index=False)

# Set up the main script

if __name__ == "__main__":

    # get command line arguments
    args_cl = parse_args()
    print(f"Dataset: {args_cl.DATASET}")
    print(f"Path: {args_cl.PATH}")

    # ini file name
    ini_file = '{}.ini'.format(args_cl.DATASET)

    # load configuration for the dataset
    args, NB_RUNS = get_config(ini_file)

    # add the command line arguments
    args["DATASET"] = args_cl.DATASET
    args["PATH"] = args_cl.PATH

    # Copy the configuration file for the record in the working directory
    ROOT_PATH = Path(os.path.dirname('./'))
    DEST_PATH = ROOT_PATH / args['PATH']
    DEST_PATH.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT_PATH / ini_file, DEST_PATH / ini_file)

    # generate the dataset
    np.random.seed(42) # we fix the seed for data shuffling of the train set (affects minibatch training)
    data_dict = generate_dataset(args)

    # Train for multiple configurations
    for run in range(NB_RUNS):
        args['SEED'] = run # seed for the model parameters, etc.
        train(args, data_dict)
