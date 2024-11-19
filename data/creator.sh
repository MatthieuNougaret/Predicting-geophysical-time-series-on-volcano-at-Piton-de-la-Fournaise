#!/bin/bash

FILE=./SISMO/bulletin_summit_hourly.csv
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else
    echo "Creation of $FILE"
    python create_hourly.py
fi

FILE=./SISMO/ds_train.csv
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else
    echo "Creation of $FILE"
    python create_features.py
fi

FILE=./SISMO/index_train.npy
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else
    echo "Creation of $FILE"
    python create_indexes.py
fi

FILE=./SISMO/train_X.npy
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else
    echo "Creation of $FILE"
    python create_dataset.py

    echo "Removing files used for indexes"
    rm ./SISMO/index_train.npy ./SISMO/index_valid.npy ./SISMO/index_test.npy 
fi
