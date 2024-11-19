# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

#=============================================================================
# Creates the indicies which will be called during the crearion of datasets.
#=============================================================================

# ================================= FUCTIUNS =================================

def create_index(df, seq_len, futur_targ):
	n_time = len(df)
	if (n_time+seq_len+futur_targ[1]+1) < 2147483646:
		dtype = 'int32'
	else:
		dtype = int

	n_samples = np.arange(0, n_time-seq_len-futur_targ[1]+1, 1,
						  dtype=dtype)[:, np.newaxis]

	length_x = np.arange(0, seq_len, 1, dtype=dtype)
	length_y = np.arange(seq_len+futur_targ[0], seq_len+futur_targ[1], 1,
						 dtype=dtype)

	indexes = np.array([{'X':n_samples+length_x, 'y':n_samples+length_y}],
						dtype=object)

	return indexes

# ============================= GENERATE INDEXES =============================

# Files for wich the indexes will be generated 
df_train = pd.read_csv('./SISMO/ds_train.csv')
df_valid = pd.read_csv('./SISMO/ds_valid.csv')
df_test = pd.read_csv('./SISMO/ds_test.csv')

# length of the inputs
in_len = 60*24
# limits to the hours to target in the futur
target = [0, 24]

idx_train = create_index(df_train, in_len, target)
idx_valid = create_index(df_valid, in_len, target)
idx_test = create_index(df_test, in_len, target)

# save the indexes
np.save('./SISMO/index_train.npy', idx_train)
np.save('./SISMO/index_valid.npy', idx_valid)
np.save('./SISMO/index_test.npy', idx_test)