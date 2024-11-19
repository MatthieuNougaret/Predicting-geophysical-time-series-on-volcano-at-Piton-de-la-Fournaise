# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

#=============================================================================
# Creates the indicies which will be called during the crearion of datasets.
#=============================================================================

# ================================= FUCTIUNS =================================

def create_dataset(path_to_df, path_to_indexes, path_name_to_save):
	# load the dataframe
	df = pd.read_csv(path_to_df)
	timing = np.array(df['date'], dtype='datetime64[h]')
	activity = np.array(df['activity'], dtype='uint8')
	df = df.drop(columns='date')
	df = df.drop(columns='activity')

	# load the indicies
	indexes = np.load(path_to_indexes, allow_pickle=True)[0]

	# Create and save X and y
	np.save(path_name_to_save+'_X.npy',
			(df.to_numpy()[indexes['X']]).astype('float32'))

	np.save(path_name_to_save+'_y.npy',
			(df.to_numpy()[indexes['y']]).astype('float32'))

	# Create and save time for X and y
	np.save(path_name_to_save+'_t_X.npy', timing[indexes['X']])

	np.save(path_name_to_save+'_t_y.npy', timing[indexes['y']])

	# Create and save activity and time for activity
	np.save(path_name_to_save+'_activity.npy',
			np.concatenate((activity[indexes['X']],
							activity[indexes['y']]), axis=1))

	np.save(path_name_to_save+'_t_activity.npy',
			np.concatenate((timing[indexes['X']],
							timing[indexes['y']]), axis=1))



# ============================== CREATE DATASETS =============================

df_path = ['./SISMO/ds_train.csv',
		   './SISMO/ds_valid.csv',
		   './SISMO/ds_test.csv',
		   './SISMO/ds_train_raw.csv',
		   './SISMO/ds_valid_raw.csv',
		   './SISMO/ds_test_raw.csv']

ix_path = ['./SISMO/index_train.npy',
		   './SISMO/index_valid.npy',
		   './SISMO/index_test.npy',
		   './SISMO/index_train.npy',
		   './SISMO/index_valid.npy',
		   './SISMO/index_test.npy']

to_path = ['./SISMO/train',
		   './SISMO/valid',
		   './SISMO/test',
		   './SISMO/train_raw',
		   './SISMO/valid_raw',
		   './SISMO/test_raw']

for i in range(len(df_path)):
	print('Creating files from: '+df_path[i])
	create_dataset(df_path[i], ix_path[i], to_path[i])
