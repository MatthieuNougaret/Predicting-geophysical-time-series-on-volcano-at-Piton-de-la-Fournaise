# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import zscore

#=============================================================================
# Transform the hourly number of earthquakes to datasets with features
# all normalized.
#=============================================================================

# ================================= FUCTIONS =================================

def log_features(df, target):
	df[target+'_log'] = np.log(df[target].to_numpy().astype('float32')+1)
	return df

def log10_features(df, target):
	df[target+'_log10'] = np.log10(df[target].to_numpy().astype('float32')+1)
	return df

def root_features(df, target):
	df[target+'_root'] = df[target.to_numpy().astype('float32')]**0.5
	return df

def compute_grad(df, target, window):
	"""
	df : pandas.DataFrame
		Dataframe from and for computation of the gradient.
	target : str
		Column targeted.
	window : int
		Taille de la fenetre glissante sur laquelle on calcule le gradient.
	"""
	gradient = np.zeros(len(df[target]), dtype='float32')
	y = df[target].to_numpy().astype('float32')
	kernel = np.arange(0, window+1, 1, dtype='float32')
	for i in range(window, len(y)):
		p = np.polyfit(kernel, y[i-window:i+1], 1)
		gradient[i] = p[0]

	df[target+'_'+str(window)] = gradient
	return df

def compute_roll(df, target, nature, window):
	if nature == 'mean':
		col = target+'_rolling_mean_'+str(window)
		df[col] = (df[target].rolling(window=window
									  ).mean()).to_numpy().astype('float32')
	elif nature == 'std':
		col = target+'_rolling_std_'+str(window)
		df[col] = (df[target].rolling(window=window
									  ).std()).to_numpy().astype('float32')

	return df

def comput_zscore(df, target):
	df[target+'_zscore'] = zscore(df[target]).to_numpy().astype('float32')
	return df

def comput_anomaly(df, target):
	y = np.zeros(len(df[target]), dtype='float32')
	y[df[target].to_numpy() > 2] = 1
	df[target+'_anomaly'] = y
	return df

def compute_intensity(df, target, limits, equivalent):
	class_manual = np.zeros(len(df[target]), dtype='float32')
	for i in range(len(limits)):
		class_manual[df[target] >= limits[i]] = equivalent[i]

	df[target+'_intensity'] = class_manual
	return df

# ============================== CREATE FEATURES =============================

# COMPUTE FEATURES

# data will already have columns: {'date', 'sismicity', 'activity'}
data = pd.read_csv('./SISMO/bulletin_summit_hourly.csv')

# PUT IN DAYS FOR FEATURES COMPUTATION AND PREDICTIONS !

# feature engineering
# comment / uncomment the line before execution to get what you want. You
# can also add lin to compute different features to your heart's content

#data = log10_features(data, 'sismicity')
#data = root_features(data, 'sismicity')

# Setting a rolling window size
wind_roll_size = 60*24
if type(wind_roll_size) == int:
	if wind_roll_size > 1:
		# Calculating rolling mean and standard deviation for seismic
		# activities
		data = compute_roll(data, 'sismicity', 'mean', wind_roll_size)
		#data = compute_roll(data, 'sismicity_log10', 'mean',
		#					 wind_roll_size[i])
		#data = compute_roll(data, 'sismicity_root', 'mean',
		#					 wind_roll_size[i])

		data = compute_roll(data, 'sismicity', 'std', wind_roll_size)
		#data = compute_roll(data, 'sismicity_log10', 'std',
		#					 wind_roll_size[i])
		#data = compute_roll(data, 'sismicity_root',
		#					 'std', wind_roll_size[i])

grad_lengths = [3*24, 10*24, 30*24]
if len(grad_lengths) > 0:
	for i in range(len(grad_lengths)):
		data = compute_grad(data, 'sismicity', grad_lengths[i])
		#data = compute_grad(data, 'sismicity_log10', grad_lengths[i])
		#data = compute_grad(data, 'sismicity_root', grad_lengths[i])

		# you can also create a new loop to have different grad_lengths for
		# different features
		#data = compute_grad(data, 'sismicity_rolling_mean_xxx',
		#					 grad_lengths[i])
		#data = compute_grad(data, 'sismicity_log10_rolling_mean_xxx',
		#					 grad_lengths[i])
		#data = compute_grad(data, 'sismicity_root_rolling_std_xxx',
		#					 grad_lengths[i])

# Compute zscore
#data = comput_zscore(data, 'sismicity')
#data = comput_zscore(data, 'sismicity_log10')
#data = comput_zscore(data, 'sismicity_root')

# Identifying potential anomalies based on Z-score (absolute value > 2)
#data = comput_anomaly(data, 'sismicity_zscore')
#data = comput_anomaly(data, 'sismicity_log10_zscore')
#data = comput_anomaly(data, 'sismicity_root_zscore')

# We may add a discrete quantification of the level of seismic activity
#data = compute_intensity(data, 'sismicity', [5, 10, 100, 500], [1, 2, 3, 4])
#data = compute_intensity(data, 'sismicity_log10', [0.7, 1, 2, 2.69],
#						  [1, 2, 3, 4])
#data = compute_intensity(data, 'sismicity_root', [2.3, 3.2, 10, 22.37],
#						  [1, 2, 3, 4])


# We remove rows with NaNs coming from rolling_mean and rolling_std and
# computation of their gradient
if type(wind_roll_size) == int:
	if wind_roll_size > 1:
		data = data[wind_roll_size+np.max(grad_lengths):]

else:
	data = data[np.max(grad_lengths):]

data = data.reset_index(drop=True)

# Drop the columns you dont wants anymore
to_drop = []
for i in to_drop:
	data = data.drop(columns=i)

# CUT AND NORMALISATION OF TRAIN VALID TEST 
# Repartition
valid_fraction = 0.15
test_fraction = 0.15
train_fraction = 1-valid_fraction-test_fraction

nb_data = len(data)
cut_train = int(train_fraction*nb_data)
cut_valid = int((train_fraction+valid_fraction)*nb_data)

# Data Frames that will be normalised
ds_train = data.loc[:cut_train, :].copy()
ds_valid = data.loc[cut_train:cut_valid, :].copy()
ds_test = data.loc[cut_valid:, :].copy()

# Data Frames that will not be normalised
ds_train_raw = ds_train.copy()
ds_valid_raw = ds_valid.copy()
ds_test_raw = ds_test.copy()

keep_in_raw = ['date', 'sismicity', 'activity']
for i in list(ds_train_raw.columns):
	if i not in keep_in_raw:
		ds_train_raw = ds_train_raw.drop(columns=i)
		ds_valid_raw = ds_valid_raw.drop(columns=i)
		ds_test_raw = ds_test_raw.drop(columns=i)

# Saving the raw Data Frames
ds_train_raw.to_csv('./SISMO/ds_train_raw.csv', index=False)
ds_valid_raw.to_csv('./SISMO/ds_valid_raw.csv', index=False)
ds_test_raw.to_csv('./SISMO/ds_test_raw.csv', index=False)

# =============================== NORMALIZATION ==============================
# Empty quote means no normalisation. The first and third columns MUST NOT
# BE NORMALIZED ! They are : date and activity type.
# 'max' : divide by np.max of i-th column;
# 'std' : divide by np.std of i-th column;
method = ['', 'max', '', 'max', 'max', 'std', 'std', 'std']
columns = list(data.columns)
for i in range(len(columns)):
	if method[i] == 'max':
		maxi = np.max(ds_train[columns[i]])
		ds_train[columns[i]] = ds_train[columns[i]]/maxi
		ds_valid[columns[i]] = ds_valid[columns[i]]/maxi
		ds_test[columns[i]] = ds_test[columns[i]]/maxi

	elif method[i] == 'min-max':
		mini = np.min(ds_train[columns[i]])
		maxi = np.max(ds_train[columns[i]]-mini)
		ds_train[columns[i]] = (ds_train[columns[i]]-mini)/maxi
		ds_valid[columns[i]] = (ds_valid[columns[i]]-mini)/maxi
		ds_test[columns[i]] = (ds_test[columns[i]]-mini)/maxi

	elif method[i] == 'std':
		devi = np.std(ds_train[columns[i]])
		ds_train[columns[i]] = ds_train[columns[i]]/devi
		ds_valid[columns[i]] = ds_valid[columns[i]]/devi
		ds_test[columns[i]] = ds_test[columns[i]]/devi

ds_train.to_csv('./SISMO/ds_train.csv', index=False)
ds_valid.to_csv('./SISMO/ds_valid.csv', index=False)
ds_test.to_csv('./SISMO/ds_test.csv', index=False)
