# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm

#=============================================================================
# Transform the bulletin to the hourly number of earthquakes from
# 2000-01-01T00:00:00.000 to the last started hour. Consequently, if the last
# earthquake was recorded at T13:00:00.001 the upper bound of the last bin
# would be T14:00:00.000
#=============================================================================

ALL = pd.read_csv('./SISMO/MC3_dump_bulletin.csv', skiprows=1, sep=';')
use_cols = ['#YYYYmmdd HHMMSS.ss', 'Nb(#)', 'Duration']
catalog = ALL.loc[:, use_cols]

date = np.copy(catalog['#YYYYmmdd HHMMSS.ss'].to_numpy())
t_sep = '-'
for i, t in enumerate(date):
	if len(t) == 19:
		timing = t[:4]+t_sep+t[4:6]+t_sep+t[6:8]+'T'+t[9:11]+':'+t[11:13]+':'+t[13:]
	elif len(t) == 18:
		timing = t[:4]+t_sep+t[4:6]+t_sep+t[6:8]+'T'+t[9:11]+':'+t[11:13]+':'+t[13:]+'0'
	elif len(t) == 17:
		timing = t[:4]+t_sep+t[4:6]+t_sep+t[6:8]+'T'+t[9:11]+':'+t[11:13]+':'+t[13:]+'00'
	elif len(t) == 16:
		timing = t[:4]+t_sep+t[4:6]+t_sep+t[6:8]+'T'+t[9:11]+':'+t[11:13]+':'+t[14:]+'.000'
	else:
		print(len(t))
		print(t)

	date[i] = timing

catalog['date'] = date.astype('datetime64[ms]')
catalog = catalog.drop(columns='#YYYYmmdd HHMMSS.ss')

# select only the non missing values
catalog = catalog.loc[np.isnan(catalog['Nb(#)']) == False]
catalog = catalog.reset_index(drop=True)

# Catalogue non complet avant 1998/2000
catalog = catalog.iloc[np.where(catalog['date'] >= np.array('2000-01-01',
														dtype='datetime64[ms]'))[0]]
catalog = catalog.reset_index(drop=True)

last_bound = str(catalog['date'].to_numpy()[-1]-np.array('2000-01-01',
														dtype='datetime64[ms]'))
# 1 hour = 60*60*1000 ms = 3 600 000 ms
# => because last_bound = 'xx...xx milliseconds' and so ' milliseconds' need to be removed.
last_bound = int(int(last_bound[:-13])/3600000) + 1
# '+ 1' comes from that with are taking bins of 1 hour

# Catalogue non complet avant 1998/2000
bounds = np.arange(np.array('2000-01-01', dtype='datetime64[h]'),
				   last_bound, 1, # '1' comes from that with are taking bins of 1 hour
				   dtype='datetime64[h]').astype('datetime64[ms]')

sismicity = np.zeros(len(bounds)-1)
for i in tqdm(range(len(bounds)-1)):
	sismicity[i] = np.sum(catalog['Nb(#)'][(catalog['date'] >= bounds[i])&(
											catalog['date'] < bounds[i+1])])

data = pd.DataFrame()
data['date'] = bounds[1:] # put as time the upper bound of the bins
data['sismicity'] = sismicity


events = pd.read_csv('./Eruptions/Events_hourly.txt')
Starting = events['Starting'].to_numpy().astype('datetime64[h]')
Ending = events['Ending'].to_numpy().astype('datetime64[h]')
activity = np.zeros(len(data), dtype='int8')
for i in range(len(Starting)):
	if (Ending[i] >= bounds[0])|(Starting[i] >= bounds[0]):
		bef = Starting[i] >= bounds[:-1]
		aft = Ending[i] < bounds[1:]
		ti = np.argwhere(bef)[-1, 0]
		tf = np.argwhere(aft)[0, 0]
		if ti == tf:
			if events['Type'][i] == 'Eruption':
				activity[ti] = 1
			elif events['Type'][i] == 'Intrusion':
				activity[ti] = 2

		elif ti < tf:
			if events['Type'][i] == 'Eruption':
				activity[ti:tf] = 1
			elif events['Type'][i] == 'Intrusion':
				activity[ti:tf] = 2


data['activity'] = activity

data.to_csv('./SISMO/bulletin_summit_hourly.csv',
			index=False)



