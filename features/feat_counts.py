import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc
from sklearn import metrics
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

#path = '../input/'
path = "/home/darragh/tdata/data/"
path = '/Users/dhanley2/Documents/tdata/data/'
#path = '/home/ubuntu/tdata/data/'
start_time = time.time()

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
ctdtypes = {
        'ip_app_channel_var_day'    : np.float32,
        'qty'                       : 'uint32',
        'ip_app_count'              : 'uint32',
        'ip_app_os_count'           : 'uint32',
        'qty_var'                   : np.float32,
        'ip_app_os_var'             : np.float32,
        'ip_app_channel_mean_hour'  : np.float32
        }

validation = False
save_df    = False
load_df    = False
debug      = False
train_usecols = ['ip', 'device', 'os', 'app', 'channel', 'click_time']
test_usecols = ['dataset', 'ip', 'device', 'os', 'app', 'channel',  'click_time']

print('[{}] Load Train'.format(time.time() - start_time))
train_df = pd.read_csv(path+"train.csv", dtype=dtypes, usecols=train_usecols)
print('[{}] Load Test'.format(time.time() - start_time))
test_df = pd.read_csv(path+"testfull.csv", dtype=dtypes, usecols=test_usecols)
print(train_df.shape)
print(test_df.shape)
test_df.head()

# Build features
testidx = test_df['dataset'].values
test_df.drop('dataset', axis = 1, inplace = True)
len_train = len(train_df)
train_df=train_df.append(test_df)


print('Extracting new features...')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
train_df.drop('click_time', axis = 1, inplace = True)
gc.collect()

del test_df
gc.collect()
train_df.shape
train_df.tail()


print('grouping by ip-day-hour combination...')
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
del gp
gc.collect()

print('grouping by ip-app combination...')
gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train_df = train_df.merge(gp, on=['ip','app'], how='left')
del gp
gc.collect()

print('grouping by ip-app-os combination...')
gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
del gp
gc.collect()

# Adding features with var and mean hour (inspired from nuhsikander's script)
print('grouping by : ip_day_chl_var_hour')
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
train_df = train_df.merge(gp, on=['ip','day','channel'], how='left')
del gp
gc.collect()

print('grouping by : ip_app_os_var_hour')
gp = train_df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
del gp
gc.collect()

print('grouping by : ip_app_channel_var_day')
gp = train_df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
del gp
gc.collect()

print('grouping by : ip_app_chl_mean_hour')
gp = train_df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
print("merging...")
train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
del gp
gc.collect()

print("vars and data type: ")
train_df.info()
train_df['ip_tcount'] = train_df['ip_tcount'].astype('uint16')
train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')
for col in ['ip_tchan_count','ip_app_os_var','ip_app_channel_var_day', 'ip_app_channel_mean_hour']:   
    train_df[col] = train_df[col].astype(np.float32)
train_df['ip_tcount'].tail(5)
train_df.dtypes


predictors = ['ip_tcount', 'ip_tchan_count', 'ip_app_count',
              'ip_app_os_count', 'ip_app_os_var',
              'ip_app_channel_var_day','ip_app_channel_mean_hour']
    
print('predictors',predictors)
train_df = train_df[predictors]
gc.collect()


feattst = train_df[len_train:]
feattst = feattst[testidx==1]
feattrn = train_df[:len_train]
feattst.reset_index(drop=True, inplace = True)
print(feattst.shape)
print(feattrn.shape)
feattst.to_feather(path+'../features/feat_count_kanbertst.feather')
feattrn.to_feather(path+'../features/feat_count_kanbertrn.feather')
feattrnval = feattrn[(60000000-2):(122080000-1)].reset_index(drop = True)
feattstval = pd.concat([feattrn[(144710000-2):(152400000-1)], \
           feattrn[(162000000-2):(168300000-1)], \
           feattrn[(175000000-2):(181880000-1)]]).reset_index(drop=True)
feattrnval.reset_index(drop=True, inplace = True)
feattstval.reset_index(drop=True, inplace = True)
feattrnval.to_feather(path+'../features/feat_count_kanbertrnval.feather')
feattstval.to_feather(path+'../features/feat_count_kanbertstval.feather')
gc.collect()