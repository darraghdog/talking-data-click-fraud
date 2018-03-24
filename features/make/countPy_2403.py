import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc
from sklearn import metrics


#path = '../input/'
path = "/home/darragh/tdata/data/"
#path = '/Users/dhanley2/Documents/tdata/data/'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

print('load train...')
train_df = pd.read_csv(path+"trainvalsmall.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time'])
print('load test...')
test_df = pd.read_csv(path+"testvalsmall.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time'])

len_train = len(train_df)
train_df=train_df.append(test_df)
del test_df
gc.collect()

print('data prep...')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
train_df.drop('click_time', inplace = True, axis=1)
train_df.head()
gc.collect()


print('group by : ip_app_channel_var_day')
gp = train_df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
gp['ip_app_channel_var_day'] = gp['ip_app_channel_var_day'].astype(np.float32)
train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
del gp
gc.collect()

print('group by : ip_day_hour_count_chl')
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
gp['qty'] = gp['qty'].astype('uint32')
train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
del gp
gc.collect()

print('group by : ip_app_count_chl')
gp = train_df[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
gp['ip_app_count'] = gp['ip_app_count'].astype('uint32')
train_df = train_df.merge(gp, on=['ip','app'], how='left')
del gp
gc.collect()

print('group by : ip_app_os_count_chl')
gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
gp['ip_app_os_count'] = gp['ip_app_os_count'].astype('uint32')
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
del gp
gc.collect()

print('group by : ip_day_chl_var_hour')
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'qty_var'})
gp['qty_var'] = gp['qty_var'].astype(np.float32)
train_df = train_df.merge(gp, on=['ip','day','channel'], how='left')
del gp
gc.collect()
train_df.head()

print('group by : ip_app_os_var_hour')
gp = train_df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
gp['ip_app_os_var'] = gp['ip_app_os_var'].astype(np.float32)
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
train_df['ip_app_os_var']
del gp
gc.collect()

print('group by : ip_app_chl_mean_hour')
gp = train_df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
gp['ip_app_channel_mean_hour'] = gp['ip_app_channel_mean_hour'].astype(np.float32)
train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
del gp
gc.collect()

train_df.fillna(9999999, inplace = True)
train_df.head()

outcols = train_df.columns.tolist()[7:]

train_df[:len_train][outcols].to_csv(path+'../features/counttrnvalsmall.gz', compression='gzip', index=False)
train_df[len_train:][outcols].to_csv(path+'../features/counttstvalsmall.gz', compression='gzip', index=False)