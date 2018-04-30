import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import os

### Below a function is written to extract cumulative count feature  from different cols    

def do_cumcount( df, group_cols, counted,agg_type='uint16', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_cumcount'.format(('_'.join(group_cols)),(counted)) 
    if show_agg:
        print( "\nCumulative count by ", group_cols , '... and saved in', agg_name  )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )
### Below a function is written to extract mean feature  from different cols
def do_mean( df, group_cols, counted, agg_type='float16', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_mean'.format(('_'.join(group_cols)),(counted))  
    if show_agg:
        print( "\nCalculating mean of ", counted, " by ", group_cols , '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )


#path = '../input/'
path = "/home/darragh/tdata/data/"
#path = '/Users/dhanley2/Documents/tdata/data/'
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
if validation:
    add_ = 'val'
    ntrees = 2000 # 200
    early_stop = 100
    test_usecols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
    val_size = 0
else:
    ntrees = 1251
    val_size = 10000
    early_stop = ntrees
    test_usecols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
    add_ = ''

predictors = []
print('[{}] Load Train'.format(time.time() - start_time))
train_df = pd.read_csv(path+"train%s.csv"%(add_), dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed', 'attributed_time'])

print('[{}] Load Test'.format(time.time() - start_time))
test_df = pd.read_csv(path+"test%s.csv"%(add_), dtype=dtypes, usecols=test_usecols)
 
gc.collect()
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('int8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('int8') 
train_df = do_cumcount( train_df, ['ip'], 'os'); gc.collect()
train_df = do_cumcount( train_df, ['ip', 'device', 'os'], 'app'); gc.collect()
len_train = len(train_df)
train_df=train_df.append(test_df)
del test_df
    

keep = [col for col in train_df.columns if col not in test_usecols+['attributed_time', 'day', 'hour']]
feattst  = train_df[len_train:][keep]
feattrn = train_df[:len_train][keep]

feattrn.reset_index(drop=True, inplace = True)
feattst.reset_index(drop=True, inplace = True)
feattrn.to_feather(path+'../features/cumCount_trn%s.feather'%(add_))
feattst.to_feather(path+'../features/cumCount_tst%s.feather'%(add_))
gc.collect()

