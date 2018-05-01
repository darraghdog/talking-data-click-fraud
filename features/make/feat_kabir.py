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


predictors=[]
def do_next_Click( df,agg_suffix='nextClick', agg_type='float32'):
    
    GROUP_BY_NEXT_CLICKS = [
    
    # V1
    # {'groupby': ['ip']},
    # {'groupby': ['ip', 'app']},
    # {'groupby': ['ip', 'channel']},
    # {'groupby': ['ip', 'os']},
    
    # V3
    {'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    {'groupby': ['ip', 'os', 'device']},
    {'groupby': ['ip', 'os', 'device', 'app']},
    {'groupby': ['device']},
    {'groupby': ['device', 'channel']},     
    {'groupby': ['app', 'device', 'channel']},
    {'groupby': ['device', 'hour']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:
    
       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    
    
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation
        df[new_feature] = (df[all_features].groupby(spec[
            'groupby']).click_time.shift(-1) - df.click_time).dt.seconds.astype(agg_type)
        
        predictors.append(new_feature)
        gc.collect()
    return (df)

def do_prev_Click( df,agg_suffix='prevClick', agg_type='float32'):

    
    GROUP_BY_NEXT_CLICKS = [
    
    # V1
    # {'groupby': ['ip']},
    # {'groupby': ['ip', 'app']},
    {'groupby': ['ip', 'channel']},
    # {'groupby': ['ip', 'os']},
    
    # V3
    #{'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    #{'groupby': ['ip', 'os', 'device']},
    #{'groupby': ['ip', 'os', 'device', 'app']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:
    
       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    
    
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation
        df[new_feature] = (df.click_time - df[all_features].groupby(spec[
                'groupby']).click_time.shift(+1) ).dt.seconds.astype(agg_type)
        
        predictors.append(new_feature)
        gc.collect()
    return (df)    




## Below a function is written to extract count feature by aggregating different cols
def do_count( df, group_cols, agg_type='uint16', show_max=False, show_agg=True ):
    agg_name='{}count'.format('_'.join(group_cols))  
    if show_agg:
        print( "\nAggregating by ", group_cols ,  '... and saved in', agg_name )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )
    
##  Below a function is written to extract unique count feature from different cols
def do_countuniq( df, group_cols, counted, agg_type='uint8', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_countuniq'.format(('_'.join(group_cols)),(counted))  
    if show_agg:
        print( "\nCounting unqiue ", counted, " by ", group_cols ,  '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )
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

def do_var( df, group_cols, counted, agg_type='float16', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_var'.format(('_'.join(group_cols)),(counted)) 
    if show_agg:
        print( "\nCalculating variance of ", counted, " by ", group_cols , '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
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
path = '/Users/dhanley2/Documents/tdata/data/'
path = '/home/ubuntu/tdata/data/'
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
test_usecols = ['dataset', 'ip', 'device', 'os', 'app', 'channel', 'click_time']

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

del test_df
gc.collect()
train_df.shape
train_df.head()
        
gc.collect()
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('int8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('int8') 
#train_df = do_next_Click( train_df,agg_suffix='nextClick', agg_type='float32'  ); gc.collect()
#train_df = do_prev_Click( train_df,agg_suffix='prevClick', agg_type='float32'  ); gc.collect()  ## Removed temporarily due RAM sortage. 

train_df = do_countuniq( train_df, ['ip'], 'channel' ); gc.collect()
train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'app'); gc.collect()
train_df = do_countuniq( train_df, ['ip', 'day'], 'hour' ); gc.collect()
train_df = do_countuniq( train_df, ['ip'], 'app'); gc.collect()
train_df = do_countuniq( train_df, ['ip', 'app'], 'os'); gc.collect()
train_df = do_countuniq( train_df, ['ip'], 'device'); gc.collect()
train_df = do_countuniq( train_df, ['app'], 'channel'); gc.collect()
train_df = do_cumcount( train_df, ['ip'], 'os'); gc.collect()
train_df = do_cumcount( train_df, ['ip', 'device', 'os'], 'app'); gc.collect()
train_df = do_count( train_df, ['ip', 'day', 'hour'] ); gc.collect()
train_df = do_count( train_df, ['ip', 'app']); gc.collect()
train_df = do_count( train_df, ['ip', 'app', 'os']); gc.collect()
train_df.drop(train_usecols, 1, inplace = True)
del train_df['day']
del train_df['hour']
gc.collect()
gc.collect()



feattst = train_df[len_train:]
feattst = feattst[testidx==1]
feattrn = train_df[:len_train]
feattst.reset_index(drop=True, inplace = True)
print(feattst.shape)
print(feattrn.shape)
feattst.to_pickle(path+'../features/feat_next_kabirtst.pkl')
feattrn.to_pickle(path+'../features/feat_next_kabirtrn.pkl')
feattrnval = feattrn[(60000000-2):(122080000-1)].reset_index(drop = True)
feattstval = pd.concat([feattrn[(144710000-2):(152400000-1)], \
           feattrn[(162000000-2):(168300000-1)], \
           feattrn[(175000000-2):(181880000-1)]]).reset_index(drop=True)
feattrnval.reset_index(drop=True, inplace = True)
feattstval.reset_index(drop=True, inplace = True)
feattrnval.to_feather(path+'../features/feat_next_kabirtrnval.pkl')
feattstval.to_feather(path+'../features/feat_next_kabirtstval.pkl')
gc.collect()
