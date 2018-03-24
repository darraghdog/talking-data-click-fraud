import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc
from sklearn import metrics

def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'use_missing':False,
        'learning_rate': 0.01,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 0,
        'metric':metrics
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)

    n_estimators = bst1.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

    return bst1

#path = '../input/'
path = "/home/darragh/tdata/data/"
path = '/Users/dhanley2/Documents/tdata/data/'
path = '/home/ubuntu/tdata/data/'
trn_load = 11180000000
val_size = 5000000

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


print('load train...')
train_df = pd.read_csv(path+"train.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
print('load test...')
test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

print('load features...')
feattrnchl = pd.read_csv(path+'../features/lead_lag_trn_ip_device_os_channel.gz', compression = 'gzip')
feattstchl = pd.read_csv(path+'../features/lead_lag_tst_ip_device_os_channel.gz', compression = 'gzip')
feattrnos  = pd.read_csv(path+'../features/lead_lag_trn_ip_device_os.gz', compression = 'gzip')
feattstos  = pd.read_csv(path+'../features/lead_lag_tst_ip_device_os.gz', compression = 'gzip')
feattrnct  = pd.read_csv(path+'../features/counttrn.gz', dtype=ctdtypes, compression = 'gzip')
feattstct  = pd.read_csv(path+'../features/counttst.gz', dtype=ctdtypes, compression = 'gzip')
#feattrnapp  = pd.read_csv(path+'../features/lead_lag_trn_ip_device_os_channel_appvalsmall.gz', compression = 'gzip')
#feattstapp  = pd.read_csv(path+'../features/lead_lag_tst_ip_device_os_channel_appvalsmall.gz', compression = 'gzip')

feattrnnext  = pd.read_csv(path+'../features/next_trn_ip_device_os.gz', compression = 'gzip').astype(np.int8)
feattstnext  = pd.read_csv(path+'../features/next_tst_ip_device_os.gz', compression = 'gzip').astype(np.int8)

entdtypes= {
'ip_device_entropy'     : np.float32,
'ip_os_entropy.x'       : np.float32,
'ip_app_entropy'        : np.float32,
'ip_channel_entropy'    : np.float32,
'ip_click_hr_entropy'   : np.float32,
'ip_click_min_entropy'  : np.float32,
'ip_os_entropy.y'       : np.float32
}

featentip  = pd.read_csv(path+'../features/entropyip.gz', dtype=entdtypes, compression = 'gzip')

entdtypes= {
'device_os_entropy'     : np.float32,
'device_channel_entropy': np.float32
}

featentdev = pd.read_csv(path+'../features/entropydev.gz', dtype=entdtypes, compression = 'gzip')

entdtypes= {
'channel_app_entropy': np.float32
}

featentchl = pd.read_csv(path+'../features/entropychl.gz', dtype=entdtypes, compression = 'gzip')

entdtypes= {
'app_channel_entropy': np.float32
}

featentapp = pd.read_csv(path+'../features/entropyapp.gz', dtype=entdtypes, compression = 'gzip')

#rolldtypes = {
#        'roll_mean_five'            : 'int32',
#        'roll_min_five'             : 'int32',
#        'roll_max_five'             : np.float16
#        'roll_var_five'             : np.float16
#        }
#feattrnroll  = pd.read_csv(path+'../features/roll_five_trn.gz', compression = 'gzip', dtype=rolldtypes)
#feattstroll  = pd.read_csv(path+'../features/roll_five_tst.gz', compression = 'gzip').astype(np.int8)
#feattrnroll.tail()

def sumfeat(df):
    dfsum = df.iloc[:,0] + df.iloc[:,1]
    dfsum[df.iloc[:,0]<0] = -1
    dfsum[df.iloc[:,1]<0] = -2
    dfsum[(df.iloc[:,1]>1000) & (df.iloc[:,0]>1000)] = -3
    dfsum[(df.iloc[:,1]<0) & (df.iloc[:,0]<0)] = -4
    return dfsum

print(train_df.shape)
print(feattrnchl.shape)

feattstchl.columns = feattrnchl.columns = [i+'_chl' for i in feattrnchl.columns.tolist()]
feattstos.columns  = feattrnos.columns  = [i+'_os' for i in feattrnos.columns.tolist()]

feattrn = pd.concat([feattrnchl, feattrnos, feattrnct], axis=1)
feattst = pd.concat([feattstchl, feattstos, feattstct], axis=1)
feattrn['click_sec_lsum_os'] = sumfeat(feattrnos)
feattrn['click_sec_lsum_chl'] = sumfeat(feattrnchl)
feattst['click_sec_lsum_os'] = sumfeat(feattstos)
feattst['click_sec_lsum_chl'] = sumfeat(feattstchl)
del feattrnchl, feattrnos , feattstchl, feattstos
import gc
gc.collect()
#feattrn.hist()
#feattst.hist()

#clip_val = 3600*9
#feattrn = feattrn.clip(-clip_val, clip_val).astype(np.int32)
#feattst = feattst.clip(-clip_val, clip_val).astype(np.int32)
gc.collect()
#feattrn.hist()
#feattst.hist()

train_df.head()

train_df = pd.concat([train_df, feattrn, feattrnnext], axis=1)
test_df  = pd.concat([test_df , feattst, feattstnext], axis=1)
del feattrn, feattst, feattrnnext, feattstnext
gc.collect()


len_train = len(train_df)
train_df=train_df.append(test_df)

del test_df
gc.collect()

print('data prep...')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
train_df.drop('click_time', axis=1, inplace = True)
gc.collect()

print('add entropy')
train_df = train_df.merge(featentip, on=['ip'], how='left')
train_df = train_df.merge(featentdev, on=['device'], how='left')
train_df = train_df.merge(featentchl, on=['channel'], how='left')
train_df = train_df.merge(featentapp, on=['app'], how='left')
train_df.head()
del featentip, featentdev, featentchl, featentapp

print("vars and data type: ")
train_df.info()
train_df['qty'] = train_df['qty'].astype('uint16')
train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')
train_df['channel_app'] = train_df['channel'] + 500*train_df['app']

print('Get common to train and test')
for col in ['app', 'channel', 'channel_app', 'ip', 'os', 'hour', 'device']:  
    gc.collect()
    print('Get common to train and test : %s'%(col))
    common = pd.Series(list(set(train_df[:(len_train-val_size)][col]) & set(train_df[len_train:][col])))
    train_df[col][~train_df[col].isin(common)] = np.nan
    del common
    gc.collect()

train_df.head(10)
print(train_df.shape)
test_df = train_df[len_train:]
val_df = train_df[(len_train-val_size):len_train]
train_df = train_df[:(len_train-val_size)]

print("train size: ", len(train_df))
print("valid size: ", len(val_df))
print("test size : ", len(test_df))

lead_cols = [col for col in train_df.columns if 'click_sec_l' in col]
lead_cols += [col for col in train_df.columns if 'next_' in col]
lead_cols += [col for col in train_df.columns if 'entropy' in col]
lead_cols += ['ip_app_channel_var_day', 'qty', 'ip_app_count', 'ip_app_os_count']
lead_cols += ['qty_var', 'ip_app_os_var','ip_app_channel_mean_hour']

target = 'is_attributed'
predictors = ['ip', 'channel_app', 'app','device','os', 'channel', 'hour', 'day', 'qty', 'ip_app_count', 'ip_app_os_count'] + lead_cols
categorical = ['ip', 'channel_app', 'app','device','os', 'channel', 'hour']
print(50*'*')
print(predictors)
print(50*'*')
print(categorical)
print(50*'*')

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')

gc.collect()

print("Training...")
params = {
    'learning_rate': 0.1,
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight':99, # because training data is extremely unbalanced 
    'use_missing':False
}
bst = lgb_modelfit_nocv(params, 
                        train_df, 
                        val_df, 
                        predictors, 
                        target, 
                        objective='binary', 
                        metrics='auc',
                        early_stopping_rounds=30, 
                        verbose_eval=True, 
                        num_boost_round=3000, 
                        categorical_features=categorical)
# [50]	train's auc: 0.97787	valid's auc: 0.981515
# [100]	train's auc: 0.984844	valid's auc: 0.984266
# [200]	train's auc: 0.987986	valid's auc: 0.985264
# [300]	train's auc: 0.989155	valid's auc: 0.98565

#del train_df
#del val_df
gc.collect()
#train_df.shape

imp = pd.DataFrame([(a,b) for (a,b) in zip(bst.feature_name(), bst.feature_importance())], columns = ['feat', 'imp'])
imp = imp.sort_values('imp', ascending = False).reset_index(drop=True)
print(imp)

print("Predicting...")
sub['is_attributed'] = bst.predict(test_df[predictors])
print("writing...")
sub.to_csv(path + '../sub/sub_lgb2403.csv',index=False, compression = 'gzip')
print("done...")
print(sub.info())

#yact  = pd.read_csv(path + 'yvalsmall.csv')
#yact.columns = ['id', 'is_attributed']
#fpr, tpr, thresholds = metrics.roc_curve(yact['is_attributed'].values, sub['is_attributed'], pos_label=1)
#print(metrics.auc(fpr, tpr))
# 0.965052
# 0.965109
# 0.96590
# 0.966364
# 0.96777
# 0.968164
# 0.968672 - common categoricals
# 0.987140

#del Report
#n_estimators :  300
#auc: 0.98564969384
#                        feat  imp
#0                         ip  464
#1                channel_app  297
#2                    channel  209
#3                        app  178
#4                         os  128
#5         click_sec_lead_chl   71
#6       ip_click_min_entropy   58
#7                       hour   58
#8                        qty   39
#9          click_sec_lead_os   34
#10              ip_app_count   29
#11            ip_app_entropy   27
#12                    device   24
#13             same_next_app   21
#14         ip_device_entropy   19
#15        ip_channel_entropy   19
#16       app_channel_entropy   15
#17           ip_os_entropy.x   15
#18           ip_app_os_count   15
#19                       qty   14
#20       channel_app_entropy   11
#21              ip_app_count    9
#22         click_sec_lsum_os    7
#23             ip_app_os_var    6
#24  ip_app_channel_mean_hour    6
#25    device_channel_entropy    5
#26       ip_click_hr_entropy    5
#27         device_os_entropy    4
#28        click_sec_lsum_chl    4
#29          click_sec_lag_os    2
#30             same_next_chl    2
#31           ip_os_entropy.y    1
#32    ip_app_channel_var_day    1
#33         click_sec_lag_chl    1
#34           ip_app_os_count    1
#35                   qty_var    1
#36                       day    0


