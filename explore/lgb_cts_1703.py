import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc

def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
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
train_df = pd.read_csv(path+"train.csv", nrows=80000000, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
print('load test...')
test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

print('load features...')
feattrnchl = pd.read_csv(path+'../features/lead_lag_trn_ip_device_os_channel.gz', compression = 'gzip', nrows=80000000)
feattstchl = pd.read_csv(path+'../features/lead_lag_tst_ip_device_os_channel.gz', compression = 'gzip')
feattrnos  = pd.read_csv(path+'../features/lead_lag_trn_ip_device_os.gz', compression = 'gzip', nrows=80000000)
feattstos  = pd.read_csv(path+'../features/lead_lag_tst_ip_device_os.gz', compression = 'gzip')

print(train_df.shape)
print(feattrnchl.shape)

feattstchl.columns = feattrnchl.columns = [i+'_chl' for i in feattrnchl.columns.tolist()]
feattstos.columns  = feattrnos.columns  = [i+'_os' for i in feattrnos.columns.tolist()]

feattrn = pd.concat([feattrnchl, feattrnos], axis=1)
feattst = pd.concat([feattstchl, feattstos], axis=1)

clip_val = 3600*9
feattrn = feattrn.clip(-clip_val, clip_val).astype(np.int16)
feattst = feattst.clip(-clip_val, clip_val).astype(np.int16)
del feattrnchl, feattrnos , feattstchl, feattstos
gc.collect()

#feattrn.hist()
#feattst.hist()

train_df = pd.concat([train_df, feattrn], axis=1)
test_df  = pd.concat([test_df , feattst], axis=1)


gc.collect()

len_train = len(train_df)
train_df=train_df.append(test_df)

del test_df
gc.collect()

print('data prep...')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')

gc.collect()

print('group by...')
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
print('merge...')
train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
del gp
gc.collect()

print('group by...')
gp = train_df[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train_df = train_df.merge(gp, on=['ip','app'], how='left')
del gp
gc.collect()


print('group by...')
gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
del gp
gc.collect()


print("vars and data type: ")
train_df.info()
train_df['qty'] = train_df['qty'].astype('uint16')
train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')
for col in train_df.columns:
    if 'click_sec' in col:
        print(col)
        train_df[col] = train_df[col].astype('int32')

train_df.head(10)
print(train_df.shape)
test_df = train_df[len_train:]
val_df = train_df[(len_train-75000000):len_train]
train_df = train_df[:(len_train-75000000)]

print("train size: ", len(train_df))
print("valid size: ", len(val_df))
print("test size : ", len(test_df))

target = 'is_attributed'
predictors = ['app','device','os', 'channel', 'hour', 'day', 'qty', 'ip_app_count', 'ip_app_os_count', \
              'click_sec_lag_os', 'click_sec_lag_chl', 'click_sec_lead_os', 'click_sec_lead_chl']
categorical = ['app','device','os', 'channel', 'hour']


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
    'scale_pos_weight':99 # because training data is extremely unbalanced 
}
bst = lgb_modelfit_nocv(params, 
                        train_df, 
                        val_df, 
                        predictors, 
                        target, 
                        objective='binary', 
                        metrics='auc',
                        early_stopping_rounds=50, 
                        verbose_eval=True, 
                        num_boost_round=300, 
                        categorical_features=categorical)

del train_df
del val_df
gc.collect()

print("Predicting...")
sub['is_attributed'] = bst.predict(test_df[predictors])
print("writing...")
sub.to_csv(path + '../sub/sub_lgb1503.csv',index=False)
print("done...")
print(sub.info())

'''
yact  = pd.read_csv(data_path + 'yvalsmall.csv')
fpr, tpr, thresholds = metrics.roc_curve(yact['is_attributed'].values, sub['is_attributed'], pos_label=1)
print(metrics.auc(fpr, tpr))
'''