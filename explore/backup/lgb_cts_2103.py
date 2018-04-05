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

trn_load = 200000000
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

print('load train...')
train_df = pd.read_csv(path+"train.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
print('load test...')
test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

print('load features...')
feattrnchl = pd.read_csv(path+'../features/lead_lag_trn_ip_device_os_channel.gz', compression = 'gzip')
feattstchl = pd.read_csv(path+'../features/lead_lag_tst_ip_device_os_channel.gz', compression = 'gzip')
feattrnos  = pd.read_csv(path+'../features/lead_lag_trn_ip_device_os.gz', compression = 'gzip')
feattstos  = pd.read_csv(path+'../features/lead_lag_tst_ip_device_os.gz', compression = 'gzip')
#feattrnapp  = pd.read_csv(path+'../features/lead_lag_trn_ip_device_os_channel_appvalsmall.gz', compression = 'gzip')
#feattstapp  = pd.read_csv(path+'../features/lead_lag_tst_ip_device_os_channel_appvalsmall.gz', compression = 'gzip')

feattrnnext  = pd.read_csv(path+'../features/next_trn_ip_device_os.gz', compression = 'gzip').astype(np.int8)
feattstnext  = pd.read_csv(path+'../features/next_tst_ip_device_os.gz', compression = 'gzip').astype(np.int8)
featentip  = pd.read_csv(path+'../features/entropyip.gz', compression = 'gzip')
#featentdev = pd.read_csv(path+'../features/entropydev.gz', compression = 'gzip')
#featentchl = pd.read_csv(path+'../features/entropychl.gz', compression = 'gzip')
#featentapp = pd.read_csv(path+'../features/entropyapp.gz', compression = 'gzip')
featnitip   = pd.read_csv(path+'../features/niteratio_ip.gz', compression = 'gzip')
featnitipdo = pd.read_csv(path+'../features/niteratio_ipdevos.gz', compression = 'gzip')
bmean_dtypes = {'bmean' : 'float32'}
featrnbmip  = pd.read_csv(path+'../features/bmeantrn_ip.gz', compression = 'gzip', dtype = bmean_dtypes)
featstbmip  = pd.read_csv(path+'../features/bmeantst_ip.gz', compression = 'gzip', dtype = bmean_dtypes)
featstbmip.head()

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

feattrn = pd.concat([feattrnchl, feattrnos], axis=1)
feattst = pd.concat([feattstchl, feattstos], axis=1)
feattrn['click_sec_lsum_os'] = sumfeat(feattrnos)
feattrn['click_sec_lsum_chl'] = sumfeat(feattrnchl)
feattst['click_sec_lsum_os'] = sumfeat(feattstos)
feattst['click_sec_lsum_chl'] = sumfeat(feattstchl)
del feattrnchl, feattrnos , feattstchl, feattstos
import gc
gc.collect()
#feattrn.hist()
#feattst.hist()

clip_val = 3600*9
feattrn = feattrn.clip(-clip_val, clip_val).astype(np.int32)
feattst = feattst.clip(-clip_val, clip_val).astype(np.int32)
gc.collect()
#feattrn.hist()
#feattst.hist()

train_df.head()

train_df = pd.concat([train_df, feattrn, feattrnnext], axis=1)
test_df  = pd.concat([test_df , feattst, feattstnext], axis=1)
del feattrn, feattst, feattrnnext, feattstnext
gc.collect()


train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
test_df['day']  = pd.to_datetime(test_df.click_time).dt.day.astype('uint8')

print('add bayes mean per ip')
featrnbmip['day'] = featrnbmip['click_day']
train_df = train_df.merge(featrnbmip, on=['ip', 'day'], how='left')
gc.collect()
test_df = test_df.merge( featstbmip, on=['ip'], how='left')
gc.collect()
train_df.head()


len_train = len(train_df)
train_df=train_df.append(test_df)

del test_df
gc.collect()

print('data prep...')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
#train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
gc.collect()
train_df.head()

train_df['day'].value_counts()


print('add entropy')
train_df = train_df.merge(featentip, on=['ip'], how='left')
#train_df = train_df.merge(featentdev, on=['device'], how='left')
#train_df = train_df.merge(featentchl, on=['channel'], how='left')
#train_df = train_df.merge(featentapp, on=['app'], how='left')
train_df.head()
del featentip#, featentdev, featentchl, featentapp

print('add nite ratio')
train_df = train_df.merge(featnitip, on=['ip'], how='left')
train_df = train_df.merge(featnitipdo, on=['ip', 'os', 'device'], how='left')


print('group by...unique app per ip/dev/os')
gp = train_df[['device', 'ip', 'os', 'app']].groupby(by=['device', 'ip', 'os'])[['app']].nunique().reset_index().rename(index=str, columns={'app': 'unique_app_ipdevos'})
print('merge...')
train_df = train_df.merge(gp[['device', 'ip', 'os', 'unique_app_ipdevos']], on=['device', 'ip', 'os'], how='left')
del gp
gc.collect()

train_df.head()

train_df.rename(columns={'unique_app_ipdevos_x': 'unique_app_ipdevos'}, inplace = True)



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
train_df['channel_app'] = train_df['channel'] + 500*train_df['app']
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
lead_cols += [col for col in train_df.columns if 'bmean' in col]

target = 'is_attributed'
predictors = ['channel_app', 'ip', 'app','device','os', 'channel', 'hour', 'day', 'qty', 'ip_app_count', 'ip_app_os_count'] + lead_cols
categorical = ['channel_app', 'app','device','os', 'channel', 'hour']
print(50*'*')
print(predictors)
print(50*'*')
print(categorical)
print(50*'*')

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')

gc.collect()
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
                        num_boost_round=30000, 
                        categorical_features=categorical)
#[50]    train's auc: 0.977003   valid's auc: 0.981833
#[100]   train's auc: 0.981865   valid's auc: 0.985199
#[200]   train's auc: 0.984189   valid's auc: 0.986701
#[300]   train's auc: 0.985121   valid's auc: 0.987021

del train_df
del val_df
gc.collect()

imp = pd.DataFrame([(a,b) for (a,b) in zip(bst.feature_name(), bst.feature_importance())], columns = ['feat', 'imp'])
imp = imp.sort_values('imp', ascending = False).reset_index(drop=True)
print(imp)

print("Predicting...")
sub['is_attributed'] = bst.predict(test_df[predictors])
print("writing...")
sub.to_csv(path + '../sub/sub_lgb2203.csv',index=False, compression = 'gzip')
print("done...")
print(sub.info())
'''
yact  = pd.read_csv(path + 'yvalsmall.csv')
yact.columns = ['id', 'is_attributed']
fpr, tpr, thresholds = metrics.roc_curve(yact['is_attributed'].values, sub['is_attributed'], pos_label=1)
print(metrics.auc(fpr, tpr))
'''
# 0.965052
# 0.965109
# 0.96590
# 0.966364
# 0.96777
# 0.968164
# 0.968696
# 0.969392




#Model Report
#('n_estimators : ', 0)
#('auc:', 0.9870205655080436)
#                         feat  imp
#0                 channel_app  569
#1                          os  225
#2                     channel  176
#3          click_sec_lead_chl  119
#4        ip_click_min_entropy   84
#5                         app   82
#6                        hour   81
#7                         qty   70
#8           ip_device_entropy   54
#9               ip_os_entropy   46
#10          click_sec_lead_os   37
#11                     device   34
#12             ip_app_entropy   33
#13            ip_app_os_count   30
#14              same_next_app   30
#15         ip_channel_entropy   23
#16        ip_click_hr_entropy   19
#17                         ip   17
#18       bmean_nite_clicks_ip   14
#19  bmean_nite_clicks_ipdevos   10
#20           click_sec_lag_os   10
#21          click_sec_lsum_os    9
#22         click_sec_lsum_chl    8
#23              same_next_chl    7
#24               ip_app_count    7
#25          click_sec_lag_chl    6
#26                        day    0
