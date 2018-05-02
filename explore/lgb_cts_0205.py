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
        'nthread': 16,
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
    ntrees = 1125
    val_size = 10000
    early_stop = ntrees
    add_ = ''
    test_usecols = ['ip','app','device','os', 'channel', 'click_time', 'click_id']

print('[{}] Load Train'.format(time.time() - start_time))
train_df = pd.read_csv(path+"train%s.csv"%(add_), dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed', 'attributed_time'])

print('[{}] Remove late downloads'.format(time.time() - start_time))
idx = train_df['attributed_time'].notnull()
click_time = pd.to_datetime(train_df[idx].click_time).astype(int)/(10**9)
attrb_time = pd.to_datetime(train_df[idx].attributed_time).astype(int)/(10**9)
diff_time  = attrb_time - click_time
fullidx = train_df[idx][diff_time>40000].index
train_df['is_attributed'].iloc[fullidx] = 0
train_df.drop('attributed_time', axis = 1, inplace = True)
train_df.head()

print('[{}] Load Test'.format(time.time() - start_time))
test_df = pd.read_csv(path+"test%s.csv"%(add_), dtype=dtypes, usecols=test_usecols)
if validation:
	y_orig = test_df['is_attributed'].values

print('[{}] Load Features'.format(time.time() - start_time))
feattrnapp = pd.read_csv(path+'../features/lead_lag_trn_ip_device_os_app%s.gz'%(add_), compression = 'gzip')
feattstapp = pd.read_csv(path+'../features/lead_lag_tst_ip_device_os_app%s.gz'%(add_), compression = 'gzip')
feattrnspl = pd.read_csv(path+'../features/lead_split_sec_trn_ip_device_os_app%s.gz'%(add_), compression = 'gzip').astype(np.float32)
feattstspl = pd.read_csv(path+'../features/lead_split_sec_tst_ip_device_os_app%s.gz'%(add_), compression = 'gzip').astype(np.float32)
feattrnsp1 = pd.read_csv(path+'../features/lead_split_sec_trn_ip%s.gz'%(add_), compression = 'gzip').astype(np.float32)
feattstsp1 = pd.read_csv(path+'../features/lead_split_sec_tst_ip%s.gz'%(add_), compression = 'gzip').astype(np.float32)
feattstsp1.columns = feattrnsp1.columns = [i+'_ip_only' for i in feattrnsp1.columns.tolist()]
feattrnnext  = pd.read_csv(path+'../features/next_trn_ip_device_os%s.gz'%(add_), compression = 'gzip').astype(np.int8)
feattstnext  = pd.read_csv(path+'../features/next_tst_ip_device_os%s.gz'%(add_), compression = 'gzip').astype(np.int8)
feattrnctn  = pd.read_csv(path+'../features/lead_count_next_ipdevosapp_trn%s.gz'%(add_), compression = 'gzip').astype(np.int16)
feattstctn  = pd.read_csv(path+'../features/lead_count_next_ipdevosapp_tst%s.gz'%(add_), compression = 'gzip').astype(np.int16)
feattrnprev  = pd.read_csv(path+'../features/prevdayipchlqtytrn%s.gz'%(add_), compression = 'gzip')#.astype(np.int32)
feattstprev  = pd.read_csv(path+'../features/prevdayipchlqtytst%s.gz'%(add_), compression = 'gzip')#.astype(np.int32)
feattrncum  = (pd.read_csv(path+'../features/cumsumday_trn%s.gz'%(add_), compression = 'gzip')*10000).astype(np.uint16)
feattstcum  = (pd.read_csv(path+'../features/cumsumday_tst%s.gz'%(add_), compression = 'gzip')*10000).astype(np.uint16)
feattrnchl = pd.read_csv(path+'../features/lead_lag_trn_ip_device_os_channel%s.gz'%(add_), compression = 'gzip')
feattstchl = pd.read_csv(path+'../features/lead_lag_tst_ip_device_os_channel%s.gz'%(add_), compression = 'gzip')
feattrnos  = pd.read_csv(path+'../features/lead_lag_trn_ip_device_os%s.gz'%(add_), compression = 'gzip')
feattstos  = pd.read_csv(path+'../features/lead_lag_tst_ip_device_os%s.gz'%(add_), compression = 'gzip')
# feattrncum = pd.read_csv(path+'../features/cum_min_trn_ip_device_os_app%s.gz'%(add_), compression = 'gzip')
# feattstcum = pd.read_csv(path+'../features/cum_min_tst_ip_device_os_app%s.gz'%(add_), compression = 'gzip')
feattrnld2 = pd.read_csv(path+'../features/lead2_trn_ip_device_os_app%s.gz'%(add_), compression = 'gzip')
feattstld2 = pd.read_csv(path+'../features/lead2_tst_ip_device_os_app%s.gz'%(add_), compression = 'gzip')

gc.collect()
feattstprev.fillna(-1, inplace = True)
feattrnprev = feattrnprev.astype(np.int32)
feattstprev = feattstprev.astype(np.int32)


print('[{}] Remove Duplicate IDs '.format(time.time() - start_time))
feattrnrdp = pd.read_csv(path+'../features/removedupetrn%s.gz'%(add_), compression = 'gzip')
feattstrdp = pd.read_csv(path+'../features/removedupetst%s.gz'%(add_), compression = 'gzip')
featrdp    = feattrnrdp.append(feattstrdp)
featrdp.shape
featrdp['x'].value_counts()

print('[{}] Load Entropy Features'.format(time.time() - start_time))

featentip  = pd.read_csv(path+'../features/entropyip.gz', compression = 'gzip')
featentip.iloc[:,1:] = featentip.iloc[:,1:].astype(np.float32)
featentip.iloc[:,0] = featentip.iloc[:,0].astype('uint32')


print('[{}] Load up the kanber features'.format(time.time() - start_time))
feattrnkan1 = pd.read_feather(path+'../features/feat_var_kanbertrn%s.feather'%(add_))
feattstkan1 = pd.read_feather(path+'../features/feat_var_kanbertst%s.feather'%(add_))
feattrnkan2 = pd.read_feather(path+'../features/feat_next_kanbertrn%s.feather'%(add_))
feattstkan2 = pd.read_feather(path+'../features/feat_next_kanbertst%s.feather'%(add_))
feattrnkan3 = pd.read_feather(path+'../features/feat_count_kanbertrn%s.feather'%(add_)).fillna(999)
feattstkan3 = pd.read_feather(path+'../features/feat_count_kanbertst%s.feather'%(add_)).fillna(999)
print(feattrnkan1.shape)
print(feattstkan1.shape)
print(feattrnkan1.head())
print(feattrnkan1.head())
print(feattstkan1.tail())
print(feattstkan1.tail())
print(feattrnkan2.shape)
print(feattstkan2.shape)
print(feattrnkan2.head())
print(feattrnkan2.head())
print(feattstkan2.tail())
print(feattstkan2.tail())
print(feattrnkan3.shape)
print(feattstkan3.shape)
print(feattrnkan3.head())
print(feattrnkan3.head())
print(feattstkan3.tail())
print(feattstkan3.tail())


print('[{}] Finished Loading Features, start concatenate'.format(time.time() - start_time))
def sumfeat(df):
    dfsum = df.iloc[:,0] + df.iloc[:,1]
    dfsum[df.iloc[:,0]<0] = -1
    dfsum[df.iloc[:,1]<0] = -2
    dfsum[(df.iloc[:,1]>1000) & (df.iloc[:,0]>1000)] = -3
    dfsum[(df.iloc[:,1]<0) & (df.iloc[:,0]<0)] = -4
    return dfsum

feattstapp.columns = feattrnapp.columns = [i+'_app' for i in feattrnapp.columns.tolist()]
feattstchl.columns = feattrnchl.columns = [i+'_chl' for i in feattrnchl.columns.tolist()]
feattstos.columns  = feattrnos.columns  = [i+'_os' for i in feattrnos.columns.tolist()]
feattrn = pd.concat([feattrnapp, feattrnkan1, feattrnkan2, feattrnkan3], axis=1)
feattst = pd.concat([feattstapp, feattstkan1, feattstkan2, feattstkan3], axis=1)

print(feattrn.shape)
print(feattst.shape)
print(feattrn.head())
print(feattrn.head())
print(feattst.tail())
print(feattst.tail())

del feattrnapp, feattrnkan1, feattrnkan2, feattrnkan3
del feattstapp, feattstkan1, feattstkan2, feattstkan3
import gc
gc.collect()

clip_val = 3600*5
feattrn = feattrn.clip(-clip_val, clip_val).astype(np.int32)
feattst = feattst.clip(-clip_val, clip_val).astype(np.int32)
feattrn = pd.concat([feattrn, feattrnspl, feattrnsp1, feattrncum, feattrnchl, feattrnos, feattrnld2], axis=1)
feattst = pd.concat([feattst, feattstspl, feattstsp1, feattstcum, feattstchl, feattstos, feattstld2], axis=1)
del feattrnspl, feattrnsp1, feattrncum, feattrnchl, feattrnos, feattrnld2
del feattstspl, feattstsp1, feattstcum, feattstchl, feattstos, feattstld2
gc.collect()
print(train_df.shape)
print(test_df.shape)
print(feattrn.shape)
print(feattst.shape)
print(feattrn.head())
print(feattrn.head())
print(feattst.tail())
print(feattst.tail())

print('[{}] Concat Train/Test'.format(time.time() - start_time))
train_df = pd.concat([train_df, feattrn, feattrnnext, feattrnprev, feattrnctn], axis=1)
test_df  = pd.concat([test_df , feattst, feattstnext, feattstprev, feattstctn], axis=1)
del feattrn, feattst, feattrnnext, feattstnext, feattrnprev, feattstprev
gc.collect()

print(train_df.shape)
print(test_df.shape)
print(train_df.columns)
print(test_df.columns)

# Drop the duplicates from train and test and add back later
train_df = train_df[feattrnrdp['x']==0]
test_df  = test_df [feattstrdp['x']==0]

len_train = len(train_df)
train_df=train_df.append(test_df)

del test_df
gc.collect()

print('[{}] Time prep'.format(time.time() - start_time))
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
gc.collect()

print('[{}] Add entropy'.format(time.time() - start_time))
train_df = train_df.merge(featentip, on=['ip'], how='left')
train_df.head()


print('[{}] Split data'.format(time.time() - start_time))
test_df = train_df[len_train:]
val_df = train_df[(len_train-val_size):len_train]
train_df = train_df[:(len_train-val_size)]

# Remove device 3032
print('Device 3032 shape : %s'%(train_df[train_df['device']==3032].shape[0]))
train_df = train_df[train_df['device']!=3032]
train_df.drop(['click_time'], axis = 1, inplace = True)
gc.collect()

print('[{}] Get common train and test'.format(time.time() - start_time))
for col in ['app', 'channel', 'os', 'hour', 'device']:  
    gc.collect()
    print('Get common to train and test : %s'%(col))
    counts_ = train_df[col].value_counts().reset_index()
    keepbig = counts_[counts_[col]>=5]['index']
    common = pd.Series(list(set(train_df[col]) & set(test_df[col])))
    keep = common #pd.Series(list(set(common).intersection(set(keepbig))))
    train_df[col][~train_df[col].isin(keep)] = np.nan
    test_df[col][~test_df[col].isin(keep)] = np.nan
    val_df  [col][~val_df[col].isin(keep)] = np.nan
    print('Length remaining for %s : %s' %(col, len(train_df[col].unique())))
    del common, keep, keepbig
    gc.collect()
for col in ['app', 'channel', 'os', 'hour', 'device']:
    train_df[col] = train_df[col].astype(np.float32)
    test_df [col] = test_df[col].astype(np.float32)
    val_df  [col] = val_df[col].astype(np.float32)
gc.collect()

print('[{}] Data split complete'.format(time.time() - start_time))
print("train size: ", len(train_df))
print("valid size: ", len(val_df))
print("test size : ", len(test_df))

exclude_cols = ['same_next_app', 'same_next_chl']
lead_cols = [col for col in train_df.columns if 'X' in col]
lead_cols += [col for col in train_df.columns if 'Click' in col]
lead_cols += [col for col in train_df.columns if 'lead_' in col]
lead_cols += [col for col in train_df.columns if 'lag_' in col]
lead_cols += [col for col in train_df.columns if 'next_' in col]
lead_cols += [col for col in train_df.columns if 'device_ct' in col]
lead_cols += [col for col in train_df.columns if 'cumsum' in col]
lead_cols += [col for col in train_df.columns if 'entropy' in col]
lead_cols += [col for col in train_df.columns if 'qty' in col]
lead_cols += [col for col in train_df.columns if 'count_in_next_' in col]
lead_cols += ['ip', 'app','device','os', 'channel', 'hour']
lead_cols = list(set(lead_cols))
lead_cols = [v for v in lead_cols if v not in exclude_cols]

target = 'is_attributed'
predictors =  lead_cols
categorical = [ 'app','device','os', 'channel', 'hour'] #'channel_app',
print(50*'*')
print(predictors)
print(50*'*')
print(categorical)
print(50*'*')

if not validation:
    train_df.drop(['click_id'], axis = 1, inplace = True)
    val_df.drop(['click_id'], axis = 1, inplace = True)
    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')
else:
    val_df = test_df#.sample(frac=.25, replace=False, random_state=0)
    gc.collect()
    
print('[{}] Drop features complete'.format(time.time() - start_time))
print("train size: ", len(train_df))
print("valid size: ", len(val_df))
print("test size : ", len(test_df))    

print('[{}] Training...'.format(time.time() - start_time))
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'learning_rate': 0.05,
    #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
    'num_leaves': 15,  # we should let it be smaller than 2^(max_depth)
    'max_depth': 6,  # -1 means no limit
    #'min_child_samples': 10,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 255,  # Number of bucketed bin for feature values
    'subsample': 0.90,  # Subsample ratio of the training instance.
    #'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.50,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    #'subsample_for_bin': 200000,  # Number of samples for constructing bin
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': 0,  # L1 regularization term on weights
    'reg_lambda': 0,  # L2 regularization term on weights
    'nthread': 16,
    'verbose': 1,
    'metric':'auc',
    'scale_pos_weight':99.0 # because training data is extremely unbalanced 
}



bst = lgb_modelfit_nocv(params, 
                        train_df, 
                        val_df, 
                        predictors, 
                        target, 
                        objective='binary', 
                        metrics='auc',
                        early_stopping_rounds=early_stop, 
                        verbose_eval=True, 
                        num_boost_round=ntrees, 
                        categorical_features=categorical)

gc.collect()
imp = pd.DataFrame([(a,b) for (a,b) in zip(bst.feature_name(), bst.feature_importance())], columns = ['feat', 'imp'])
imp = imp.sort_values('imp', ascending = False).reset_index(drop=True)
print(imp)

if not validation:
    print("Predicting...")
    sub['is_attributed'] = bst.predict(test_df[predictors])
    print("writing...")
    sub.to_csv(path + '../sub/sub_lgb0205.csv.gz',index=False, compression = 'gzip')
    print("done...")
    print(sub.info())
else:
    max_ip = 126413
    preds =   bst.predict(test_df[predictors])
    fpr, tpr, thresholds = metrics.roc_curve(test_df['is_attributed'].values, preds, pos_label=1)
    print('Auc for all hours in testval : %s'%(metrics.auc(fpr, tpr)))
    idx = test_df['ip']<=max_ip
    fpr1, tpr1, thresholds1 = metrics.roc_curve(test_df[idx]['is_attributed'].values, preds[idx], pos_label=1)
    print('Auc for select hours in testval : %s'%(metrics.auc(fpr1, tpr1)))
    print("writing...")
    predsdf = pd.DataFrame(preds)
    predsdf.to_csv(path + '../sub/sub_lgb0205valnodupe.csv.gz',index=False, compression = 'gzip')
    predsdupe = abs(1-feattstrdp['x'])
    predsdupe[feattstrdp['x']==0] = preds
    fpr, tpr, thresholds = metrics.roc_curve(y_orig, predsdupe, pos_label=1)
    print('Auc for original in testval : %s'%(metrics.auc(fpr, tpr)))
    print("writing...")
    predsdf = pd.DataFrame(predsdupe)
    predsdf.to_csv(path + '../sub/sub_lgb0205val.csv.gz',index=False, compression = 'gzip')

