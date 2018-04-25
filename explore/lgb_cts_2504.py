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
    '''
    xgtrain.save_binary(path + '../weights/train_lgb.bin')
    xgtraindisk = lgb.Dataset(path + '../weights/train_lgb.bin', 
                             feature_name=predictors, 
                             categorical_feature=categorical,
                             free_raw_data=False)
    '''
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

validation = True
save_df    = False
load_df    = False
if validation:
    add_ = 'val'
    ntrees = 2000 # 200
    early_stop = 100
    test_usecols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
    val_size = 0
else:
    ntrees = 1200
    val_size = 1000
    early_stop = ntrees
    add_ = ''
    test_usecols = ['ip','app','device','os', 'channel', 'click_time', 'click_id']

print('[{}] Load Train'.format(time.time() - start_time))
train_df = pd.read_csv(path+"train%s.csv"%(add_), dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
print('[{}] Load Test'.format(time.time() - start_time))
test_df = pd.read_csv(path+"test%s.csv"%(add_), dtype=dtypes, usecols=test_usecols)

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
feattrnctsm = pd.read_csv(path+'../features/count_same_in_next_trn%s.gz'%(add_), compression = 'gzip')
feattstctsm = pd.read_csv(path+'../features/count_same_in_next_tst%s.gz'%(add_), compression = 'gzip')

feattstctsm.head()


gc.collect()
feattstprev.fillna(-1, inplace = True)
feattrnprev = feattrnprev.astype(np.int32)
feattstprev = feattstprev.astype(np.int32)

print('[{}] Load Entropy Features'.format(time.time() - start_time))
featentip  = pd.read_csv(path+'../features/entropyip.gz', compression = 'gzip', dtype = dtypes)
featentipdevos  = pd.read_csv(path+'../features/entropyipdevos.gz', compression = 'gzip', dtype = dtypes)
cols_ = [c for c in featentip.columns if c not in dtypes.keys()]
featentip[cols_] = featentip[cols_].astype(np.float32)
cols_ = [c for c in featentipdevos.columns if c not in dtypes.keys()]
featentipdevos[cols_] = featentipdevos[cols_].astype(np.float32)



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
feattrn = pd.concat([feattrnchl, feattrnos, feattrnapp], axis=1)
feattst = pd.concat([feattstchl, feattstos, feattstapp], axis=1)

print(feattrn.shape)
print(feattst.shape)
print(feattrn.head())
print(feattrn.head())
print(feattst.tail())
print(feattst.tail())

import gc
gc.collect()

clip_val = 3600*4
feattrn = feattrn.clip(-clip_val, clip_val).astype(np.int32)
feattst = feattst.clip(-clip_val, clip_val).astype(np.int32)
feattrn = pd.concat([feattrn, feattrnspl, feattrnsp1, feattrncum, feattrnld2], axis=1)
feattst = pd.concat([feattst, feattstspl, feattstsp1, feattstcum, feattstld2], axis=1)
del feattrnspl, feattrnsp1, feattrncum, feattrnchl, feattrnos, feattrnld2
del feattstspl, feattstsp1, feattstcum, feattstchl, feattstos, feattstld2
del feattrnapp, feattrnkan1, feattrnkan2, feattrnkan3
del feattstapp, feattstkan1, feattstkan2, feattstkan3
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
train_df = pd.concat([train_df, feattrn, feattrnnext, feattrnprev, feattrnctn, feattrnctsm, feattrnkan1, feattrnkan2, feattrnkan3], axis=1)
test_df  = pd.concat([test_df , feattst, feattstnext, feattstprev, feattstctn, feattstctsm, feattstkan1, feattstkan2, feattstkan3], axis=1)
del feattrn, feattst, feattrnnext, feattstnext, feattrnprev, feattstprev, feattrnctsm, feattrnctsm
gc.collect()

print(train_df.shape)
print(test_df.shape)
print(train_df.columns)
print(test_df.columns)

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
train_df = train_df.merge(featentipdevos, on=['ip', 'device', 'os'], how='left')
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
lead_cols += [col for col in train_df.columns if 'count_same_in_next_' in col]
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
    sub.to_csv(path + '../sub/sub_lgb1704.csv.gz',index=False, compression = 'gzip')
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
    predsdf.to_csv(path + '../sub/sub_lgb1704val.csv.gz',index=False, compression = 'gzip')


'''
[10]    train's auc: 0.969362   valid's auc: 0.960889
[20]    train's auc: 0.97306    valid's auc: 0.965649
[30]    train's auc: 0.975      valid's auc: 0.968813
[40]    train's auc: 0.977584   valid's auc: 0.972102
[50]    train's auc: 0.97958    valid's auc: 0.974637
[60]    train's auc: 0.981141   valid's auc: 0.976632
[70]    train's auc: 0.982035   valid's auc: 0.977678
[80]    train's auc: 0.982795   valid's auc: 0.978448
[90]    train's auc: 0.983277   valid's auc: 0.978942
[100]   train's auc: 0.98366    valid's auc: 0.979282
[110]   train's auc: 0.983933   valid's auc: 0.979589
[120]   train's auc: 0.984193   valid's auc: 0.979866
[130]   train's auc: 0.984456   valid's auc: 0.980128
[140]   train's auc: 0.984666   valid's auc: 0.9803
[150]   train's auc: 0.984871   valid's auc: 0.980486
[160]   train's auc: 0.985058   valid's auc: 0.980681
[170]   train's auc: 0.985236   valid's auc: 0.980865
[180]   train's auc: 0.985403   valid's auc: 0.981023
[190]   train's auc: 0.985545   valid's auc: 0.981163
[200]   train's auc: 0.985703   valid's auc: 0.98129
[210]   train's auc: 0.985837   valid's auc: 0.981429
[220]   train's auc: 0.985977   valid's auc: 0.981562
[230]   train's auc: 0.986093   valid's auc: 0.981659
[240]   train's auc: 0.986205   valid's auc: 0.981765
[250]   train's auc: 0.986312   valid's auc: 0.981848
[260]   train's auc: 0.9864     valid's auc: 0.981899
[270]   train's auc: 0.98646    valid's auc: 0.981962
[280]   train's auc: 0.986524   valid's auc: 0.981981
[290]   train's auc: 0.986593   valid's auc: 0.982055
[300]   train's auc: 0.986682   valid's auc: 0.982141
[310]   train's auc: 0.986754   valid's auc: 0.982202
[320]   train's auc: 0.986824   valid's auc: 0.98228
[330]   train's auc: 0.986886   valid's auc: 0.982328
[340]   train's auc: 0.986943   valid's auc: 0.982356
[350]   train's auc: 0.986999   valid's auc: 0.982375
[360]   train's auc: 0.987053   valid's auc: 0.98242
[370]   train's auc: 0.987113   valid's auc: 0.982453
[380]   train's auc: 0.987156   valid's auc: 0.982494
[390]   train's auc: 0.987209   valid's auc: 0.982543
[400]   train's auc: 0.987253   valid's auc: 0.982568
[410]   train's auc: 0.98729    valid's auc: 0.982583
[420]   train's auc: 0.987351   valid's auc: 0.982617
[430]   train's auc: 0.987389   valid's auc: 0.982661
[440]   train's auc: 0.987424   valid's auc: 0.982667
[450]   train's auc: 0.98746    valid's auc: 0.982715
[460]   train's auc: 0.9875     valid's auc: 0.98274
[470]   train's auc: 0.987525   valid's auc: 0.982764
[480]   train's auc: 0.987559   valid's auc: 0.982788
[490]   train's auc: 0.987588   valid's auc: 0.982797
[510]   train's auc: 0.987665   valid's auc: 0.982812
[520]   train's auc: 0.987696   valid's auc: 0.982839
[530]   train's auc: 0.987723   valid's auc: 0.982849
[540]   train's auc: 0.987755   valid's auc: 0.982861
[550]   train's auc: 0.987786   valid's auc: 0.982866
[560]   train's auc: 0.987809   valid's auc: 0.982875
[570]   train's auc: 0.987837   valid's auc: 0.982892
[580]   train's auc: 0.987867   valid's auc: 0.982897
[590]   train's auc: 0.987894   valid's auc: 0.982928
[600]   train's auc: 0.987919   valid's auc: 0.982946
[610]   train's auc: 0.987944   valid's auc: 0.982976
[620]   train's auc: 0.987965   valid's auc: 0.98297
[630]   train's auc: 0.987988   valid's auc: 0.982962
[640]   train's auc: 0.988012   valid's auc: 0.982967
[650]   train's auc: 0.988039   valid's auc: 0.982973
[660]   train's auc: 0.988063   valid's auc: 0.982993
[670]   train's auc: 0.988091   valid's auc: 0.983011
[680]   train's auc: 0.988122   valid's auc: 0.983018
[690]   train's auc: 0.988147   valid's auc: 0.983031
[700]   train's auc: 0.988173   valid's auc: 0.983032
[710]   train's auc: 0.988197   valid's auc: 0.983026
[720]   train's auc: 0.988221   valid's auc: 0.98303
[730]   train's auc: 0.988243   valid's auc: 0.983032
[740]   train's auc: 0.988277   valid's auc: 0.983041
[750]   train's auc: 0.988301   valid's auc: 0.983037
[760]   train's auc: 0.988317   valid's auc: 0.983037
[770]   train's auc: 0.988335   valid's auc: 0.983045
[780]   train's auc: 0.988357   valid's auc: 0.983051
[790]   train's auc: 0.988374   valid's auc: 0.983073
[800]   train's auc: 0.988397   valid's auc: 0.983075
[810]   train's auc: 0.988419   valid's auc: 0.983091
[820]   train's auc: 0.988447   valid's auc: 0.983112
[830]   train's auc: 0.988468   valid's auc: 0.983102
[840]   train's auc: 0.988482   valid's auc: 0.983112
[850]   train's auc: 0.988507   valid's auc: 0.983118
[860]   train's auc: 0.988527   valid's auc: 0.983113
[870]   train's auc: 0.988546   valid's auc: 0.983119
[880]   train's auc: 0.988568   valid's auc: 0.983113
[890]   train's auc: 0.988586   valid's auc: 0.983106
[900]   train's auc: 0.988606   valid's auc: 0.983109
[910]   train's auc: 0.988628   valid's auc: 0.983115
[920]   train's auc: 0.988646   valid's auc: 0.983104
[930]   train's auc: 0.988665   valid's auc: 0.983113
[940]   train's auc: 0.988686   valid's auc: 0.983118
[950]   train's auc: 0.988707   valid's auc: 0.983121
[960]   train's auc: 0.988727   valid's auc: 0.983126
[970]   train's auc: 0.988744   valid's auc: 0.983115
[980]   train's auc: 0.988764   valid's auc: 0.983111
[990]   train's auc: 0.988782   valid's auc: 0.983118
[1000]  train's auc: 0.9888     valid's auc: 0.983107
[1010]  train's auc: 0.988818   valid's auc: 0.983105
[1020]  train's auc: 0.988839   valid's auc: 0.983109
[1030]  train's auc: 0.988853   valid's auc: 0.983113
[1040]  train's auc: 0.988868   valid's auc: 0.983121
[1050]  train's auc: 0.988885   valid's auc: 0.983126
[1060]  train's auc: 0.988905   valid's auc: 0.983132
[1080]  train's auc: 0.988945   valid's auc: 0.983147
[1090]  train's auc: 0.988961   valid's auc: 0.983135
[1100]  train's auc: 0.988981   valid's auc: 0.983143
[1110]  train's auc: 0.988995   valid's auc: 0.983161
[1120]  train's auc: 0.989007   valid's auc: 0.983155
[1130]  train's auc: 0.989025   valid's auc: 0.983157
[1140]  train's auc: 0.989039   valid's auc: 0.983154
[1150]  train's auc: 0.989055   valid's auc: 0.98316
[1160]  train's auc: 0.989073   valid's auc: 0.983079
[1170]  train's auc: 0.989088   valid's auc: 0.983079
[1180]  train's auc: 0.989101   valid's auc: 0.983077
[1190]  train's auc: 0.989112   valid's auc: 0.983073
[1200]  train's auc: 0.98913    valid's auc: 0.983078
[1210]  train's auc: 0.989146   valid's auc: 0.983082
[1220]  train's auc: 0.98916    valid's auc: 0.983074
[1230]  train's auc: 0.989175   valid's auc: 0.98307
[1240]  train's auc: 0.989184   valid's auc: 0.983074
[1250]  train's auc: 0.989198   valid's auc: 0.98306
Early stopping, best iteration is:
[1151]  train's auc: 0.989055   valid's auc: 0.983164

Model Report
n_estimators :  1151
auc: 0.9831636575402569
                                feat   imp
0                            channel  1564
1                                 os  1081
2                                app  1039
3                               hour   769
4           click_sec_lead_split_sec   272
5                                 X8   143
6                      ip_os_entropy   133
7                                 X4   132
8                  ip_device_entropy   127
9                          nextClick   126
10                ip_channel_entropy   102
11               ip_click_hr_entropy    95
12                               qty    94
13                                X0    93
14                    ip_app_entropy    86
15                                X5    83
16                 click_sec_lag_app    80
17  click_sec_lead_split_sec_ip_only    78
18                                X3    77
19                                X1    73
20                            device    72
21                click_sec_lead_app    68
22                                ip    66
23             click_sec_lead_shift2    62
24            count_in_next_ten_mins    61
25                                X6    59
26                  count_in_next_hr    55
27              ip_click_min_entropy    36
28                      prevhour_qty    33
29                       prevday_qty    32
30                          cumsum50    27
31                          cumsum10    26
32                                X7    25
33                   nextClick_shift    21
34                                X2    16
Auc for all hours in testval : 0.9831636575402569
Auc for select hours in testval : 0.9667742378443394
writing...
'''
