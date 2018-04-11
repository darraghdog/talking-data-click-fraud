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
                          categorical_feature=categorical_features,
                          group=trngroup
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features,
                          group=tstgroup
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

validation =  True
if validation:
    add_ = 'val'
    ntrees = 2000 # 200
    early_stop = 100
    test_usecols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
    val_size = 0
else:
    ntrees = 900
    val_size = 10000
    early_stop = ntrees
    add_ = ''
    test_usecols = ['ip','app','device','os', 'channel', 'click_time', 'click_id']




int_cols = ['app', 'device', 'os', 'channel', 'day_minute']
logsm1_cols = ['qty', 'prevday_qty', 'prevhour_qty', 'count_in_next_ten_mins']

logsm2_cols = ['click_sec_lead_chl', 'click_sec_lag_chl', 'click_sec_lag_os', \
            'click_sec_lead_sameappchl', 'click_sec_lead_shift2', 'count_in_next_hr', \
            'ip_app_count', 'ip_app_os_count', 'qty_chl',  'unique_app_ipdevosmin', \
            'click_sec_lead_os', 'click_sec_lead_app', 'click_sec_lag_app']
logsm4_cols = ['click_sec_lead_split_sec', 'click_sec_lead_split_sec_ip_only']
numsm_cols =  ['ip_device_entropy', 'ip_app_entropy', 'ip_os_entropy', 'ip_click_min_entropy', \
               'ip_click_hr_entropy', 'ip_channel_entropy', ]

target = 'is_attributed'
predictors =  int_cols + logsm1_cols + logsm2_cols + logsm4_cols + numsm_cols + ['hour']
categorical = [ 'app','device','os', 'channel', 'hour']

train_df = pd.read_feather(path + '../weights/train_df%s.feather'%(add_))
test_df  = pd.read_feather(path + '../weights/test_df%s.feather'%(add_))
train_df.columns

print('[{}] Training...'.format(time.time() - start_time))
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

train_df = train_df[:500000]
test_df  = test_df[:500000]

trngroup = [train_df.shape[0]]
tstgroup = [test_df.shape[0]]

bst = lgb_modelfit_nocv(params, 
                        train_df, 
                        test_df, 
                        predictors, 
                        target, 
                        objective='lambdarank', #'binary', 
                        metrics='auc',
                        early_stopping_rounds=early_stop, 
                        verbose_eval=True, 
                        num_boost_round=ntrees, 
                        categorical_features=categorical)


#[50]    train's auc: 0.980264   valid's auc: 0.974851
#[100]   train's auc: 0.984106   valid's auc: 0.97945
#[200]   train's auc: 0.986127   valid's auc: 0.981569
#[300]   train's auc: 0.986882   valid's auc: 0.982256
#[400]   train's auc: 0.987341   valid's auc: 0.982577
#[500]   train's auc: 0.987666   valid's auc: 0.982792
#[600]   train's auc: 0.987927   valid's auc: 0.982883
#[700]   train's auc: 0.988151   valid's auc: 0.982953
#[800]   train's auc: 0.988366   valid's auc: 0.982993
#[900]   train's auc: 0.988544   valid's auc: 0.983015


gc.collect()
imp = pd.DataFrame([(a,b) for (a,b) in zip(bst.feature_name(), bst.feature_importance())], columns = ['feat', 'imp'])
imp = imp.sort_values('imp', ascending = False).reset_index(drop=True)
print(imp)

if not validation:
    print("Predicting...")
    sub['is_attributed'] = bst.predict(test_df[predictors])
    print("writing...")
    sub.to_csv(path + '../sub/sub_lgb0704A.csv.gz',index=False, compression = 'gzip')
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
    predsdf.to_csv(path + '../sub/sub_lgb0704val.csv.gz',index=False, compression = 'gzip')

#Early stopping, best iteration is:
#[875]   train's auc: 0.988508   valid's auc: 0.983024
#Model Report
#('n_estimators : ', 875)
#('auc:', 0.9830235581497813)
#                                feat   imp
#0                            channel  1379
#1                                 os   869
#2                                app   772
#3           click_sec_lead_split_sec   281
#4                      ip_os_entropy   142
#5                               hour   132
#6                  ip_device_entropy   127
#7                 ip_channel_entropy   112
#8               ip_click_min_entropy   100
#9                            qty_chl    98
#10                    ip_app_entropy    90
#11                               qty    90
#12                click_sec_lead_app    81
#13               ip_click_hr_entropy    77
#14                        day_minute    74
#15             unique_app_ipdevosmin    74
#16  click_sec_lead_split_sec_ip_only    71
#17                      ip_app_count    68
#18            count_in_next_ten_mins    66
#19                                ip    62
#20                   ip_app_os_count    60
#21                 click_sec_lag_app    56
#22                            device    53
#23                      prevhour_qty    49
#24                  count_in_next_hr    47
#25                 click_sec_lead_os    46
#26             click_sec_lead_shift2    39
#27                       prevday_qty    39
#28                click_sec_lead_chl    36
#29                  click_sec_lag_os    32
#30                 click_sec_lag_chl    18
#31         click_sec_lead_sameappchl    10
#Auc for all hours in testval : 0.9830235581497813
#Auc for select hours in testval : 0.9664638042562594

