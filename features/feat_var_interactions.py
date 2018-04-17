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
train_usecols = ['ip', 'channel', 'device', 'os', 'app', 'click_time']
test_usecols = ['dataset', 'ip', 'channel', 'device', 'os', 'app', 'click_time']

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

print('Extracting new features...')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
train_df.drop('click_time', axis = 1, inplace = True)
gc.collect()
gc.collect()
train_df.head()

naddfeat=9
for i in range(0,naddfeat):
    if i==0: selcols=['ip', 'channel']; QQ=4;
    if i==1: selcols=['ip', 'device', 'os', 'app']; QQ=5;
    if i==2: selcols=['ip', 'day', 'hour']; QQ=4;
    if i==3: selcols=['ip', 'app']; QQ=4;
    if i==4: selcols=['ip', 'app', 'os']; QQ=4;
    if i==5: selcols=['ip', 'device']; QQ=4;
    if i==6: selcols=['app', 'channel']; QQ=4;
    if i==7: selcols=['ip', 'os']; QQ=5;
    if i==8: selcols=['ip', 'device', 'os', 'app']; QQ=4;
    print('selcols',selcols,'QQ',QQ)
    
    if QQ==0:
        gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].count().reset_index().\
            rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
        train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
    if QQ==1:
        gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].mean().reset_index().\
            rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
        train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
    if QQ==2:
        gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].var().reset_index().\
            rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
        train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
    if QQ==3:
        gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].skew().reset_index().\
            rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
        train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
    if QQ==4:
        gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].nunique().reset_index().\
            rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
        train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
    if QQ==5:
        gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].cumcount()
        train_df['X'+str(i)]=gp.values
        
    del gp
    gc.collect()    
# write there features to disk
features = [c for c in train_df.columns if c[0] == 'X']
for col in features:
    print('Change type %s'%(col))
    train_df[col].fillna(9999, inplace = True)
    train_df[col] = train_df[col].astype(np.int32)
    gc.collect()
drop = [col for col in train_df.columns if col not in features]
train_df.drop(drop, axis = 1, inplace = True)
gc.collect()

train_df.head()

feattst = train_df[len_train:]
feattst = feattst[testidx==1]
feattrn = train_df[:len_train]
feattst.reset_index(drop=True, inplace = True)
print(feattst.shape)
print(feattrn.shape)
feattst.to_feather(path+'../features/feat_var_kanbertst.feather')
feattrn.to_feather(path+'../features/feat_var_kanbertrn.feather')
feattrnval = feattrn[(60000000-2):(122080000-1)].reset_index(drop = True)
feattstval = pd.concat([feattrn[(144710000-2):(152400000-1)], \
           feattrn[(162000000-2):(168300000-1)], \
           feattrn[(175000000-2):(181880000-1)]]).reset_index(drop=True)

feattrnval.to_feather(path+'../features/feat_var_kanbertrnval.feather')
feattstval.to_feather(path+'../features/feat_var_kanbertstval.feather')
gc.collect()



'''
print('doing nextClick')
predictors=[]
new_feature = 'nextClick'
filename=path+'../features/nextClick_%s.csv'%(add_)

if os.path.exists(filename):
    print('loading from save file')
    QQ=pd.read_csv(filename).values
else:
    D=2**26
    train_df['category'] = (train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + "_" + train_df['device'].astype(str) \
        + "_" + train_df['os'].astype(str)).apply(hash) % D
    click_buffer= np.full(D, 3000000000, dtype=np.uint32)
    train_df['epochtime']= pd.to_datetime(train_df['click_time']).astype(np.int64) // 10 ** 9
    next_clicks= []
    for category, t in tqdm(zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values))):
        next_clicks.append(click_buffer[category]-t)
        click_buffer[category]= t
    del(click_buffer)
    QQ= list(reversed(next_clicks))

    if not debug:
        print('saving')
        pd.DataFrame(QQ).to_csv(filename,index=False)

train_df[new_feature] = QQ
predictors.append(new_feature)

train_df[new_feature+'_shift'] = pd.DataFrame(QQ).shift(+1).values
predictors.append(new_feature+'_shift')

del QQ
gc.collect()

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
    
predictors.extend(['ip_tcount', 'ip_tchan_count', 'ip_app_count',
              'ip_app_os_count', 'ip_app_os_var',
              'ip_app_channel_var_day','ip_app_channel_mean_hour'])
for i in range(0,naddfeat):
    predictors.append('X'+str(i))
    
print('predictors',predictors)

predictors = [c for c in predictors if c in train_df.columns]
print('New Features',predictors)

(train_df[len_train:][predictors]).to_csv(path+'../features/feat_kanbertst.csv.gz', compression = 'gzip')
(train_df[:len_train][predictors]).to_csv(path+'../features/feat_kanbertrn.csv.gz', compression = 'gzip')
train_df.shape
'''