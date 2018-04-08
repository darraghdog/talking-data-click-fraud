# https://www.kaggle.com/anttip/talkingdata-wordbatch-fm-ftrl-lb-0-9681/code
import sys, gc
from tqdm import tqdm
import wordbatch
from wordbatch.extractors import WordHash
from wordbatch.models import FM_FTRL
import threading
import pandas as pd
from sklearn.metrics import roc_auc_score
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
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

validation =  True
load_raw = False
if validation:
    add_ = 'val'
    ntrees = 2000 # 200
    early_stop = 100
    test_usecols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
    val_size = 0
else:
    ntrees = 1000
    val_size = 10000
    early_stop = ntrees
    add_ = ''
    test_usecols = ['ip','app','device','os', 'channel', 'click_time', 'click_id']




if load_raw:
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
    
    feattrnchl = pd.read_csv(path+'../features/lead_lag_trn_ip_device_os_channel%s.gz'%(add_), compression = 'gzip')
    feattstchl = pd.read_csv(path+'../features/lead_lag_tst_ip_device_os_channel%s.gz'%(add_), compression = 'gzip')
    feattrnos  = pd.read_csv(path+'../features/lead_lag_trn_ip_device_os%s.gz'%(add_), compression = 'gzip')
    feattstos  = pd.read_csv(path+'../features/lead_lag_tst_ip_device_os%s.gz'%(add_), compression = 'gzip')
    # feattrncum = pd.read_csv(path+'../features/cum_min_trn_ip_device_os_app%s.gz'%(add_), compression = 'gzip')
    # feattstcum = pd.read_csv(path+'../features/cum_min_tst_ip_device_os_app%s.gz'%(add_), compression = 'gzip')
    feattrnld2 = pd.read_csv(path+'../features/lead2_trn_ip_device_os_app%s.gz'%(add_), compression = 'gzip')
    feattstld2 = pd.read_csv(path+'../features/lead2_tst_ip_device_os_app%s.gz'%(add_), compression = 'gzip')
    feattrnnext  = pd.read_csv(path+'../features/next_trn_ip_device_os%s.gz'%(add_), compression = 'gzip').astype(np.int8)
    feattstnext  = pd.read_csv(path+'../features/next_tst_ip_device_os%s.gz'%(add_), compression = 'gzip').astype(np.int8)
    # feattrnprev  = pd.read_csv(path+'../features/prevdayipchlqtytrn%s.gz'%(add_), compression = 'gzip').astype(np.int32)
    # feattstprev  = pd.read_csv(path+'../features/prevdayipchlqtytst%s.gz'%(add_), compression = 'gzip').astype(np.int32)
    feattrnctn  = pd.read_csv(path+'../features/lead_count_next_ipdevosapp_trn%s.gz'%(add_), compression = 'gzip').astype(np.int16)
    feattstctn  = pd.read_csv(path+'../features/lead_count_next_ipdevosapp_tst%s.gz'%(add_), compression = 'gzip').astype(np.int16)
    feattrnprev  = pd.read_csv(path+'../features/prevdayipchlqtytrn%s.gz'%(add_), compression = 'gzip')#.astype(np.int32)
    feattstprev  = pd.read_csv(path+'../features/prevdayipchlqtytst%s.gz'%(add_), compression = 'gzip')#.astype(np.int32)
    feattstprev.fillna(-1, inplace = True)
    feattrnprev = feattrnprev.astype(np.int32)
    feattstprev = feattstprev.astype(np.int32)
    
    featentip  = pd.read_csv(path+'../features/entropyip.gz', compression = 'gzip')
    featentip.iloc[:,1:] = featentip.iloc[:,1:].astype(np.float32)
    featentip.iloc[:,0] = featentip.iloc[:,0].astype('uint32')
    
    
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
    #feattrn['click_sec_lsum_os'] = sumfeat(feattrnos)
    #feattrn['click_sec_lsum_chl'] = sumfeat(feattrnchl)
    #feattst['click_sec_lsum_os'] = sumfeat(feattstos)
    #feattst['click_sec_lsum_chl'] = sumfeat(feattstchl)
    feattst[['click_sec_lead_chl', 'click_sec_lead_app']].head(300)
    feattrn['click_sec_lead_sameappchl'] = \
            (feattrn['click_sec_lead_chl']==feattrn['click_sec_lead_app']).astype('int8')
    feattst['click_sec_lead_sameappchl'] = \
            (feattst['click_sec_lead_chl']==feattst['click_sec_lead_app']).astype('int8')
    
    del feattrnchl, feattrnos, feattrnapp
    del feattstchl, feattstos, feattstapp
    import gc
    gc.collect()
    #feattrn.hist()
    #pd.crosstab(feattrn['click_sec_lag_sameappchl'],train_df['is_attributed'])#feattst.hist()
    
    clip_val = 3600*9
    feattrn = feattrn.clip(-clip_val, clip_val).astype(np.int32)
    feattst = feattst.clip(-clip_val, clip_val).astype(np.int32)
    feattrn = pd.concat([feattrn, feattrnld2, feattrnspl, feattrnsp1], axis=1)
    feattst = pd.concat([feattst, feattstld2, feattstspl, feattstsp1], axis=1)
    del feattrnld2, feattrnspl, feattrnsp1
    del feattstld2, feattstspl, feattstsp1
    gc.collect()
    #feattrn.hist()
    #feattst.hist()
    
    
    print(train_df.shape)
    print(test_df.shape)
    
    
    print('[{}] Concat Train/Test'.format(time.time() - start_time))
    train_df = pd.concat([train_df, feattrn, feattrnnext, feattrnprev, feattrnctn], axis=1)
    test_df  = pd.concat([test_df , feattst, feattstnext, feattstprev, feattstctn], axis=1)
    del feattrn, feattst, feattrnnext, feattstnext, feattrnprev, feattstprev
    gc.collect()
    
    print(train_df.shape)
    print(test_df.shape)
    
    len_train = len(train_df)
    train_df=train_df.append(test_df)
    
    del test_df
    gc.collect()
    
    print('[{}] Time prep'.format(time.time() - start_time))
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    train_df['minute'] = pd.to_datetime(train_df.click_time).dt.minute.astype('uint8')
    train_df['day_minute'] = (train_df['hour']*24) + train_df['minute']
    gc.collect()
    
    
    print('[{}] group by...unique app per ip/dev/os'.format(time.time() - start_time))
    gp = train_df[['device', 'ip', 'os', 'app']].groupby(by=['device', 'ip', 'os'])[['app']].nunique().reset_index().rename(index=str, columns={'app': 'unique_app_ipdevos'})
    print('merge...')
    train_df = train_df.merge(gp[['device', 'ip', 'os', 'unique_app_ipdevos']], on=['device', 'ip', 'os'], how='left')
    del gp
    gc.collect()
    train_df.rename(columns={'unique_app_ipdevos_x': 'unique_app_ipdevos'}, inplace = True)
    
    print('[{}] group by...count app per ip/dev/os/min'.format(time.time() - start_time))
    gp = train_df[['device', 'ip', 'os', 'app', 'minute']].groupby(by=['device', 'ip', 'os', 'minute'])[['app']].count().reset_index().rename(index=str, columns={'app': 'unique_app_ipdevosmin'})
    print('merge...')
    train_df = train_df.merge(gp[['device', 'ip', 'os', 'minute', 'unique_app_ipdevosmin']], on=['device', 'ip', 'os', 'minute'], how='left')
    del gp
    gc.collect()
    
    print('[{}] group by...unique app per ip/day/hr/chl'.format(time.time() - start_time))
    gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty_chl'})
    print('merge...')
    train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
    del gp
    gc.collect()
    
    
    print('[{}] group by...unique app per ip/app/chl'.format(time.time() - start_time))
    gp = train_df[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    train_df = train_df.merge(gp, on=['ip','app'], how='left')
    del gp
    gc.collect()
    
    
    print('[{}] group by...unique app per ip/app/os/chl'.format(time.time() - start_time))
    gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
    del gp
    gc.collect()
    
    #print('[{}] group by...count device'.format(time.time() - start_time))
    #gp = train_df['device'].value_counts().reset_index().rename(index=str, columns={'device': 'device_ct', 'index' : 'device'})
    #print('merge...')
    #train_df = train_df.merge(gp, on=['device'], how='left')
    #del gp
    gc.collect()
    
    print('[{}] Add entropy'.format(time.time() - start_time))
    train_df = train_df.merge(featentip, on=['ip'], how='left')
    train_df.head()
    
    featentip.head()
    
    print('[{}] Data types'.format(time.time() - start_time))
    train_df['qty'] = train_df['qty'].astype('uint16')
    train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
    train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')
    train_df['click_sec_lead_shift2'] = train_df['click_sec_lead_shift2'].astype('int32')
    #train_df['channel_app'] = train_df['channel'] + 500*train_df['app']
    train_df.info()
    gc.collect()
    
    train_df.head(5)
    print(train_df.shape)
    test_df = train_df[len_train:]
    val_df = train_df[(len_train-val_size):len_train]
    train_df = train_df[:(len_train-val_size)]
    gc.collect()
    
    ## Remove the first day
    #train_df = train_df[(train_df['day']>6) & ~((train_df['day']==7) & ( train_df['hour']<4))]
    
    # Remove device 3032
    print('Device 3032 shape : %s'%(train_df[train_df['device']==3032].shape[0]))
    train_df = train_df[train_df['device']!=3032]
    train_df.drop(['day', 'click_time' ,'unique_app_ipdevos'], axis = 1, inplace = True)
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
    lead_cols = [col for col in train_df.columns if 'lead_' in col]
    lead_cols += [col for col in train_df.columns if 'lag_' in col]
    lead_cols += [col for col in train_df.columns if 'next_' in col]
    lead_cols += [col for col in train_df.columns if 'device_ct' in col]
    lead_cols += [col for col in train_df.columns if 'entropy' in col]
    lead_cols += [col for col in train_df.columns if 'qty' in col]
    lead_cols += [col for col in train_df.columns if 'count_in_next_' in col]
    lead_cols += ['ip', 'app','device','os', 'channel', 'hour', 'ip_app_count', 'ip_app_os_count', 'unique_app_ipdevosmin', 'day_minute']
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
        val_df = test_df#.sample(frac=0.025, replace=False, random_state=0)
        gc.collect()
        
    drop_cols = [col for col in train_df.columns if col not in [target]+predictors]
    train_df.drop(drop_cols, axis = 1, inplace = True)
    val_df.drop(drop_cols, axis = 1, inplace = True)
    drop_cols = [col for col in test_df.columns if col not in [target]+predictors]
    test_df.drop(drop_cols, axis = 1, inplace = True)
    gc.collect()
        
    print('[{}] Drop features complete'.format(time.time() - start_time))
    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))
    print("test size : ", len(test_df))
    
    train_df.to_pickle(path + '../weights/train_df%s.pkl'%(add_))
    val_df.to_pickle(path + '../weights/val_df%s.pkl'%(add_))
    test_df.to_pickle(path + '../weights/test_df%s.pkl'%(add_))
    if not validation:
        sub.to_pickle(path + '../weights/sub%s.pkl'%(add_))
        del sub
    del train_df, val_df, test_df
    gc.collect()


class FFMFormatPandas:
    def __init__(self):
        self.field_index_ = None
        self.feature_index_ = None
        self.y = None

    def fit(self, df, y=None):
        self.y = y
        df_ffm = df[df.columns.difference([self.y])]
        if self.field_index_ is None:
            self.field_index_ = {col: i for i, col in enumerate(df_ffm)}

        if self.feature_index_ is not None:
            last_idx = max(list(self.feature_index_.values()))

        if self.feature_index_ is None:
            self.feature_index_ = dict()
            last_idx = 0

        for col in df.columns:
            vals = df[col].unique()
            for val in vals:
                if pd.isnull(val):
                    continue
                name = '{}_{}'.format(col, val)
                if name not in self.feature_index_:
                    self.feature_index_[name] = last_idx
                    last_idx += 1
            self.feature_index_[col] = last_idx
            last_idx += 1
        return self

    def fit_transform(self, df, y=None):
        self.fit(df, y)
        return self.transform(df)

    def transform_row_(self, row, t):
        ffm = []
        if self.y != None:
            ffm.append(str(row.loc[row.index == self.y][0]))
        if self.y is None:
            ffm.append(str(0))

        for col, val in row.loc[row.index != self.y].to_dict().items():
            col_type = t[col]
            name = '{}_{}'.format(col, val)
            if col_type.kind ==  'O':
                ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
            elif col_type.kind == 'i':
                ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], val))
        return ' '.join(ffm)

    def transform(self, df):
        t = df.dtypes.to_dict()
        return pd.Series({idx: self.transform_row_(row, t) for idx, row in df.iterrows()})


def logsm(var, mult):
    var[var == -1] = 999999
    return np.log2((1+(var**1.5))*mult).astype(np.int32)#.astype('str')

def numsm(var, mult):
    return (var*mult).astype(np.int32)#.astype('str')

def df2ffmdf(df):    
    print('intcols')
    for col in int_cols:
        df[col] = df[col].fillna(60000).astype(np.int16)
    gc.collect()
    
    print('logsm1_cols')
    for col in logsm1_cols:
        df[col] = logsm(df[col].values, 1)
    gc.collect()
    
    print('logsm2_cols')
    for col in logsm2_cols:
        df[col] = logsm(df[col].values, 2)
    gc.collect()
    
    print('logsm4_cols')
    for col in logsm4_cols:
        df[col] = logsm(df[col].values, 4)
    gc.collect()
    
    print('numsm_cols')
    for col in numsm_cols:
        df[col] = numsm(df[col].values, 100)
    gc.collect()
    '''
    df.drop(drop_cols, axis = 1, inplace = True)
    '''
    return df


int_cols = ['app', 'device', 'os', 'channel', 'day_minute', 'hour']
logsm1_cols = ['qty', 'prevday_qty', 'prevhour_qty', 'count_in_next_ten_mins']

logsm2_cols = ['click_sec_lead_chl', 'click_sec_lag_chl', 'click_sec_lag_os', \
            'click_sec_lead_sameappchl', 'click_sec_lead_shift2', 'count_in_next_hr', \
            'ip_app_count', 'ip_app_os_count', 'qty_chl',  'unique_app_ipdevosmin', \
            'click_sec_lead_os', 'click_sec_lead_app', 'click_sec_lag_app']
logsm4_cols = ['click_sec_lead_split_sec', 'click_sec_lead_split_sec_ip_only', 'ip_os_entropy.y']
numsm_cols =  ['ip_device_entropy', 'ip_app_entropy', 'ip_os_entropy.x', 'ip_click_min_entropy', \
               'ip_click_hr_entropy', 'ip_channel_entropy']
drop_cols = []

allcols = int_cols + logsm1_cols + logsm2_cols + logsm4_cols + numsm_cols 
ffmcolmap = dict((k, 1+v) for (v, k) in enumerate(list(set(allcols))))

# Check we are not missing any
#[col for col in train_df.columns if col not in allcols]


train_df = df2ffmdf(pd.read_pickle(path + '../weights/train_df%s.pkl'%(add_)))
gc.collect()
test_df  = df2ffmdf(pd.read_pickle(path + '../weights/test_df%s.pkl'%(add_)))
gc.collect()

if not validation:
    sub  = read_pickle(path + '../weights/sub%s.pkl'%(add_))

test_df.reset_index(drop=True, inplace = True)
train_df.reset_index(drop=True, inplace = True)

train_df.dtypes
train_df.head()


for t, row in test_df.iterrows():
    if t%4000==0:
        break
fname = path + '../weights/test_df.ffm'


ffm_train = FFMFormatPandas()
ffm_test_data = ffm_train.fit_transform(test_df, y='is_attributed')

with open(fname, 'w') as outfile:
    for t, row in tqdm(test_df.iterrows()):
        #if t%1000000==0:
        #    print("Write file processed rows : %sM ; %ss "%( int(t/1e+6), '%0.0f'%(time.time()-start_time)))
            
        # Make target
        if 'is_attributed' in row:
            out = str(int(row['is_attributed']))
            del row['is_attributed']
        else:
            out = '0'
            
        # Add features
        for kidx, vidx in ffmcolmap.items():
            if row[kidx]!=row[kidx]:
                continue
            v = int(row[kidx])
            out += ' %s:%s:1'%(ffmcolmap[kidx], vidx)
        
        outfile.write('%s\n' % (out))
    
    print('Files missing key counts...')
    for k, v in missing_keys.items():
        print(k, len(list(set(v))),len(list(v)))

# 'click_sec_os_chl'
#allcols = int_cols+logsm1_cols+logsm2_cols+logsm4_cols+numsm_cols
#[col for col in train_df.columns if col not in allcols]



for t, row in tqdm(test_df.iterrows()):
    row
    if t > 1000000:
        break




'''            
darragh@darragh-xps:~/pkgs/libffm/libffm_toy$ head criteo.tr.r100.gbdt0.ffm 
0 0:194689:1 1:219069:1 2:319325:1 3:754755:1 4:515702:1 5:359746:1 6:707332:1 7:396128:1 8:650556:1 9:530364:1 10:148557:1 11:548643:1 12:790331:1 13:344176:1 14:145710:1 15:299224:1 16:901863:1 17:66695:1 18:182671:1 19:832740:1 20:869462:1 21:198788:1 22:660059:1 23:134912:1 24:430579:1 25:285442:1 26:78061:1 27:660466:1 28:326359:1 29:633290:1 30:271271:1 31:765122:1 32:322020:1 33:943765:1 34:541960:1 35:821014:1 36:428618:1 37:607936:1 38:142693:1
0 0:40189:1 1:164503:1 2:659433:1 3:100700:1 4:224808:1 5:463263:1 6:608573:1 7:223864:1 8:169525:1 9:255381:1 10:756430:1 11:832677:1 12:429274:1 13:370671:1 14:226100:1 15:98199:1 16:218827:1 17:397270:1 18:316115:1 19:561396:1 20:50216:1 21:198788:1 22:693070:1 23:859765:1 24:811335:1 25:812374:1 26:198506:1 27:581745:1 28:809536:1 29:166818:1 30:642460:1 31:998174:1 32:614600:1 33:34050:1 34:541960:1 35:543189:1 36:101169:1 37:335222:1 38:975766:1
1 0:194689:1 1:17524:1 2:790098:1 3:25173:1 4:543954:1 5:239747:1 6:946867:1 7:761173:1 8:248150:1 9:530364:1 10:523673:1 11:365321:1 12:130107:1 13:161559:1 14:421400:1 15:98199:1 16:218827:1 17:397270:1 18:182671:1 19:624875:1 20:50216:1 21:198788:1 22:580421:1 23:853737:1 24:811335:1 25:173288:1 26:78061:1 27:740257:1 28:809536:1 29:43420:1 30:952896:1 31:998174:1 32:614600:1 33:34050:1 34:203287:1 35:821014:1 36:101169:1 37:335222:1 38:975766:1

darragh@darragh-xps:~/pkgs/libffm/libffm_toy$ head criteo.va.r100.gbdt0.ffm
1 0:290489:1 1:17524:1 2:790098:1 3:25173:1 4:230523:1 5:786172:1 6:166412:1 7:396128:1 8:582090:1 9:512211:1 10:982324:1 11:548643:1 12:130107:1 13:344176:1 14:704452:1 15:98079:1 16:550602:1 17:397270:1 18:182671:1 19:942674:1 20:275751:1 21:198788:1 22:318202:1 23:900144:1 24:540660:1 25:134997:1 26:198506:1 27:561281:1 28:433710:1 29:633290:1 30:577012:1 31:7551:1 32:382381:1 33:758475:1 34:203287:1 35:797911:1 36:345058:1 37:596634:1 38:111569:1
1 0:40189:1 1:17524:1 2:790098:1 3:25173:1 4:683212:1 5:591159:1 6:834726:1 7:407144:1 8:679173:1 9:255381:1 10:813309:1 11:832677:1 12:130107:1 13:344176:1 14:2533:1 15:125954:1 16:97273:1 17:397270:1 18:182671:1 19:629947:1 20:645612:1 21:198788:1 22:970937:1 23:582961:1 24:150835:1 25:89198:1 26:229832:1 27:48224:1 28:50926:1 29:166818:1 30:946171:1 31:765122:1 32:594502:1 33:926005:1 34:541960:1 35:543189:1 36:924148:1 37:698916:1 38:669560:1
0 0:194689:1 1:219069:1 2:917730:1 3:637235:1 4:478654:1 5:819010:1 6:287534:1 7:680006:1 8:755846:1 9:530364:1 10:124999:1 11:832677:1 12:677475:1 13:126853:1 14:322536:1 15:360605:1 16:854747:1 17:66695:1 18:637420:1 19:803852:1 20:50216:1 21:198788:1 22:537652:1 23:28936:1 24:427192:1 25:814455:1 26:78061:1 27:511252:1 28:654902:1 29:104167:1 30:678329:1 31:388836:1 32:382381:1 33:705479:1 34:541960:1 35:59109:1 36:908719:1 37:607936:1 38:923815:1
'''



'''


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
    sub.to_csv(path + '../sub/sub_lgb0704.csv.gz',index=False, compression = 'gzip')
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

'''