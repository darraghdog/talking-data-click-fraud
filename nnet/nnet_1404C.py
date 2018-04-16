# https://www.kaggle.com/alexanderkireev/experiments-with-imbalance-nn-architecture
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
import os
#os.environ['OMP_NUM_THREADS'] = '4'
import gc
print ('neural network....')
import tensorflow as tf
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D, Conv1D
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam
import keras.layers as L
from keras import backend as K
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import warnings
from sklearn import metrics
from sklearn.metrics import roc_auc_score
#warnings.filterwarnings("ignore", category=DeprecationWarning)


#path = '../input/'
path = "/home/darragh/tdata/data/"
path = '/Users/dhanley2/Documents/tdata/data/'
path = '/home/ubuntu/tdata/data/'
start_time = time.time()
validation =  True
if validation:
    add_ = 'val'
    test_usecols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
    val_size = 0
else:
    val_size = 10000
    add_ = ''
    test_usecols = ['ip','app','device','os', 'channel', 'click_time', 'click_id']
    
    
def transform_lead(df, bins = 60, nafillfrom = -1, nafillto = 3600, bin_it = True):
    all_cols = df.columns
    for col in all_cols :
        print('Transform col : %s'%(col))
        idx_ = df[col]==nafillfrom
        if bin_it:
            bins_ = bins
            df[col + '__bins'] = pd.qcut(df[col], q = bins_, labels = False, duplicates = 'drop')
            df[col + '__bins'][idx_] = bins + 1
            df[col + '__bins'] = df[col + '__bins'].astype(np.int32)
        df[col][idx_] = nafillto
        df[col] = np.log(df[col]+0.1111111)
    scaler = StandardScaler()
    df[all_cols] = scaler.fit_transform(df[all_cols])
    df[all_cols] = df[all_cols].astype(np.float32)
    for col in all_cols:
        df.rename(columns={col: col+'__scale'}, inplace = True)
    gc.collect()
    return df

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }


def load_feat(fname = '../features/lead_split_sec_trn_ip_device_os_app'):
    df = pd.concat([pd.read_csv(path + fname +'%s.gz'%(add_), compression = 'gzip'),
          pd.read_csv(path+fname.replace('trn', 'tst')+'%s.gz'%(add_), compression = 'gzip')]).astype(np.float32)
    return df


print('load train....')
print('[{}] Load Train'.format(time.time() - start_time))
train_df = pd.read_csv(path+"train%s.csv"%(add_), dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
print('[{}] Load Test'.format(time.time() - start_time))
test_df = pd.read_csv(path+"test%s.csv"%(add_), dtype=dtypes, usecols=test_usecols)

print('[{}] Load Lead/Lag Features'.format(time.time() - start_time))
featapp1 = load_feat('../features/lead_lag_trn_ip_device_os_app')
featapp1.columns = [col+'_app1' for col in featapp1.columns]
featapp1 = transform_lead(featapp1, bins = 50, bin_it = True)
print(featapp1.shape)

print('[{}] Load Lead/Lag Features'.format(time.time() - start_time))
featapp2 = load_feat('../features/lead_split_sec_trn_ip_device_os_app')
featapp2.columns = [col+'_app2' for col in featapp2.columns]
featapp2 = transform_lead(featapp2, bins = 50, bin_it = True)
print(featapp2.shape)
print(featapp2.columns)

print('[{}] Load Lead/Lag Features'.format(time.time() - start_time))
featapp3 = load_feat('../features/lead_split_sec_trn_ip_device_os_appchl')
featapp3.columns = [col+'_app3' for col in featapp3.columns]
featapp3 = transform_lead(featapp3, bins = 50, bin_it = True)
print(featapp3.shape)

print('[{}] Load Lead/Lag Features'.format(time.time() - start_time))
featapp4 = load_feat('../features/lead_lag_trn_ip_device_os_channel')
featapp4.columns = [col+'_app4' for col in featapp4.columns]
featapp4 = transform_lead(featapp4, bins = 50, bin_it = True)
print(featapp4.shape)

print('[{}] Load Lead/Lag Features'.format(time.time() - start_time))
featapp5 = load_feat('../features/lead_lag_trn_ip_device_os_channel_app')
featapp5.columns = [col+'_app5' for col in featapp5.columns]
featapp5 = transform_lead(featapp5, bins = 50, bin_it = True)
print(featapp5.shape)

print('[{}] Load Lead/Lag Split Sec Features'.format(time.time() - start_time))
featspl = load_feat('../features/lead_split_sec_trn_ip_device_os_app')
featspl = transform_lead(featspl, bins = 200, nafillfrom = 999999.000000, nafillto = 3600, bin_it = True)
print(featspl.shape)

print('[{}] Load Lead Count next period'.format(time.time() - start_time))
featctn = load_feat('../features/lead_count_next_ipdevosapp_trn')
featctn = transform_lead(featctn, bin_it = True)
print(featctn.shape)

print('[{}] Load Lead Count next period ipdevos'.format(time.time() - start_time))
featctn1 = load_feat('../features/lead_count_next_ipdevos_trn')
featctn1 = transform_lead(featctn1, bin_it = True)
print(featctn1.shape)

print('[{}] Load Lead Count next period ipdevos'.format(time.time() - start_time))
featld2 = load_feat('../features/lead2_trn_ip_device_os_app')
featld2 = transform_lead(featld2, bin_it = True)
print(featld2.shape)

'''
print('[{}] Level counts'.format(time.time() - start_time))
featctr = load_feat('../features/counttrn')
featctr = transform_lead(featctr, bin_it = False)
'''
print('[{}] Level counts'.format(time.time() - start_time))
featcum = load_feat('../features/cum_min_trn_ip_device_os_app')
featcum = transform_lead(featcum, bin_it = False)
print(featcum.shape)

print('[{}] Load Previous Day Clicks'.format(time.time() - start_time))
featprev1 = load_feat('../features/prevqdayipchlqtytrn')
featprev1.fillna(-1, inplace = True)
featprev1 = transform_lead(featprev1, bin_it = False) 
print(featprev1.shape)

print('[{}] Load Previous Day Clicks'.format(time.time() - start_time))
featprev2 = load_feat('../features/prevdayipchlqtytrn')
featprev2.fillna(-1, inplace = True)
featprev2 = transform_lead(featprev2, bin_it = False)
print(featprev2.shape)



feat =     pd.concat([featapp1, featapp2, featapp3, featapp4, featspl, featctn, \
        featctn1, featld2, featcum, featprev1, featprev2], axis = 1)
del featapp1, featapp2, featapp3, featapp4, featapp5, featspl, featctn
del featctn1, featld2, featcum, featprev1, featprev2
gc.collect()


print('[{}] Load Entropy Features'.format(time.time() - start_time))
def scale_entropy(ffeat):
    cols_ = [c for c in ffeat.columns if c not in ['ip', 'os', 'device']]
    print(cols_)
    colsid_ = [c for c in ffeat.columns if c in ['ip', 'os', 'device']]
    scaler = MinMaxScaler()
    ffeat[cols_] = scaler.fit_transform(ffeat[cols_])
    ffeat[cols_] = ffeat[cols_].astype(np.float16)
    ffeat[colsid_] = ffeat[colsid_].astype('uint32')
    return ffeat

featentip      = scale_entropy(pd.read_csv(path+'../features/entropyip.gz', compression = 'gzip'))
featentipdevos = scale_entropy(pd.read_csv(path+'../features/entropyipdevos.gz', compression = 'gzip'))
featentapp     = scale_entropy(pd.read_csv(path+'../features/entropyapp.gz', compression = 'gzip'))
featentchl     = scale_entropy(pd.read_csv(path+'../features/entropychl.gz', compression = 'gzip'))



print('[{}] Join test and train'.format(time.time() - start_time))
len_train = len(train_df)
train_df=train_df.append(test_df)
del test_df; gc.collect()
print('[{}] Concat Features'.format(time.time() - start_time))
print(train_df.shape)
print(train_df.dtypes)
train_df = pd.concat([train_df, feat], axis = 1)
print(train_df.shape)
print(train_df.dtypes)

#len_train = 1000000
#train_df = train_df[:(len_train*2)]


print('[{}] Add entropy'.format(time.time() - start_time))
train_df = train_df.merge(featentip, on=['ip'], how='left')
train_df = train_df.merge(featentapp, on=['app'], how='left')
train_df = train_df.merge(featentchl, on=['channel'], how='left')
train_df = train_df.merge(featentipdevos, on=['ip', 'device', 'os'], how='left')
print(train_df.shape)

print('[{}] hour, day, wday....'.format(time.time() - start_time))
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
#train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
#train_df['wday']  = pd.to_datetime(train_df.click_time).dt.dayofweek.astype('uint8')

#print("label encoding....")
train_df[['app','device','os', 'channel', 'hour']].apply(LabelEncoder().fit_transform)
print('[{}] Split train/val'.format(time.time() - start_time))
test_df = train_df[len_train:]
train_df = train_df[:len_train]
y_train = train_df['is_attributed'].values
# train_df.drop(['click_id', 'click_time','ip','is_attributed'],1,inplace=True)
train_df.drop(['click_time','ip','is_attributed'],1,inplace=True)


print('[{}] Create model'.format(time.time() - start_time))
embids = ['app', 'channel', 'device', 'os', 'hour']
embids += [col for col in train_df.columns if '_bins' in col]
embsz = {'app': 50, 'channel': 50, 'device':100, 'os': 50, 'hour': 10}
for col in train_df.columns:
    if '_bins' in col:
        embsz[col] = 25

# get the max of each code type
embmaxs = dict((col, np.max([train_df[col].max(), test_df[col].max()])+1) for col in embids)

cont_cols = [c for c in train_df.columns if 'entropy' in c]
cont_cols += [c for c in train_df.columns if '_scale' in c]
# Generator
def get_keras_data(dataset):
    X = dict((col, np.array(dataset[col])) for col in embids)
    for col in cont_cols:
        X[col] = dataset[col].values
    return X

# Dictionary of inputs
emb_n = 40
dense_n = 1000
# Build the inputs, embeddings and concatenate them all for each column
emb_inputs = dict((col, Input(shape=[1], name = col))  for col in embids)
cont_inputs = dict((col, Input(shape=[1], name = col))  for col in cont_cols)
emb_model  = dict((col, Embedding(embmaxs[col], emb_n)(emb_inputs[col])) for col in embids)
fe = concatenate([(emb_) for emb_ in emb_model.values()])
# Rest of the model
s_dout = SpatialDropout1D(0.4)(fe)
fl1 = Flatten()(s_dout)
conv = Conv1D(200, kernel_size=4, strides=1, padding='same')(s_dout)
fl2 = Flatten()(conv)
concat = concatenate([(fl1), (fl2)] + [(c_inp) for c_inp in cont_inputs.values()])
x = Dropout(0.4)(Dense(dense_n,activation='relu')(concat))
x = Dropout(0.4)(Dense(dense_n,activation='relu')(x))
outp = Dense(1,activation='sigmoid')(x)
model = Model(inputs=[inp for inp in emb_inputs.values()] + [(c_inp) for c_inp in cont_inputs.values()], outputs=outp)

# Parameters
batch_size   = 200000
epochs       = 4
blend_epochs = 2

exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(list(train_df)[0]) / batch_size) * epochs
lr_init, lr_fin = 0.0015, 0.0002
lr_decay = exp_decay(lr_init, lr_fin, steps)
optimizer_adam = Adam() #Adam(lr=0.002, decay=lr_decay)
model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['accuracy'])

model.summary()


log = {'val_auc': []}
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))
            log['val_auc'].append(score)

if validation:
    val_df = get_keras_data(test_df)
    y_val = test_df['is_attributed'].values
    y_act = y_val
else:
    click_ids = test_df['click_id'].astype(np.int32)
    #RocAuc = RocAucEvaluation(validation_data=(val_df, y_val), interval=1)

train_df = get_keras_data(train_df)
test_df  = get_keras_data(test_df)
predsls = []

class_weight = {0:.01,1:.99} # magic

print('[{}] Start fitting'.format(time.time() - start_time))
if validation:
    for i in range(epochs):
        model.fit(train_df, 
              y_train, 
              batch_size=batch_size, 
              epochs=1, 
              class_weight=class_weight, 
              validation_data=(val_df, y_val),
              shuffle=True, 
              verbose=2)
        if epochs - i <= blend_epochs:
            print('[{}] Predicting'.format(time.time() - start_time))
            predsls.append(model.predict(test_df, batch_size=batch_size, verbose=2))
            fpr, tpr, thresholds = metrics.roc_curve(y_act, predsls[-1], pos_label=1)
            print('Auc for all hours in testval : %s'%(metrics.auc(fpr, tpr)))
            model.save_weights(path + '../weights/imbalanced_data_epoch%s_%s.h5'%(i, 'val'))
    preds = sum(predsls)/len(predsls)
else:
    for i in range(epochs):
        model.fit(train_df, 
              y_train, 
              batch_size=batch_size, 
              epochs=1, 
              class_weight=class_weight, 
              shuffle=True, 
              verbose=2)
        if epochs - i <= blend_epochs:
            print('[{}] Predicting'.format(time.time() - start_time))
            predsls.append(model.predict(test_df, batch_size=batch_size, verbose=2))
            model.save_weights(path + '../weights/imbalanced_data_epoch%s_%s.h5'%(i, 'full'))
    preds = sum(predsls)/len(predsls)

    
sub = pd.DataFrame()
# test_df.drop(['click_time','ip','is_attributed'],1,inplace=True)

if not validation:    
    print('[{}] Build sub and write'.format(time.time() - start_time))
    sub['click_id'] = click_ids
    sub['is_attributed'] = preds
    del test_df; gc.collect()
    sub.to_csv(path + '../sub/sub_nnet1404C.csv.gz',index=False, compression = 'gzip')
    print(sub.info())
    print('[{}] All done ...'.format(time.time() - start_time))
else:
    print('[{}] Build sub and write'.format(time.time() - start_time))
    sub['click_id'] = range(len(preds))
    fpr, tpr, thresholds = metrics.roc_curve(y_act, preds, pos_label=1)
    print('Auc for all hours in testval : %s'%(metrics.auc(fpr, tpr)))
    sub['is_attributed'] = preds
    sub.to_csv(path + '../sub/sub_nnet1404Cval.csv.gz',index=False, compression = 'gzip')
    print('[{}] All done ...'.format(time.time() - start_time))

