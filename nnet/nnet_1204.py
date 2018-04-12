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
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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
    
    
def transform_lead(df, bins = 60, nafillfrom = -1, nafillto = 3600):
    for col in df.columns:
        idx_ = df[col]==nafillfrom
        bins_ = bins
        df[col + '_bins'] = pd.qcut(df[col], q = bins_, labels = False, duplicates = 'drop')
        df[col + '_bins'][idx_] = bins + 1
        df[col + '_bins']
        df[col][idx_] = nafillto
        df[col] = np.log(df[col]+0.1111111)
        scaler = StandardScaler().fit(df[col].values.reshape(1, -1))
        df[col] = scaler.fit_transform(df[col].values.reshape(1, -1))
        df.rename(columns={col: col+'_scale'}, inplace = True)
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
print('load train....')
print('[{}] Load Train'.format(time.time() - start_time))
train_df = pd.read_csv(path+"train%s.csv"%(add_), dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
print('[{}] Load Test'.format(time.time() - start_time))
test_df = pd.read_csv(path+"test%s.csv"%(add_), dtype=dtypes, usecols=test_usecols)

print('[{}] Load Lead/Lag Features'.format(time.time() - start_time))
featapp = pd.concat([pd.read_csv(path+'../features/lead_lag_trn_ip_device_os_app%s.gz'%(add_), compression = 'gzip'), \
                    pd.read_csv(path+'../features/lead_lag_tst_ip_device_os_app%s.gz'%(add_), compression = 'gzip')])
featapp = transform_lead(featapp)
'''
print('[{}] Load Entropy Features'.format(time.time() - start_time))
featentip  = pd.read_csv(path+'../features/entropyip.gz', compression = 'gzip')
featentip.iloc[:,1:] = featentip.iloc[:,1:].astype(np.float32)
featentip.iloc[:,0] = featentip.iloc[:,0].astype('uint32')
featentip.columns
featentip['ip_click_min_entropy'].hist()
'''
# featapp['click_sec_lead_scale'].hist()
# featapp['click_sec_lead_bins'].hist()



len_train = len(train_df)
train_df=train_df.append(test_df)
del test_df; gc.collect()
train_df = pd.concat([train_df, featapp], axis = 1)

print('hour, day, wday....')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
train_df['wday']  = pd.to_datetime(train_df.click_time).dt.dayofweek.astype('uint8')

print('grouping by ip-day-hour combination....')
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
del gp; gc.collect()
print('group by ip-app combination....')
gp = train_df[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train_df = train_df.merge(gp, on=['ip','app'], how='left')
del gp; gc.collect()
print('group by ip-app-os combination....')
gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
del gp; gc.collect()
print("vars and data type....")
train_df['qty'] = train_df['qty'].astype('uint16')
train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')

print("label encoding....")
train_df[['app','device','os', 'channel', 'hour', 'day', 'wday']].apply(LabelEncoder().fit_transform)
print ('final part of preparation....')
test_df = train_df[len_train:]
train_df = train_df[:len_train]
y_train = train_df['is_attributed'].values
# train_df.drop(['click_id', 'click_time','ip','is_attributed'],1,inplace=True)
train_df.drop(['click_time','ip','is_attributed'],1,inplace=True)


embids = ['app', 'channel', 'device', 'os', 'hour', 'day', 'wday', 'qty', 'ip_app_count', 'ip_app_os_count']
embids += [col for col in train_df.columns if '_bins' in col]
# get the max of each code type
embmaxs = dict((col, np.max([train_df[col].max(), test_df[col].max()])+1) for col in embids)
# Generator
def get_keras_data(dataset):
    X = dict((col, np.array(dataset[col])) for col in embids)
    return X

# Dictionary of inputs
emb_n = 40
dense_n = 1000
# Build the inputs, embeddings and concatenate them all for each column
emb_inputs = dict((col, Input(shape=[1], name = col))  for col in embids)
emb_model  = dict((col, Embedding(embmaxs[col], emb_n)(emb_inputs[col])) for col in embids)
fe = concatenate([(emb_) for emb_ in emb_model.values()])
# Rest of the model
s_dout = SpatialDropout1D(0.2)(fe)
fl1 = Flatten()(s_dout)
conv = Conv1D(100, kernel_size=4, strides=1, padding='same')(s_dout)
fl2 = Flatten()(conv)
concat = concatenate([(fl1), (fl2)])
x = Dropout(0.2)(Dense(dense_n,activation='relu')(concat))
x = Dropout(0.2)(Dense(dense_n,activation='relu')(x))
outp = Dense(1,activation='sigmoid')(x)
model = Model(inputs=[inp for inp in emb_inputs.values()], outputs=outp)


batch_size = 50000
epochs = 2
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(list(train_df)[0]) / batch_size) * epochs
lr_init, lr_fin = 0.002, 0.0002
lr_decay = exp_decay(lr_init, lr_fin, steps)
optimizer_adam = Adam(lr=0.002, decay=lr_decay)
model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['accuracy'])

model.summary()

from sklearn.metrics import roc_auc_score
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



train_df = get_keras_data(train_df[embids])
if validation:
    val_df = get_keras_data(test_df[embids])
    y_val = test_df['is_attributed'].values
    #RocAuc = RocAucEvaluation(validation_data=(val_df, y_val), interval=1)
    
class_weight = {0:.01,1:.99} # magic


if validation:
    model.fit(train_df, 
          y_train, 
          batch_size=batch_size, 
          epochs=2, 
          class_weight=class_weight, 
          #callbacks=[RocAuc, EarlyStopping()],
          validation_data=(val_df, y_val),
          shuffle=True, 
          verbose=1
          )
    del train_df, val_df, y_val, y_train; gc.collect()
else:
    model.fit(train_df, 
          y_train, 
          batch_size=batch_size, 
          epochs=2, 
          class_weight=class_weight, 
          shuffle=True, 
          verbose=1
          )
    del train_df, y_train; gc.collect()
    
    
model.save_weights(path + '../weights/imbalanced_data.h5')
sub = pd.DataFrame()

if not validation:
    print("Predicting...")
    sub['click_id'] = test_df['click_id'].astype('int')
    test_df.drop(['click_time','ip','is_attributed'],1,inplace=True)
    test_df = get_keras_data(test_df)
    sub['is_attributed'] = model.predict(test_df, batch_size=batch_size, verbose=2)
    del test_df; gc.collect()
    print("writing...")
    sub.to_csv(path + '../sub/sub_lgb0704A.csv.gz',index=False, compression = 'gzip')
    print("done...")
    print(sub.info())
else:
    print("Predicting...")
    sub['click_id'] = range(test_df.shape[0]) #test_df['click_id'].astype('int')
    y_act = test_df['is_attributed'].values
    test_df.drop(['click_time','ip','is_attributed'],1,inplace=True)
    test_df = get_keras_data(test_df)
    sub['is_attributed'] = model.predict(test_df, batch_size=batch_size, verbose=2)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_act, preds, pos_label=1)
    print('Auc for all hours in testval : %s'%(metrics.auc(fpr, tpr)))
    idx = test_df['ip']<=max_ip
    fpr1, tpr1, thresholds1 = metrics.roc_curve(y_act[idx], preds[idx], pos_label=1)
    print('Auc for select hours in testval : %s'%(metrics.auc(fpr1, tpr1)))
    print("writing...")
    sub.to_csv(path + '../sub/sub_lgb0704val.csv.gz',index=False, compression = 'gzip')
    
# Original 
# 62080001/62080001 [==============================] - 699s 11us/step - loss: 0.0016 - acc: 0.9873 - val_loss: 0.0753 - val_acc: 0.9837

