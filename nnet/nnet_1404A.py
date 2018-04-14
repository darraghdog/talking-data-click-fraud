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
    
    
def transform_lead(df, bins = 60, nafillfrom = -1, nafillto = 3600):
    all_cols = df.columns
    for col in all_cols :
        print('Transform col : %s'%(col))
        idx_ = df[col]==nafillfrom
        bins_ = bins
        df[col + '_bins'] = pd.qcut(df[col], q = bins_, labels = False, duplicates = 'drop')
        df[col + '_bins'][idx_] = bins + 1
        df[col + '_bins'] = df[col + '_bins'].astype(np.int32)
        df[col][idx_] = nafillto
        df[col] = np.log(df[col]+0.1111111)
    scaler = StandardScaler()
    df[all_cols] = scaler.fit_transform(df[all_cols])
    df[all_cols] = df[all_cols].astype(np.float32)
    for col in all_cols:
        df.rename(columns={col: col+'_scale'}, inplace = True)
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
print('load train....')
print('[{}] Load Train'.format(time.time() - start_time))
train_df = pd.read_csv(path+"train%s.csv"%(add_), dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
print('[{}] Load Test'.format(time.time() - start_time))
test_df = pd.read_csv(path+"test%s.csv"%(add_), dtype=dtypes, usecols=test_usecols)

print('[{}] Load Lead/Lag Features'.format(time.time() - start_time))
featapp = pd.concat([pd.read_csv(path+'../features/lead_lag_trn_ip_device_os_app%s.gz'%(add_), compression = 'gzip'), \
                    pd.read_csv(path+'../features/lead_lag_tst_ip_device_os_app%s.gz'%(add_), compression = 'gzip')])
featapp = transform_lead(featapp)

print('[{}] Load Lead/Lag Split Sec Features'.format(time.time() - start_time))
featspl = pd.concat([pd.read_csv(path+'../features/lead_split_sec_trn_ip_device_os_app%s.gz'%(add_), compression = 'gzip'),
                        pd.read_csv(path+'../features/lead_split_sec_tst_ip_device_os_app%s.gz'%(add_), compression = 'gzip')]).astype(np.float32)
featspl = transform_lead(featspl, bins = 200, nafillfrom = 999999.000000, nafillto = 3600)

print('[{}] Load Lead Count next period'.format(time.time() - start_time))
featctn  = pd.concat([pd.read_csv(path+'../features/lead_count_next_ipdevosapp_trn%s.gz'%(add_), compression = 'gzip'),
                      pd.read_csv(path+'../features/lead_count_next_ipdevosapp_tst%s.gz'%(add_), compression = 'gzip')]).astype(np.int16)
featctn = transform_lead(featctn)

print('[{}] Load Previous Day Clicks'.format(time.time() - start_time))
featprev  = pd.concat([pd.read_csv(path+'../features/prevdayipchlqtytrn%s.gz'%(add_), compression = 'gzip'),
                       pd.read_csv(path+'../features/prevdayipchlqtytst%s.gz'%(add_), compression = 'gzip')])
featprev.fillna(-1, inplace = True)
featprev = transform_lead(featprev)

print('[{}] Load Entropy Features'.format(time.time() - start_time))
featentip  = pd.read_csv(path+'../features/entropyip.gz', compression = 'gzip')
featentip.iloc[:,1:] = featentip.iloc[:,1:].astype(np.float32)
featentip.iloc[:,0] = featentip.iloc[:,0].astype('uint32')
scaler = MinMaxScaler()
cols_ = [c for c in featentip.columns if c != 'ip']
featentip[cols_] = scaler.fit_transform(featentip[cols_])
featentip[cols_] = featentip[cols_].astype(np.float16)


len_train = len(train_df)
train_df=train_df.append(test_df)
del test_df; gc.collect()
print('[{}] Concat Features'.format(time.time() - start_time))
train_df = pd.concat([train_df, featapp, featspl, featctn], axis = 1)


#len_train = 1000000
#train_df = train_df[:(len_train*2)]


print('[{}] Add entropy'.format(time.time() - start_time))
train_df = train_df.merge(featentip, on=['ip'], how='left')

print('[{}] hour, day, wday....'.format(time.time() - start_time))
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
train_df['wday']  = pd.to_datetime(train_df.click_time).dt.dayofweek.astype('uint8')

print('[{}] grouping by ip-day-hour combination'.format(time.time() - start_time))
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
del gp; gc.collect()
print('[{}] group by ip-app combination'.format(time.time() - start_time))
gp = train_df[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train_df = train_df.merge(gp, on=['ip','app'], how='left')
del gp; gc.collect()
print('[{}] group by ip-app-os combination'.format(time.time() - start_time))
gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
del gp; gc.collect()

print('[{}] vars and data type....'.format(time.time() - start_time))
num_vars = ['qty', 'ip_app_count', 'ip_app_os_count']
train_dfnv = transform_lead(train_df[num_vars])
train_df.drop(num_vars, 1, inplace = True)
# train_dfnv.hist()
train_df = pd.concat([train_df, train_dfnv], axis = 1)
del train_dfnv; gc.collect()
train_df.columns

'''
print('[{}] Get the outersections and label encode'.format(time.time() - start_time))
for col in ['app','device','os', 'channel', 'hour']:
    print('outsersect cols : %s'%(col))
    col_intersect = np.intersect1d(train_df[col][len_train:].unique(), train_df[col][:len_train].unique())
    outersect_ = int(max(col_intersect)+1)
    labels = np.append(col_intersect, [outersect_])
    le = LabelEncoder()
    le.fit(labels)
    train_df[col][~train_df[col].isin(col_intersect)] = outersect_
    train_df[col] = le.transform(train_df[col])
    del col_intersect, outersect_, le, labels
    gc.collect()


'''

#print("label encoding....")
train_df[['app','device','os', 'channel', 'hour', 'day', 'wday']].apply(LabelEncoder().fit_transform)
print('[{}] Split train/val'.format(time.time() - start_time))
test_df = train_df[len_train:]
train_df = train_df[:len_train]
y_train = train_df['is_attributed'].values
# train_df.drop(['click_id', 'click_time','ip','is_attributed'],1,inplace=True)
train_df.drop(['click_time','ip','is_attributed'],1,inplace=True)

print('[{}] Create model'.format(time.time() - start_time))
embids = ['app', 'channel', 'device', 'os', 'hour']
embids += [col for col in train_df.columns if '_bins' in col]
# Make the size of the embeddings
embsz = dict([(c, 100) if '_bins' in c else (c, 50) for c in embids])
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
dense_n1, dense_n2 = 1000, 100
# Build the inputs, embeddings and concatenate them all for each column
emb_inputs = dict((col, Input(shape=[1], name = col))  for col in embids)
cont_inputs = dict((col, Input(shape=[1], name = col))  for col in cont_cols)
emb_model  = dict((col, Embedding(embmaxs[col], embsz[col], name= 'emb_'+col)(emb_inputs[col])) for col in embids)
# Sum the embeddings of the continuous
embbin_sum = L.Lambda(lambda x: K.sum(x, axis=0), name = 'emb_sum_binned_cols')([t for t in emb_model.values() if '_bins' in t.name])
emb_model['emb_sum_binned_cols'] = embbin_sum
# Concat the sum of the contiuous with the categorical
fe = concatenate([(emb_) for emb_ in emb_model.values() if not '_bins' in emb_.name])
# Rest of the model
s_dout = SpatialDropout1D(0.4)(fe)
fl1 = Flatten()(s_dout)
conv = Conv1D(200, kernel_size=4, strides=1, padding='same')(s_dout)
fl2 = Flatten()(conv)
concat = concatenate([(fl1), (fl2)] + [(c_inp) for c_inp in cont_inputs.values()])
x = Dropout(0.4)(Dense(dense_n1,activation='relu')(concat))
x = Dropout(0.4)(Dense(dense_n2,activation='relu')(x))
outp = Dense(1,activation='sigmoid')(x)
model = Model(inputs=[inp for inp in emb_inputs.values()] + [(c_inp) for c_inp in cont_inputs.values()], outputs=outp)

print('[{}] Plot the model'.format(time.time() - start_time))
from keras.utils import plot_model
plot_model(model, to_file=path+'../nnet/plot/model_nnet1404A.png', show_shapes = True)

# Parameters
batch_size   = 200000
epochs       = 5
blend_epochs = 3

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
    sub.to_csv(path + '../sub/sub_lgb1404A.csv.gz',index=False, compression = 'gzip')
    print(sub.info())
    print('[{}] All done ...'.format(time.time() - start_time))
else:
    print('[{}] Build sub and write'.format(time.time() - start_time))
    sub['click_id'] = range(len(preds))
    fpr, tpr, thresholds = metrics.roc_curve(y_act, preds, pos_label=1)
    print('Auc for all hours in testval : %s'%(metrics.auc(fpr, tpr)))
    sub['is_attributed'] = preds
    sub.to_csv(path + '../sub/sub_lgb1404val.csv.gz',index=False, compression = 'gzip')
    print('[{}] All done ...'.format(time.time() - start_time))

