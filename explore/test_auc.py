import pandas as pd
from sklearn import metrics

data_path = "/home/darragh/tdata/data/"
yact  = pd.read_csv(data_path + 'yvalsmall.csv')
ypred = pd.read_csv(data_path + '../sub/sub_probasmall.csv')
ypred = pd.read_csv(data_path + '../explore/sub_probasmall.csv')
yact.columns = ['id', 'is_attributed']

print(ypred.shape)
print(yact.shape)

fpr, tpr, thresholds = metrics.roc_curve(yact['is_attributed'].values, ypred['is_attributed'], pos_label=1)
print(metrics.auc(fpr, tpr))

# 0.95945