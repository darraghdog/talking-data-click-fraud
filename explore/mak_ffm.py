import csv
import os
import time
import gzip

#path = '../input/'
path = "/home/darragh/tdata/data/"
path = '/Users/dhanley2/Documents/tdata/data/'
#path = '/home/ubuntu/tdata/data/'
start_time = time.time()

fname_tst = path + '../weights/test_df%s.ffm.csv.gz'%(add_)
fname_trn = path + '../weights/train_df%s.ffm.csv.gz'%(add_)


for c,row in enumerate(csv.DictReader(gzip.open(fname_tst))):
    
    y = row[]