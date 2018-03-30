import csv
import time
from csv import DictReader
import numpy as np
import pandas as pd


path = "/home/darragh/tdata/data/"
path = '/Users/dhanley2/Documents/tdata/data/'
#path = '/home/ubuntu/tdata/data/'
start_time = time.time()

# For ngrams make sure the prefix is 3 in length
mapper  = {'app':'app', 'device':'dev', 'channel':'chl', 'ip':'ipp', 'os':'oss' }
mapper_reverse = dict((v,k) for (k,v) in mapper.items())

fname = '../features/model.vec'
lines_ = []
for t, row in enumerate(csv.reader(open(path+fname), delimiter = ' ')):
    #if t >5:
    #    break
    #row
    for key in mapper_reverse.keys():
        if key in row[0]:
            line_ = map(float, row[1:-1])
            line_ = [ row[0].replace(key, '')  ]+line_
            line_ = [mapper_reverse[key]]+line_
            lines_.append(line_)
    
ftdf = pd.DataFrame(lines_, columns = ['ftkey', 'value']+ ['ftdim%s'%(i) for i in range(len(lines_[0])-2)])   
ftdf.to_csv(path + '../features/ftdim10000.csv.gz',index=False, compression = 'gzip')
