import csv
import time
from csv import DictReader

path = "/home/darragh/tdata/data/"
path = '/Users/dhanley2/Documents/tdata/data/'
#path = '/home/ubuntu/tdata/data/'
start_time = time.time()

# For ngrams make sure the prefix is 3 in length
mapper  = {'app':'app', 'device':'dev', 'channel':'chl', 'ip':'ipp', 'os':'oss' }
files_  = ["old/test.csv", "train.csv"]
outname = path + '../fastText/data.txt'

outfile = open(outname, 'w')

for fname in files_:
    for t, row in enumerate(DictReader(open(path+fname))):
        
        if t%1000000 == 0:
            print("Processed Rows : %sM in %s sec, file %s "%( int(t/1e+6), '%0.0f'%(time.time()-start_time), fname))
        
        # delete clicks
        if 'is_attributed' in row:
            del row['is_attributed'], row['attributed_time']
        if 'click_id' in row:
            del row['click_id']
            
        
        # Parse hour 
        date, time_ = row['click_time'].split(' ')
        hour = time_.split(':')[0]
        day_time = '%s%s'%(date.split('-')[2], hour)
        del row['click_time']
        
        # process id
        line_ = ['%s%s'%(mapper[k],v) for (k,v) in row.items()]
        line_ += ['hrr%s'%(hour), 'day%s'%(day_time)]
        
        # write file
        outfile.write('%s\n' % (' '.join(line_)))

outfile.close()

'''
# Command line for fastText
# After installing, from directory /fastText/build
./fasttext cbow -dim 50 -ws 15 -epoch 30 -thread 12 -input ../data.txt -output model
'''