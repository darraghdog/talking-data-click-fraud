import csv
from datetime import datetime
import time
from collections import Counter
#from tqdm import tqdm


data_path = "/home/darragh/tdata/data/"
fbig   = 'old/test.csv'
fsmall = 'test.csv'
ffull  = 'testfull.csv'

def time2string(s):
    t = datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    sec = time.mktime(t.timetuple())
    return sec


small_file_splits = set([])
small_file_splits_id = []
big_file_splits_id = []

iter_small = csv.reader(open(data_path + fsmall))
iter_big   = csv.reader(open(data_path + fbig))

'''
Get where the time splits are in the small file
'''

# Skip the header
test_header, _ = next(iter_small), next(iter_big)
# Get the first row of the small
row = next(iter_small)
s_prev = time2string(row[-1])
small_file_splits.add(','.join(row[1:]))
small_file_splits_id.append(row[0])

for c, row in enumerate(iter_small):
    if c%5000000==0:
        print('Small file line %s'%(c))
    s = time2string(row[-1])
    if s-s_prev > 300:
        small_file_splits.add(','.join(row_prev[1:]))
        small_file_splits.add(','.join(row[1:]))        
        small_file_splits_id.append(row_prev[0])       
        small_file_splits_id.append(row[0])
    s_prev = s
    row_prev = row

small_file_splits.add(','.join(row[1:]))
small_file_splits_id.append(row[0])
small_file_splits_id = [tuple(map(int, small_file_splits_id[i:i+2])) for i in range(0, len(small_file_splits_id), 2)]

'''
Find the time splits are in the big file
'''
# Iterate through the 
row_prev = ''
big_file_splits_id.append(0)
for c, row in enumerate(iter_big):
    if c%5000000==0:
        print('Big file line %s'%(c))
    id_, vals_ = row[0], ','.join(row[1:])
    if vals_ in small_file_splits:
        big_file_splits_id.append(row[0])
big_file_splits_id.append(row[0])
big_file_splits_id = [tuple(map(int, big_file_splits_id[i:i+2])) for i in range(0, len(big_file_splits_id), 2)]


print(50*'*')
print('File Splits for small and big file --->>>')
print(50*'*')
print(small_file_splits_id)
print(big_file_splits_id)
print(50*'*')
'''
**************************************************
small_file_splits_id = [(0, 6202932), (6202933, 12316146), (12316147, 18790468)]
big_file_splits_id = [(0, 21290878), (27493808, 35678696), (41791909, 48109937), (54584258, 57537504)]
**************************************************
'''

'''
Make a full file with part of the big and part of the small file
'''
# Now lets write out a new file
iter_small = csv.reader(open(data_path + fsmall))
iter_big   = csv.reader(open(data_path + fbig))
# Skip the header
test_header, _ = next(iter_small), next(iter_big)
test_header = ['dataset'] + test_header


fo = open(data_path + 'testfull.csv','w')
fo.write('%s\n'%(','.join(test_header)))
for i in range(3):
    for row in iter_big:
        fo.write('0,%s\n'%(','.join(row)))
        if int(row[0]) == big_file_splits_id[i+0][1]:
            break
    for row in iter_small:
        fo.write('1,%s\n'%(','.join(row)))
        if int(row[0]) == small_file_splits_id[i+0][1]:
            break
    for row in iter_big:
        if int(row[0]) == big_file_splits_id[i+1][0]-1:
            break
for row in iter_big:
    fo.write('0,%s\n'%(','.join(row)))
fo.close()

'''
# check all is equal
import pandas as pd
testfull = pd.read_csv(data_path + 'testfull.csv')
test = pd.read_csv(data_path + 'test.csv')
test.equals( testfull[testfull['dataset']==1][test.columns.tolist()].reset_index(drop=True) )
'''

'''
Check for breaks in the full file
'''
iter_full= csv.reader(open(data_path + ffull))
# Skip the header
full_header = next(iter_full)
# Get the first row of the small
row = next(iter_full)
s_prev = time2string(row[-1])

for c, row in enumerate(iter_full):
    if c%5000000==0:
        print('Full file line %s'%(c))
    s = time2string(row[-1])
    if s-s_prev > 300:
        'Big File break'
    s_prev = s
    row_prev = row