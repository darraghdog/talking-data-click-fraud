import csv
import collections

#path = '../input/'
path = "/home/darragh/tdata/data/"
path = "/Users/dhanley2/Documents/tdata/data/"

small_test = csv.reader(open(path+'test.csv'))
big_test = csv.reader(open(path+'old/test.csv'))
id_list = []

small_test.next(), big_test.next()

small_row = small_test.next()

big_before = ['', '']
for t, big_row in enumerate(big_test):
    if t%10000001==10000000:
        print t/10000000
    if big_row[1:] == big_before[1:]:
        print('Duplicate....')
    big_before = big_row 
    if small_row[1:] == big_row[1:]:
        id_list.append(1)
        small_row = small_test.next()
    else:
        id_list.append(0)
        
collections.Counter(id_list)

big_row