import csv
import time as timer
from csv import DictReader
from math import exp, log, sqrt
import time
import collections

# TL; DR, the main training process starts on line: 250,
# you may want to start reading the code from there


# A, paths
data_path = "/home/darragh/tdata/data/"
train = data_path + 'trainval.csv'  #'trainval.csv'   # path to training file
test = data_path +  'testval.csv'   #'testval.csv'    # path to testing file
testact = data_path +  'yval.csv'   #'testval.csv'    # path to testing file

feattrn = data_path + '../features/trnval1103.csv'
feattst = data_path + '../features/tstval1103.csv'

# Set up containers to hold ids
keys   = {'ip' : {}, 'app' : {}, 'device' : {}, 'os' : {}, 'channel' : {}, 'click_hour': {}, 
          'app_channel' : {}, 'os_channel' : {}}
key_mapper = collections.OrderedDict((k, c) for (c, k) in enumerate(keys.keys()))

print('Ids .... ')
print(key_mapper)
counts = keys.copy()

def get_ids(path, keys, counts):
    
    counter_ = dict((k, 1) for k in keys.keys())
    
    for k in keys.keys():
        counts[k] = {}
    
    start_time = timer.time()
    for t, row in enumerate(DictReader(open(path))):
        
        if t%10000000==0:
            print("Get ids Processed Rows : %sM ; %ss "%( int(t/1e+6), '%0.0f'%(timer.time()-start_time)))
        
        del row['is_attributed'], row['attributed_time']
        
        # Parse hour
        date, time = row['click_time'].split(' ')
        row['click_hour'] = time.split(':')[0]
        del row['click_time']
        
        row['app_channel'] = '%s_%s'%(row['app'], row['channel'])
        row['os_channel']  = '%s_%s'%(row['os'], row['channel'])
        
            
        for k, v in row.items():
            if v not in keys[k]:
                keys[k][v] = counter_[k]
                counter_[k] += 1
    '''
            try:
                counts[k][v] += 1
            except:
                counts[k][v] = 1
    
                
    print('Keys counter ....')
    print(counter_)
    
    # Remove low counts and reset the indices
    for k_ct, v_ct in counts.items():
        freq_ = set([k for (k, v) in v_ct.items() if v > 4])
        keys[k_ct] = dict((k, v) for (k, v) in keys[k_ct].items() if k in freq_)
        keys[k_ct] = dict((k, c+2) for (c, (k, v)) in enumerate(keys[k_ct].items()))
    '''
    
    return keys, counts 

keys, counts = get_ids(train, keys, counts)

for k, v in keys.items():
    print("Keys, counts : %s  -   %s "%(k, len(v)))
    


##############################################################################
# build feature files ########################################################
##############################################################################
    



def make_file(fname, path, keys, yfile = None):
    if yfile:
        start_time = timer.time()
        test_act = []
        for t, row in enumerate(csv.reader(open(yfile))):
            if t > 0:    
                test_act.append(row[1]) 
            
            if t%10000000==0:
                print("load actuals : %sM ; %ss "%( int(t/1e+6), '%0.0f'%(timer.time()-start_time)))
        print(collections.Counter(test_act))
    
    missing_keys = {}
        
    with open(fname, 'w') as outfile:
        start_time = timer.time()
        for t, row in enumerate(DictReader(open(path))):
            if t%10000000==0:
                print("Write file processed rows : %sM ; %ss "%( int(t/1e+6), '%0.0f'%(timer.time()-start_time)))
        
            # Make features
            row['app_channel'] = '%s_%s'%(row['app'], row['channel'])
            row['os_channel']  = '%s_%s'%(row['os'], row['channel'])
            date, time = row['click_time'].split(' ')
            row['click_hour'] = time.split(':')[0]
            del row['click_time']
            
            # Make target
            if 'is_attributed' in row:
                out = ' ' + row['is_attributed']
                del row['is_attributed'], row['attributed_time']
            else:
                out = ' ' + test_act[t]
                
            try:
                del row['click_id']
            except:
                1
            
            # Add features
            for kidx, vidx in key_mapper.items():
                v = row[kidx]
                if v in keys[kidx]:
                    out += ' %s:%s:1'%(vidx, keys[kidx][v])
                else:
                    # out += ' %s:%s:1'%(vidx, 1) # 1 is a placeholder for low occurence or missing value
                    if kidx not in missing_keys:
                        missing_keys[kidx] = []
                    missing_keys[kidx].append(v)
            out = out[1:]
            
            outfile.write('%s\n' % (out))
        
        print('Files missing key counts...')
        for k, v in missing_keys.items():
            print(k, len(list(set(v))),len(list(v)))

#x{'ip': 0, 'app': 1, 'device': 2, 'os': 3, 'channel': 4, 'click_hour': 5, 'app_channel': 6, 'os_channel': 7}

print('Start train file ....')
make_file(feattrn, train, keys)
print('Start test file ....')
make_file(feattst, test, keys, testact)
        
        
    
'''
start_time = time.time()

with open(submission, 'w') as outfile:
    outfile.write('click_id,is_attributed\n')
    for t, x, y, date, click_id in data(test, D):
        p = learner.predict(x)
        outfile.write('%s,%s\n' % (click_id, str(p)))
        if t%1000000 == 0:
            print("Test Processed Rows : %sM ; %ss "%( int(t/1e+6), '%0.0f'%(time.time()-start_time)))
            
            
darragh@darragh-xps:~/pkgs/libffm/libffm_toy$ head criteo.tr.r100.gbdt0.ffm 
0 0:194689:1 1:219069:1 2:319325:1 3:754755:1 4:515702:1 5:359746:1 6:707332:1 7:396128:1 8:650556:1 9:530364:1 10:148557:1 11:548643:1 12:790331:1 13:344176:1 14:145710:1 15:299224:1 16:901863:1 17:66695:1 18:182671:1 19:832740:1 20:869462:1 21:198788:1 22:660059:1 23:134912:1 24:430579:1 25:285442:1 26:78061:1 27:660466:1 28:326359:1 29:633290:1 30:271271:1 31:765122:1 32:322020:1 33:943765:1 34:541960:1 35:821014:1 36:428618:1 37:607936:1 38:142693:1
0 0:40189:1 1:164503:1 2:659433:1 3:100700:1 4:224808:1 5:463263:1 6:608573:1 7:223864:1 8:169525:1 9:255381:1 10:756430:1 11:832677:1 12:429274:1 13:370671:1 14:226100:1 15:98199:1 16:218827:1 17:397270:1 18:316115:1 19:561396:1 20:50216:1 21:198788:1 22:693070:1 23:859765:1 24:811335:1 25:812374:1 26:198506:1 27:581745:1 28:809536:1 29:166818:1 30:642460:1 31:998174:1 32:614600:1 33:34050:1 34:541960:1 35:543189:1 36:101169:1 37:335222:1 38:975766:1
1 0:194689:1 1:17524:1 2:790098:1 3:25173:1 4:543954:1 5:239747:1 6:946867:1 7:761173:1 8:248150:1 9:530364:1 10:523673:1 11:365321:1 12:130107:1 13:161559:1 14:421400:1 15:98199:1 16:218827:1 17:397270:1 18:182671:1 19:624875:1 20:50216:1 21:198788:1 22:580421:1 23:853737:1 24:811335:1 25:173288:1 26:78061:1 27:740257:1 28:809536:1 29:43420:1 30:952896:1 31:998174:1 32:614600:1 33:34050:1 34:203287:1 35:821014:1 36:101169:1 37:335222:1 38:975766:1

darragh@darragh-xps:~/pkgs/libffm/libffm_toy$ head criteo.va.r100.gbdt0.ffm
1 0:290489:1 1:17524:1 2:790098:1 3:25173:1 4:230523:1 5:786172:1 6:166412:1 7:396128:1 8:582090:1 9:512211:1 10:982324:1 11:548643:1 12:130107:1 13:344176:1 14:704452:1 15:98079:1 16:550602:1 17:397270:1 18:182671:1 19:942674:1 20:275751:1 21:198788:1 22:318202:1 23:900144:1 24:540660:1 25:134997:1 26:198506:1 27:561281:1 28:433710:1 29:633290:1 30:577012:1 31:7551:1 32:382381:1 33:758475:1 34:203287:1 35:797911:1 36:345058:1 37:596634:1 38:111569:1
1 0:40189:1 1:17524:1 2:790098:1 3:25173:1 4:683212:1 5:591159:1 6:834726:1 7:407144:1 8:679173:1 9:255381:1 10:813309:1 11:832677:1 12:130107:1 13:344176:1 14:2533:1 15:125954:1 16:97273:1 17:397270:1 18:182671:1 19:629947:1 20:645612:1 21:198788:1 22:970937:1 23:582961:1 24:150835:1 25:89198:1 26:229832:1 27:48224:1 28:50926:1 29:166818:1 30:946171:1 31:765122:1 32:594502:1 33:926005:1 34:541960:1 35:543189:1 36:924148:1 37:698916:1 38:669560:1
0 0:194689:1 1:219069:1 2:917730:1 3:637235:1 4:478654:1 5:819010:1 6:287534:1 7:680006:1 8:755846:1 9:530364:1 10:124999:1 11:832677:1 12:677475:1 13:126853:1 14:322536:1 15:360605:1 16:854747:1 17:66695:1 18:637420:1 19:803852:1 20:50216:1 21:198788:1 22:537652:1 23:28936:1 24:427192:1 25:814455:1 26:78061:1 27:511252:1 28:654902:1 29:104167:1 30:678329:1 31:388836:1 32:382381:1 33:705479:1 34:541960:1 35:59109:1 36:908719:1 37:607936:1 38:923815:1
'''