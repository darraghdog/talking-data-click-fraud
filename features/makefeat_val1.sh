
# ctr_test_hours_
gunzip  ctr_test_hours_trn.gz
sed -n 1,1p ctr_test_hours_trn > ctr_test_hours_trnval
sed -n 60000000,122080000p ctr_test_hours_trn >> ctr_test_hours_trnval
sed -n 1,1p ctr_test_hours_trn > ctr_test_hours_tstval
sed -n 144710000,152400000p  ctr_test_hours_trn >> ctr_test_hours_tstval
sed -n 162000000,168300000p  ctr_test_hours_trn >> ctr_test_hours_tstval
sed -n 175000000,181880000p  ctr_test_hours_trn >> ctr_test_hours_tstval
gzip ctr_test_hours_trnval
gzip ctr_test_hours_tstval
gzip ctr_test_hours_trn
