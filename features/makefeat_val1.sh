
## ctr_test_hours_
#gunzip  ctr_test_hours_trn.gz
#sed -n 1,1p ctr_test_hours_trn > ctr_test_hours_trnval
#sed -n 60000000,122080000p ctr_test_hours_trn >> ctr_test_hours_trnval
#sed -n 1,1p ctr_test_hours_trn > ctr_test_hours_tstval
#sed -n 144710000,152400000p  ctr_test_hours_trn >> ctr_test_hours_tstval
#sed -n 162000000,168300000p  ctr_test_hours_trn >> ctr_test_hours_tstval
#sed -n 175000000,181880000p  ctr_test_hours_trn >> ctr_test_hours_tstval
#gzip ctr_test_hours_trnval
#gzip ctr_test_hours_tstval
#gzip ctr_test_hours_trn


## prevdayipchlqty
#gunzip  prevqdayipchlqtytrn.gz
#sed -n 1,1p prevqdayipchlqtytrn > prevqdayipchlqtytrnval
#sed -n 60000000,122080000p prevqdayipchlqtytrn >> prevqdayipchlqtytrnval
#sed -n 1,1p prevqdayipchlqtytrn > prevqdayipchlqtytstval
#sed -n 144710000,152400000p prevqdayipchlqtytrn >> prevqdayipchlqtytstval
#sed -n 162000000,168300000p  prevqdayipchlqtytrn >> prevqdayipchlqtytstval
#sed -n 175000000,181880000p  prevqdayipchlqtytrn >> prevqdayipchlqtytstval
#gzip prevqdayipchlqtytrnval
#gzip prevqdayipchlqtytstval
#gzip prevqdayipchlqtytrn

# prev_hday_clicks_
gunzip  prev_hday_clicks_trn.gz
sed -n 1,1p prev_hday_clicks_trn > prev_hday_clicks_trnval
sed -n 60000000,122080000p prev_hday_clicks_trn >> prev_hday_clicks_trnval
sed -n 1,1p prev_hday_clicks_trn > prev_hday_clicks_tstval
sed -n 144710000,152400000p  prev_hday_clicks_trn >> prev_hday_clicks_tstval
sed -n 162000000,168300000p  prev_hday_clicks_trn >> prev_hday_clicks_tstval
sed -n 175000000,181880000p  prev_hday_clicks_trn >> prev_hday_clicks_tstval
gzip prev_hday_clicks_trnval
gzip prev_hday_clicks_tstval
gzip prev_hday_clicks_trn
