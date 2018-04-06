# lead_count_next_ipdevosapp_trn
gunzip lead_count_next_ipdevosapp_trn.gz
sed -n 1,1p lead_count_next_ipdevosapp_trn > lead_count_next_ipdevosapp_trnval
sed -n 60000000,122080000p lead_count_next_ipdevosapp_trn >> lead_count_next_ipdevosapp_trnval

sed -n 1,1p lead_count_next_ipdevosapp_trn > lead_count_next_ipdevosapp_tstval
sed -n 144710000,152400000p lead_count_next_ipdevosapp_trn >> lead_count_next_ipdevosapp_tstval
sed -n 162000000,168300000p lead_count_next_ipdevosapp_trn >> lead_count_next_ipdevosapp_tstval
sed -n 175000000,181880000p lead_count_next_ipdevosapp_trn >> lead_count_next_ipdevosapp_tstval
gzip lead_count_next_ipdevosapp_trn
gzip lead_count_next_ipdevosapp_trnval
gzip lead_count_next_ipdevosapp_tstval


## lead_split_sec_trn_ip
#gunzip lead_split_sec_trn_ip.gz
#sed -n 1,1p lead_split_sec_trn_ip > lead_split_sec_trn_ipval
#sed -n 60000000,122080000p lead_split_sec_trn_ip >> lead_split_sec_trn_ipval

#sed -n 1,1p lead_split_sec_trn_ip > lead_split_sec_tst_ipval
#sed -n 144710000,152400000p lead_split_sec_trn_ip >> lead_split_sec_tst_ipval
#sed -n 162000000,168300000p lead_split_sec_trn_ip >> lead_split_sec_tst_ipval
#sed -n 175000000,181880000p lead_split_sec_trn_ip >> lead_split_sec_tst_ipval
#gzip lead_split_sec_trn_ip
#gzip lead_split_sec_trn_ipval
#gzip lead_split_sec_tst_ipval


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

