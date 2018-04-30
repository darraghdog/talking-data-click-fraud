# dupeseqtrn
gunzip dupeseqtrn.gz
sed -n 1,1p dupeseqtrn > dupeseqtrnval
sed -n 60000000,122080000p dupeseqtrn >> dupeseqtrnval
sed -n 1,1p dupeseqtrn > dupeseqtstval
sed -n 144710000,152400000p dupeseqtrn >> dupeseqtstval
sed -n 162000000,168300000p dupeseqtrn >> dupeseqtstval
sed -n 175000000,181880000p dupeseqtrn >> dupeseqtstval
gzip dupeseqtrn
gzip dupeseqtrnval
gzip dupeseqtstval



## count_same_in_next_trn
#gunzip count_same_in_next_trn.gz
#sed -n 1,1p count_same_in_next_trn > count_same_in_next_trnval
#sed -n 60000000,122080000p count_same_in_next_trn >> count_same_in_next_trnval
#sed -n 1,1p count_same_in_next_trn > count_same_in_next_tstval
#sed -n 144710000,152400000p count_same_in_next_trn >> count_same_in_next_tstval
#sed -n 162000000,168300000p count_same_in_next_trn >> count_same_in_next_tstval
#sed -n 175000000,181880000p count_same_in_next_trn >> count_same_in_next_tstval
#gzip count_same_in_next_trn
#gzip count_same_in_next_trnval
#gzip count_same_in_next_tstval

## leads_ratios_trn
#gunzip leads_ratios_trn.gz
#sed -n 1,1p leads_ratios_trn > leads_ratios_trnval
#sed -n 60000000,122080000p leads_ratios_trn >> leads_ratios_trnval
#sed -n 1,1p leads_ratios_trn > leads_ratios_tstval
#sed -n 144710000,152400000p leads_ratios_trn >> leads_ratios_tstval
#sed -n 162000000,168300000p leads_ratios_trn >> leads_ratios_tstval
#sed -n 175000000,181880000p leads_ratios_trn >> leads_ratios_tstval
#gzip leads_ratios_trn
#gzip leads_ratios_trnval
#gzip leads_ratios_tstval

## cumsumday_trn
#gunzip cumsumday_trn.gz
#sed -n 1,1p cumsumday_trn > cumsumday_trnval
#sed -n 60000000,122080000p cumsumday_trn >> cumsumday_trnval
#sed -n 1,1p cumsumday_trn > cumsumday_tstval
#sed -n 144710000,152400000p cumsumday_trn >> cumsumday_tstval
#sed -n 162000000,168300000p cumsumday_trn >> cumsumday_tstval
#sed -n 175000000,181880000p cumsumday_trn >> cumsumday_tstval
#gzip cumsumday_trn
#gzip cumsumday_trnval
#gzip cumsumday_tstval

