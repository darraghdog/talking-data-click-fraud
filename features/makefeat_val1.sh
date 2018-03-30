
# prevdayipchlqty
gunzip  prevdayipchlqtytrn.gz
sed -n 1,1p prevdayipchlqtytrn > prevdayipchlqtytrnval
sed -n 60000000,122080000p prevdayipchlqtytrn >> prevdayipchlqtytrnval
sed -n 1,1p prevdayipchlqtytrn > prevdayipchlqtytstval
sed -n 144710000,152400000p prevdayipchlqtytrn >> prevdayipchlqtytstval
sed -n 162000000,168300000p  prevdayipchlqtytrn >> prevdayipchlqtytstval
sed -n 175000000,181880000p  prevdayipchlqtytrn >> prevdayipchlqtytstval
gzip prevdayipchlqtytrnval
gzip prevdayipchlqtytstval
gzip prevdayipchlqtytrn
