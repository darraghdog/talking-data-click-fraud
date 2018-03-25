sed -n 1,1p train.csv > trainval.csv
sed -n 60000000,122080000p train.csv >> trainval.csv

sed -n 1,1p train.csv > testval.csv
sed -n 144710000,152400000p train.csv >> testval.csv
sed -n 162000000,168300000p train.csv >> testval.csv
sed -n 175000000,181880000p train.csv >> testval.csv
