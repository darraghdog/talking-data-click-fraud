# lead_lag_trn_ip_device_os_app
gunzip lead_lag_trn_ip_device_os_app.gz
sed -n 1,1p lead_lag_trn_ip_device_os_app > lead_lag_trn_ip_device_os_appval
sed -n 60000000,122080000p lead_lag_trn_ip_device_os_app >> lead_lag_trn_ip_device_os_appval

sed -n 1,1p lead_lag_trn_ip_device_os_app > lead_lag_tst_ip_device_os_appval
sed -n 144710000,152400000p lead_lag_trn_ip_device_os_app >> lead_lag_tst_ip_device_os_appval
sed -n 162000000,168300000p lead_lag_trn_ip_device_os_app >> lead_lag_tst_ip_device_os_appval
sed -n 175000000,181880000p lead_lag_trn_ip_device_os_app >> lead_lag_tst_ip_device_os_appval
gzip lead_lag_trn_ip_device_os_app
gzip lead_lag_trn_ip_device_os_appval
gzip lead_lag_tst_ip_device_os_appval
