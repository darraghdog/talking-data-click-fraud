# cum_min_trn_ip_device_os_app
gunzip cum_min_trn_ip_device_os_app.gz
sed -n 1,1p cum_min_trn_ip_device_os_app > cum_min_trn_ip_device_os_appval
sed -n 60000000,122080000p cum_min_trn_ip_device_os_app >> cum_min_trn_ip_device_os_appval

sed -n 1,1p cum_min_trn_ip_device_os_app > cum_min_tst_ip_device_os_appval
sed -n 144710000,152400000p cum_min_trn_ip_device_os_app >> cum_min_tst_ip_device_os_appval
sed -n 162000000,168300000p cum_min_trn_ip_device_os_app >> cum_min_tst_ip_device_os_appval
sed -n 175000000,181880000p cum_min_trn_ip_device_os_app >> cum_min_tst_ip_device_os_appval
gzip cum_min_trn_ip_device_os_app
gzip cum_min_trn_ip_device_os_appval
gzip cum_min_tst_ip_device_os_appval

# lead2_trn_ip_device_os_app
gunzip lead2_trn_ip_device_os_app.gz
sed -n 1,1p lead2_trn_ip_device_os_app > lead2_trn_ip_device_os_appval
sed -n 60000000,122080000p lead2_trn_ip_device_os_app >> lead2_trn_ip_device_os_appval

sed -n 1,1p lead2_trn_ip_device_os_app > lead2_tst_ip_device_os_appval
sed -n 144710000,152400000p lead2_trn_ip_device_os_app >> lead2_tst_ip_device_os_appval
sed -n 162000000,168300000p lead2_trn_ip_device_os_app >> lead2_tst_ip_device_os_appval
sed -n 175000000,181880000p lead2_trn_ip_device_os_app >> lead2_tst_ip_device_os_appval
gzip lead2_trn_ip_device_os_app
gzip lead2_trn_ip_device_os_appval
gzip lead2_tst_ip_device_os_appval
