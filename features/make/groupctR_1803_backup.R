#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)

path = '~/tdata/data/'
#path = '/Users/dhanley2/Documents/tdata/data/'

# Write out the <ip, device, os> level
trndf = fread(paste0(path, 'train.csv'))
trndf
trndf[,ct_clicks]




# Write out the <ip, device, os> level
#trndf = fread(paste0(path, 'trainvalsmall.csv'))
trndf = trndf[,.(ip, app, device, os, channel, click_time)]
gc();gc()
trndf$click_time = as.numeric(fasttime::fastPOSIXct(trndf$click_time))
gc();gc()
    trndf[,click_day_hr := round(click_time/3600)]
trndf[,click_day_hr := click_day_hr - 419476]
trndf[,click_hr := click_day_hr%%24]
trndf[,click_day_hr_ct:=.N, by = click_day_hr]
trndf[,click_hr_ct:=.N, by = click_hr]

trndf[,click_dh_ratio := click_day_hr_ct / click_hr_ct]


#trndf[, bmean := () ]
trndf[]






trndf$click_sec = fasttime::fastPOSIXct(trndf$click_time)
trndf$click_hr = format(trndf$click_sec, "%D-%H")
trndf[, ctdev:=length(unique(1000000*app)),by=.(ip, click_hr)]
table(trndf$ctdev==1, trndf$is_attributed)



trndf[, ctip:= .N, by=ip]
trndf[, ctipdo:= .N, by=.(ip, device, os,channel)]
trndf[, ctdev1:=length(unique(os+device*1000+1000000*channel)),by=ip]
trndf
table(trndf$ctdev1==1, trndf$is_attributed)


trndf


trndf[, ctdev:=length(unique(os+device*1000+1000000*app)),by=.(device, ip)]
trndf
table(trndf$ctdev==1, trndf$is_attributed)


trndf[, ctdev:=length(unique(app)),by=.(device, ip, os)]
table(trndf$ctdev==1, trndf$is_attributed)
trndf



table(trndf$ctdev==2, trndf$is_attributed)

trndf[, ctdev1:=length(unique(os+device*1000)),by=ip]
trndf

table(trndf$ctdev==1, trndf$is_attributed)
table(trndf$ctdev==2, trndf$is_attributed)


############################################
################ Lead & Lag ################
############################################

# Write out the <ip, device, os> level
cols_ = c("ip", "device", "os")

trndf = fread(paste0(path, 'train.csv'))
fname = "next_trn_ip_device_os.gz"
nextTime(trndf, cols_, fname, path)


# Write out the <ip, device, os> level
trndf = fread(paste0(path, 'trainvalsmall.csv'))
fname = "next_trn_ip_device_osvalsmall.gz"
nextTime(trndf, cols_, fname, path)

tstdf = fread(paste0(path, 'testfull.csv'))
setidx = tstdf$dataset
fname = "next_tst_ip_device_os.gz"
feats = nextTime(tstdf, cols_, fname, path, TRUE)
write.csv(feats[setidx==1], 
          gzfile(paste0(path, fname)), 
          row.names = F, quote = F)

tstdf = fread(paste0(path, 'testvalsmall.csv'))
fname = "next_tst_ip_device_osvalsmall.gz"
nextTime(tstdf, cols_, fname, path)