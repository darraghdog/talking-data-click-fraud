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
trndf = fread(paste0(path, 'trainvalsmall.csv'))
trndf

trndf[, ctip:= .N, by=ip]
trndf[, ctipdo:= .N, by=.(ip, device, os)]
trndf[, ctdev:=length(unique(os*100000+device)),by=ip]
trndf

hist(trndf$ctdev)
table(cut2(trndf$ctdev, g = 20), trndf$is_attributed, trndf$ipdo<100)


table(trndf$ctdev==1, trndf$is_attributed, trndf$ctipdo<2)

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