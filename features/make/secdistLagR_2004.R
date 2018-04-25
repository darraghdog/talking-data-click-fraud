#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)

getSplitLag = function(df, cols_, fname, path, shift_n = 1){
  df$click_sec = as.numeric(fasttime::fastPOSIXct(df$click_time))
  df[, split_sec := round((0:(.N-1))/.N, 4), by = click_time]
  df = df[,c(cols_, "click_sec", "split_sec"), with = F]
  df[, index := 1:nrow(df)]
  setorderv(df, c(cols_, "click_sec", "split_sec"))
  df[,click_sec_shift_lag := shift(click_sec+split_sec, shift_n, type = "lag")]
  df[,seq_lag := 1:.N, by = cols_ ]
  df[,click_sec_lag := click_sec_shift_lag - (click_sec + split_sec)]
  df[,click_sec_lag := round(click_sec_lag, 4)]
  df[seq_lag %in% 1:shift_n, click_sec_lag := 1]
  df[click_sec_lag < (-3600*6), click_sec_lag := 1]
  setorderv(df, "index")
  new_name = "click_sec_lag_split_sec"
  setnames(df, "click_sec_lag", new_name)
  df = df[,new_name,with=F]
  return(df)
}

path = '~/tdata/data/'
path = '/Users/dhanley2/Documents/tdata/data/'

# Write out the <ip, device, os, channel> level -- shift2
cols_ = c("ip", "device", "os", "app")
trndf = fread(paste0(path, 'train.csv'))
fname = "lag_split_sec_trn_ip_device_os_app.gz"
feattrn  = getSplitLag(trndf, cols_, fname, path, TRUE)
write.csv(feattrn, 
          gzfile(paste0(path, fname)), 
          row.names = F, quote = F)
rm(featstrn, trndf)
gc()

tstdf = fread(paste0(path, 'testfull.csv'))
setidx = tstdf$dataset
fname = "lag_split_sec_tst_ip_device_os_app.gz"
featstst = getSplitLag(tstdf, cols_, fname, path)
write.csv(featstst[setidx==1], 
          gzfile(paste0(path, fname)), 
          row.names = F, quote = F)
rm(featstst, tstdf)
gc()
