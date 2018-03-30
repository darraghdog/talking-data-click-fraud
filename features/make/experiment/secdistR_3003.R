#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)

getSplitLead = function(df, cols_, fname, path, shift_n = 1){
  df$click_sec = as.numeric(fasttime::fastPOSIXct(df$click_time))
  df[, split_sec := round((0:(.N-1))/.N, 4), by = click_time]
  df = df[,c(cols_, "click_sec", "split_sec"), with = F]
  df[, index := 1:nrow(df)]
  setorderv(df, c(cols_, "click_sec", "split_sec"))
  df[,click_sec_shift_lead := shift(click_sec+split_sec, shift_n, type = "lead")]
  df[,seq_lead := .N:1, by = cols_ ]
  df[,click_sec_lead := click_sec_shift_lead - (click_sec + split_sec)]
  df[,click_sec_lead := round(click_sec_lead, 4)]
  df[seq_lead %in% 1:shift_n, click_sec_lead := 999999]
  setorderv(df, "index")
  new_name = "click_sec_lead_split_sec"
  setnames(df, "click_sec_lead", new_name)
  df = df[,new_name,with=F]
  return(df)
}

path = '~/tdata/data/'
#path = '/Users/dhanley2/Documents/tdata/data/'

# Write out the <ip, device, os, channel> level -- shift2
cols_ = c("ip", "device", "os", "app")
trndf = fread(paste0(path, 'trainval.csv'))
fname = "lead_split_sec_trn_ip_device_os_app.gz"
feattrn  = getSplitLead(trndf, cols_, fname, path, TRUE)
trndf[, sec:= as.numeric(fasttime::fastPOSIXct(click_time))]
trndf[, sec_ct:= .N, by = .(sec, ip, device, os, app)]




write.csv(feattrn, 
          gzfile(paste0(path, fname)), 
          row.names = F, quote = F)
rm(featstrn, trndf)
gc()
# table(cut2(feattrn$click_sec_lead_split_sec[1:10000000], g = 100), trndf$is_attributed[1:10000000])

tstdf = fread(paste0(path, 'testfull.csv'))
setidx = tstdf$dataset
fname = "lead_split_sec_tst_ip_device_os_app.gz"
featstst = getSplitLead(tstdf, cols_, fname, path)
write.csv(featstst[setidx==1], 
          gzfile(paste0(path, fname)), 
          row.names = F, quote = F)
rm(featstst, tstdf)
gc()


