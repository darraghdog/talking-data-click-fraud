#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)


getLag = function(df, cols_, fname, path){
  df = df[,c(cols_, "click_sec"), with = F]
  df[, index := 1:nrow(df)]
  setorderv(df, c(cols_, "click_sec"))
  df[,click_sec_shift_lead := shift(click_sec, 1, type = "lead")]
  df[,click_sec_shift_lag  := shift(click_sec, 1, type = "lag")]
  df[,seq_lead := .N:1, by = .(ip, device, os) ]
  df[,seq_lag  := 1:.N, by = .(ip, device, os) ]
  df[,click_sec_lead := click_sec_shift_lead - click_sec]
  df[,click_sec_lag := click_sec - click_sec_shift_lag]
  df[seq_lead==1, click_sec_lead := -1]
  df[seq_lag==1,  click_sec_lag := -1]
  setorderv(df, "index")
  df[,.(click_sec_lead, click_sec_lag)]
  write.csv(df[,.(click_sec_lead, click_sec_lag)], 
            gzfile(paste0(path, fname)), 
            row.names = F, quote = F)
}

# Write out the 
path = '~/tdata/data/'
path = '/Users/dhanley2/Documents/tdata/data/'
trndf = fread(paste0(path, 'train.csv'))
gc();gc();gc()
trndf$click_sec = as.numeric(fasttime::fastPOSIXct(trndf$click_time))
trndf$click_time = NULL

fname = "lead_lag_trn_ip_device_os_tmp.gz"
cols_ = c("ip", "device", "os")
getLag(trndf, cols_, fname, path)
  
  
# Write out the 
path = '~/tdata/data/'
path = '/Users/dhanley2/Documents/tdata/data/'
trndf = fread(paste0(path, 'train.csv'))
gc();gc();gc()
trndf$click_sec = as.numeric(fasttime::fastPOSIXct(trndf$click_time))
trndf$click_time = NULL

fname = "lead_lag_trn_ip_device_os_channel.gz"
cols_ = c("ip", "device", "os", "channel")
getLag(trndf, cols_, fname, path)



