#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)


getLag = function(df, cols_, fname, path){
  df$click_sec = as.numeric(fasttime::fastPOSIXct(df$click_time))
  df$click_time = NULL
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


path = '~/tdata/data/'
path = '/Users/dhanley2/Documents/tdata/data/'

############################################
################ Lead & Lag ################
############################################

# Write out the <ip, device, os, channel, app> level
trndf = fread(paste0(path, 'trainvalsmall.csv'))
fname = "lead_lag_trn_ip_device_os_channel_app_valsmall.gz"
cols_ = c("ip", "device", "os", "app", "channel")
getLag(trndf, cols_, fname, path)

tstdf = fread(paste0(path, 'testvalsmall.csv'))
fname = "lead_lag_tst_ip_device_os_channel_app_valsmall.gz"
cols_ = c("ip", "device", "os", "app", "channel")
getLag(tstdf, cols_, fname, path)

# Write out the <ip, device, os> level
trndf = fread(paste0(path, 'train.csv'))
fname = "lead_lag_trn_ip_device_os.gz"
cols_ = c("ip", "device", "os")
getLag(trndf, cols_, fname, path)

tstdf = fread(paste0(path, 'test.csv'))
fname = "lead_lag_tst_ip_device_os.gz"
cols_ = c("ip", "device", "os")
getLag(tstdf, cols_, fname, path)

  
# Write out the <ip, device, os, channel> level
trndf = fread(paste0(path, 'train.csv'))
fname = "lead_lag_trn_ip_device_os_channel.gz"
cols_ = c("ip", "device", "os", "channel")
getLag(trndf, cols_, fname, path)

tstdf = fread(paste0(path, 'test.csv'))
fname = "lead_lag_tst_ip_device_os_channel.gz"
cols_ = c("ip", "device", "os", "channel")
getLag(tstdf, cols_, fname, path)

############################################
########## Click Rolling Mean ##############
############################################

# Write out the <ip, device, os, channel, app> level
trndf = fread(paste0(path, 'trainvalsmall.csv'))
# fname = "lead_lag_trn_ip_device_os_channel_app_valsmall.gz"
cols_ = c("device", "os", "app")
trndf$click_time = fasttime::fastPOSIXct(trndf$click_time)
trndf$click_sec  = as.numeric(trndf$click_time)
trndf$click_hr   = hour(trndf$click_time)
trndf$click_day  = wday(trndf$click_time)
trndf$click_time = NULL
trndf = trndf[,c(cols_, "click_sec", "click_hr", "click_day"), with = F]
trndf[, index := 1:nrow(trndf)]
setorderv(trndf, c(cols_, "click_sec"))
trndf[,count := length(click_sec), by = cols_ ]
trndf[,count_hour := length(click_sec), by = c(cols_, "click_hr", "click_day") ]
trndf[, click_sec_lead_hr := shift(click_sec, 1, type = "lead") - click_sec, by = c(cols_, "click_hr", "click_day") ]
trndf[count_hour>=4,rmeanhr4 := roll_mean(click_sec_lead_hr, 4, align = "center"), by = c(cols_, "click_hr", "click_day")]
trndf[count_hour>=40,rmeanhr40 := roll_mean(click_sec_lead_hr, 40, align = "center"), by = c(cols_, "click_hr", "click_day")]

# Make the bayesian mean of rolling mean


?RcppRoll::roll_mean
trndf

table(is.na(trndf$rmeanhr40))

