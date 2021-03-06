#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)


getLag = function(df, cols_, fname, path, df_out = FALSE, shift_n = 1){
  df$click_sec = as.numeric(fasttime::fastPOSIXct(df$click_time))
  df$click_time = NULL
  df = df[,c(cols_, "click_sec"), with = F]
  df[, index := 1:nrow(df)]
  setorderv(df, c(cols_, "click_sec"))
  df[,click_sec_shift_lead := shift(click_sec, 1, type = "lead")]
  df[,click_sec_shift_lag  := shift(click_sec, 1, type = "lag")]
  df[,seq_lead := .N:1, by = cols_ ]
  df[,seq_lag  := 1:.N, by = cols_ ]
  df[,click_sec_lead := click_sec_shift_lead - click_sec]
  df[,click_sec_lag := click_sec - click_sec_shift_lag]
  df[seq_lead %in% 1:shift_n, click_sec_lead := -1]
  df[seq_lag %in% 1:shift_n,  click_sec_lag := -1]
  setorderv(df, "index")
  df = df[,.(click_sec_lead, click_sec_lag)]
  if (df_out){
    return(df)
  }else{
    write.csv(df, 
              gzfile(paste0(path, fname)), 
              row.names = F, quote = F)
  }
  gc();gc();gc()
}

getLead = function(df, cols_, fname, path, df_out = FALSE, shift_n = 1){
  df$click_sec = as.numeric(fasttime::fastPOSIXct(df$click_time))
  df$click_time = NULL
  df = df[,c(cols_, "click_sec"), with = F]
  df[, index := 1:nrow(df)]
  setorderv(df, c(cols_, "click_sec"))
  df[,click_sec_shift_lead := shift(click_sec, shift_n, type = "lead")]
  df[,seq_lead := .N:1, by = cols_ ]
  df[,click_sec_lead := click_sec_shift_lead - click_sec]
  df[seq_lead %in% 1:shift_n, click_sec_lead := -1]
  setorderv(df, "index")
  new_name = paste0("click_sec_lead_shift", shift_n)
  setnames(df, "click_sec_lead", new_name)
  df = df[,new_name,with=F]
  return(df)
}

path = '~/tdata/data/'
path = '/Users/dhanley2/Documents/tdata/data/'

############################################
################ Lead & Lag ################
############################################

# Write out the <ip, device, os, channel, app> level
trndf = fread(paste0(path, 'train.csv'))
fname = "lead_lag_trn_ip_device_os_channel_app.gz"
cols_ = c("ip", "device", "os", "app", "channel")
getLag(trndf, cols_, fname, path)


# Write out the <ip, device, os, channel, app> level
trndf = fread(paste0(path, 'trainvalsmall.csv'))
fname = "lead_lag_trn_ip_device_os_channel_appvalsmall.gz"
cols_ = c("ip", "device", "os", "app", "channel")
getLag(trndf, cols_, fname, path)

tstdf = fread(paste0(path, 'testfull.csv'))
setidx = tstdf$dataset
fname = "lead_lag_tst_ip_device_os_channel_app.gz"
cols_ = c("ip", "device", "os", "app", "channel")
feats = getLag(tstdf, cols_, fname, path, TRUE)
write.csv(feats[setidx==1], 
          gzfile(paste0(path, fname)), 
          row.names = F, quote = F)

tstdf = fread(paste0(path, 'testvalsmall.csv'))
fname = "lead_lag_tst_ip_device_os_channel_appvalsmall.gz"
cols_ = c("ip", "device", "os", "app", "channel")
getLag(tstdf, cols_, fname, path)

##################################################

# Write out the <ip, device, os> level
cols_ = c("ip", "device", "os")

trndf = fread(paste0(path, 'train.csv'))
fname = "lead_lag_trn_ip_device_os.gz"
getLag(trndf, cols_, fname, path)


# Write out the <ip, device, os> level
trndf = fread(paste0(path, 'trainvalsmall.csv'))
fname = "lead_lag_trn_ip_device_osvalsmall.gz"
getLag(trndf, cols_, fname, path)

tstdf = fread(paste0(path, 'testfull.csv'))
setidx = tstdf$dataset
fname = "lead_lag_tst_ip_device_os.gz"
feats = getLag(tstdf, cols_, fname, path, TRUE)
write.csv(feats[setidx==1], 
          gzfile(paste0(path, fname)), 
          row.names = F, quote = F)

tstdf = fread(paste0(path, 'testvalsmall.csv'))
fname = "lead_lag_tst_ip_device_osvalsmall.gz"
getLag(tstdf, cols_, fname, path)

#################################################
# Write out the <ip, device, os, channel> level
cols_ = c("ip", "device", "os", "channel")

trndf = fread(paste0(path, 'train.csv'))
fname = "lead_lag_trn_ip_device_os_channel.gz"
getLag(trndf, cols_, fname, path)

# Write out the <ip, device, os, channel> level
trndf = fread(paste0(path, 'trainvalsmall.csv'))
fname = "lead_lag_trn_ip_device_os_channelvalsmall.gz"
getLag(trndf, cols_, fname, path)

tstdf = fread(paste0(path, 'testfull.csv'))
setidx = tstdf$dataset
fname = "lead_lag_tst_ip_device_os_channel.gz"
feats = getLag(tstdf, cols_, fname, path, TRUE)
write.csv(feats[setidx==1], 
          gzfile(paste0(path, fname)), 
          row.names = F, quote = F)

tstdf = fread(paste0(path, 'testvalsmall.csv'))
fname = "lead_lag_tst_ip_device_os_channelvalsmall.gz"
getLag(tstdf, cols_, fname, path)


# Write out the <ip, device, os, channel> level
cols_ = c("ip", "device", "os", "app")

trndf = fread(paste0(path, 'train.csv'))
fname = "lead_lag_trn_ip_device_os_app.gz"
getLag(trndf, cols_, fname, path)

tstdf = fread(paste0(path, 'testfull.csv'))
setidx = tstdf$dataset
fname = "lead_lag_tst_ip_device_os_app.gz"
feats = getLag(tstdf, cols_, fname, path, TRUE)
write.csv(feats[setidx==1], 
          gzfile(paste0(path, fname)), 
          row.names = F, quote = F)

# Write out the <ip, device, os, channel> level -- shift2
cols_ = c("ip", "device", "os", "app")

trndf = fread(paste0(path, 'train.csv'))
fname = "lead2_trn_ip_device_os_app.gz"
feattrn2  = getLead(trndf, cols_, fname, path, TRUE, shift_n = 2)
write.csv(feattrn2, 
          gzfile(paste0(path, fname)), 
          row.names = F, quote = F)

tstdf = fread(paste0(path, 'testfull.csv'))
setidx = tstdf$dataset
fname = "lead2_tst_ip_device_os_app.gz"
featstst2 = getLead(tstdf, cols_, fname, path, TRUE, shift_n = 2)
write.csv(featstst2[setidx==1], 
          gzfile(paste0(path, fname)), 
          row.names = F, quote = F)











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

############################################
################ Lead & Lag ################
############################################
