#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)

nextTime = function(df, cols_, fname, path, df_out = FALSE){
  df$click_sec = as.numeric(fasttime::fastPOSIXct(df$click_time))
  df$click_time = NULL
  df = df[,c(cols_, "click_sec", "app", "channel"), with = F]
  df[, index := 1:nrow(df)]
  setorderv(df, c(cols_, "click_sec"))
  df[,app_shift_lead := shift(app, 1, type = "lead")]
  df[,app_shift_lag  := shift(app, 1, type = "lag")]
  df[,chl_shift_lead := shift(channel, 1, type = "lead")]
  df[,chl_shift_lag  := shift(channel, 1, type = "lag")]
  df[,seq_lead := .N:1, by = cols_ ]
  df[,seq_lag  := 1:.N, by = cols_ ]
  df[,same_next_app := as.numeric(app == app_shift_lead)]
  df[,same_prev_app := as.numeric(app == app_shift_lag)]
  df[,same_next_chl := as.numeric(channel == chl_shift_lead)]
  df[,same_prev_chl := as.numeric(channel == chl_shift_lag)]
  df[seq_lead==1, c("same_next_app", "same_next_chl") := -1]
  df[seq_lag ==1, c("same_prev_app", "same_prev_chl") := -1]
  setorderv(df, "index")
  df = df[,.(same_next_app, same_prev_app, same_next_chl, same_prev_chl)]
  if (df_out){
    return(df)
  }else{
    write.csv(df, 
              gzfile(paste0(path, fname)), 
              row.names = F, quote = F)
  }
  gc();gc();gc()
}


path = '~/tdata/data/'
#path = '/Users/dhanley2/Documents/tdata/data/'

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