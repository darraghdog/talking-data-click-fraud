#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)

getRollMinLead = function(df, cols_, fname, path, df_out = FALSE, shift_n = 1){
  df$click_time = fasttime::fastPOSIXct(df$click_time)
  df$click_sec = as.numeric(df$click_time)
  df$click_day  = wday(df$click_time)
  df = df[,c(cols_, "click_sec", "click_time", "click_day"), with = F]
  df[, index := 1:nrow(df)]
  setorderv(df, c(cols_, "click_sec"))
  df[,click_sec_shift_lead := shift(click_sec, shift_n, type = "lead")]
  df[,seq_lead := .N:1, by = cols_ ]
  df[,click_sec_lead := click_sec_shift_lead - click_sec]
  df[seq_lead %in% 1:shift_n, click_sec_lead := -1]
  df[,lead_app_cum_min := click_sec_lead]
  df[lead_app_cum_min == -1 ,lead_app_cum_min:= 99999]
  df[,lead_app_cum_min := cummin(lead_app_cum_min) , by = c(cols_, "click_day")]
  df[,lead_app_cum_click_day := 1:length(lead_app_cum_min) , by = c(cols_, "click_day")]
  df = df[,.(lead_app_cum_min, lead_app_cum_click_day)]
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
path = '/Users/dhanley2/Documents/tdata/data/'

#############################################################################
###################### Rolling min on the day ###############################
#############################################################################

# Write out the <ip, device, os, channel> level -- shift2
cols_ = c("ip", "device", "os", "app")

trndf = fread(paste0(path, 'train.csv'))
fname = "cum_min_trn_ip_device_os_app.gz"
getRollMinLead(trndf, cols_, fname, path)

tstdf = fread(paste0(path, 'testfull.csv'))
setidx = tstdf$dataset
fname = "cum_min_tst_ip_device_os_app.gz"
feats = getRollMinLead(tstdf, cols_, fname, path, TRUE)
write.csv(feats[setidx==1], 
          gzfile(paste0(path, fname)), 
          row.names = F, quote = F)
rm(tstdf, feats)

