#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)

path = '~/tdata/data/'
#path = '/Users/dhanley2/Documents/tdata/data/'
source(paste0(path, "../features/make/utils.R"))

getchlDf = function(fname){
  keepcols_ = c("ip", "channel", "click_time")
  cols_ = c("ip", "device", "os", "app")
  df = fread(paste0(path, fname))[,keepcols_,with=F]
  gc()
  df[, click_time := fasttime::fastPOSIXct(click_time)]
  df[, click_hour := round((as.numeric(click_time))/3600)%%24]
  df[, click_day := round(as.numeric(click_time)/(3600*24)) - min(round(as.numeric(click_time)/(3600*24)))]
  df[, click_time := NULL];
  gc()
  return(df)
}

trndf = getchlDf('train.csv')
tstdf = getchlDf('testfull.csv')

# Write out the <ip, device, os, channel> level -- shift2
keepcols_ = c("ip", "channel", "click_time", "is_attributed")
cols_ = c("ip", "device", "os", "app")
trndf = fread(paste0(path, 'train.csv'))[,keepcols_,with=F]
gc()
trndf[, click_time := fasttime::fastPOSIXct(click_time)]
trndf[, click_hour := round((as.numeric(click_time))/3600)%%24]
trndf[, click_day := round(as.numeric(click_time)/(3600*24)) - min(round(as.numeric(click_time)/(3600*24)))]
trndf[, click_time := NULL];
gc()
trndf[,qty:=.N, by=.(ip, click_day, click_hour, channel)]
trndf[,index:= 1:.N]

aggdf = trndf[,.N, by=.(ip, channel, click_hour, click_day)]
aggdf = aggdf[order(ip, channel, click_hour, click_day)]
aggdf[, `:=`(prevday_qty=shift(N, 1, type="lag"), 
          click_day_prev=shift(click_day, 1, type="lag"),
          click_hour_prev=shift(click_hour, 1, type="lag"),
          channel_prev=shift(channel, 1, type="lag"),
          ip_prev=shift(ip, 1, type="lag")) ]
aggdf[(click_day_prev!=click_day-1), prevday_qty := 0]
aggdf[(ip != ip_prev), prevday_qty := 0]
aggdf[(channel != channel_prev), prevday_qty := 0]
aggdf[(click_hour_prev != click_hour_prev), prevday_qty := 0]
aggdf = aggdf[, .(ip, channel, click_hour, click_day, prevday_qty)]
#View(aggdf[ip==5147 & channel == 280])
trndf = merge(trndf, aggdf, by = c("ip", "channel", "click_hour", "click_day"), all.x=T, all.y=F)
rm(aggdf)
gc();gc();gc();gc();gc();gc();gc();gc()
trndf = trndf[order(index)]
trndf




