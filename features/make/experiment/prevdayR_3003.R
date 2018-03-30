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
train_rows = nrow(trndf)
alldf = rbind(trndf, tstdf)
rm(trndf, tstdf)
gc();gc()
alldf[,index:=1:.N]

# Aggregate and get the quantity for the same hour on the prev day
# Day 0 gets a -1, an dno occurrence in same hour prev day gets 0. 
aggdf = alldf[,.N, by=.(ip, channel, click_hour, click_day)]
aggdf = aggdf[order(ip, channel, click_hour, click_day)]
aggdf[, prevday_qty:=shift(N, 1, type="lag")] 
aggdf[shift(click_day, 1, type="lag")!=click_day-1,   prevday_qty := 0]
aggdf[ip != shift(ip, 1, type="lag"),                 prevday_qty := 0]
aggdf[channel != shift(channel, 1, type="lag"),       prevday_qty := 0]
aggdf[shift(click_hour, 1, type="lag") != click_hour, prevday_qty := 0]
setnames(aggdf, "N", "qty")
#View(aggdf[ip==5147 & channel == 280])
trndf = merge(alldf, aggdf, by = c("ip", "channel", "click_hour", "click_day"), all.x=T, all.y=F)
rm(aggdf)
gc();gc();gc();gc();gc();gc();gc();gc()
trndf = trndf[order(index)]
trndf




