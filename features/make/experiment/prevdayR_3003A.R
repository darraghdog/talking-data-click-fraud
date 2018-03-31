#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)

path = '~/tdata/data/'
path = '/Users/dhanley2/Documents/tdata/data/'
source(paste0(path, "../features/make/utils.R"))

getchlDf = function(fname){
  keepcols_ = c("ip", "click_time")
  cols_ = c("ip", "device", "os", "app")
  df = fread(paste0(path, fname))[,keepcols_,with=F]
  gc()
  df[, click_time := fasttime::fastPOSIXct(click_time)]
  df[, click_hour := round((as.numeric(click_time))/3600)%%24]
  df[, click_day := round(as.numeric(click_time)/(3600*24))]
  df[, click_time := NULL];
  gc()
  return(df)
}

tst_dataset = fread(paste0(path, "testfull.csv"))$dataset
trndf = getchlDf('train.csv')
tstdf = getchlDf('testfull.csv')
train_rows = nrow(trndf)
alldf = rbind(trndf, tstdf)
rm(trndf, tstdf)
gc();gc()
alldf[,index:=1:.N]

# Aggregate and get the quantity for the same hour on the prev day
# Day 0 gets a -1, an dno occurrence in same hour prev day gets 0. 
aggdf = alldf[,.N, by=.(ip, click_hour, click_day)]
aggdf = aggdf[order(ip, click_hour, click_day)]
aggdf[, prevday_qty:=shift(N, 1, type="lag")] 
aggdf[shift(click_day, 1, type="lag")!=click_day-1,   prevday_qty := 0]
aggdf[ip != shift(ip, 1, type="lag"),                 prevday_qty := 0]
aggdf[shift(click_hour, 1, type="lag") != click_hour, prevday_qty := 0]
aggdf = aggdf[order(ip, click_day, click_hour)]
aggdf[, prevhour_qty:=shift(N, 1, type="lag")] 
aggdf[shift(click_day, 1, type="lag")!=click_day,      prevhour_qty := 0]
aggdf[ip != shift(ip, 1, type="lag"),                  prevhour_qty := 0]
aggdf[shift(click_hour, 1, type="lag") != click_hour-1,prevhour_qty := 0]
setnames(aggdf, "N", "qty")
aggdf = aggdf[order(click_day, click_hour)]
aggdf[click_day == min(aggdf$click_day), prevday_qty := -1] # Make the first day be all -1
aggdf[click_day == min(aggdf$click_day) & click_hour==min(click_hour), prevhour_qty := -1] # Make the first day be all -1
#View(aggdf[ip==5147])
alldf = merge(alldf, aggdf, by = c("ip", "click_hour", "click_day"), all.x=T, all.y=F)
rm(aggdf)

gc();gc();gc();gc()
alldf = alldf[order(index)]
alldf = alldf[,.(qty, prevday_qty, prevhour_qty)]
gc();gc();gc();gc()
feattrn = alldf[1:train_rows]
feattst = alldf[(1+train_rows):nrow(alldf)][tst_dataset==1]
rm(alldf);gc();gc()

writeme(feattrn, "prevdayipchlqtytrn")
writeme(feattst, "prevdayipchlqtytst")
head(feattrn, 30)
# 
# hist(feattrn$qty)
# 
# y = fread(paste0(path, "train.csv"))$is_attributed
# 
# table(cut2(feattrn$qty[1:10000000], g= 20), y[1:10000000])
# table(cut2(feattrn$prevday_qty[110000000:120000000], g= 20), y[110000000:120000000])
# 
# sum(is.infinite(feattrn$prevday_qty[1:10000000]))

