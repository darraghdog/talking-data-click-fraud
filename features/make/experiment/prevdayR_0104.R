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
  df[, click_hourseq := round((as.numeric(click_time))/3600)]
  df[, click_qday := round((as.numeric(click_time))/3600)%%4]
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
aggdf = alldf[,.N, by=.(ip, click_hour, click_hourseq, click_day)]
aggdf = aggdf[order(ip, click_hour, click_day)]
aggdf[, prevdayhr_qty:=shift(N, 1, type="lag")] 
aggdf[shift(click_day, 1, type="lag")!=click_day-1,     prevdayhr_qty := 0]
aggdf[ip != shift(ip, 1, type="lag"),                   prevdayhr_qty := 0]
aggdf[shift(click_hour, 1, type="lag") != click_hour,   prevdayhr_qty := 0]

aggdf = aggdf[order(ip, click_hourseq)]
aggdf[, prevhr_qty:=shift(N, 1, type="lag")] 
aggdf[ip != shift(ip, 1, type="lag"),                         prevhr_qty := 0]
aggdf[shift(click_hourseq, 1, type="lag") != click_hourseq-1, prevhr_qty := 0]
aggdf[, nexthr_qty:=shift(N, 1, type="lead")] 
aggdf[ip != shift(ip, 1, type="lead"),                        nexthr_qty := 0]
aggdf[shift(click_hourseq, 1, type="lead") != click_hourseq+1,nexthr_qty := 0]
setnames(aggdf, "N", "qty")
aggdf = aggdf[order(click_day, click_hour)]
View(aggdf[ip==5147])
aggdf = aggdf[,lapply(.SD,function(x){ifelse(is.na(x),-1,x)})]
alldf = merge(alldf, aggdf, by = c("ip", "click_hour", "click_hourseq", "click_day"), all.x=T, all.y=F)
rm(aggdf)

alldf[, click_qday := round(click_hourseq/6)]
aggdf = alldf[,.N, by=.(ip, click_qday)]
aggdf = aggdf[order(ip, click_qday)]
aggdf[, prevqday_qty:=shift(N, 1, type="lag")] 
aggdf[ip != shift(ip, 1, type="lag"),                         prevqday_qty := 0]
aggdf[shift(click_qday, 1, type="lag") != click_qday-1,       prevqday_qty := 0]
aggdf[, nextqday_qty:=shift(N, 1, type="lead")] 
aggdf[ip != shift(ip, 1, type="lead"),                        nextqday_qty := 0]
aggdf[shift(click_qday, 1, type="lead") != click_qday+1,      nextqday_qty := 0]
aggdf = aggdf[,lapply(.SD,function(x){ifelse(is.na(x),-1,x)})]
aggdf[,N := NULL]
alldf = merge(alldf, aggdf, by = c("ip", "click_qday"), all.x=T, all.y=F)
#View(aggdf[ip==5147])
rm(aggdf)

gc();gc();gc();gc()
alldf = alldf[order(index)]
alldf = alldf[,.(prevdayhr_qty, prevhr_qty, nexthr_qty, prevqday_qty, nextqday_qty)]
gc();gc();gc();gc()
feattrn = alldf[1:train_rows]
feattst = alldf[(1+train_rows):nrow(alldf)][tst_dataset==1]
rm(alldf);gc();gc()

writeme(feattrn, "prevqdayipchlqtytrn")
writeme(feattst, "prevqdayipchlqtytst")
head(feattrn, 30)


# # prevdayipchlqty
# gunzip  prevqdayipchlqtytrn.gz
# sed -n 1,1p prevqdayipchlqtytrn > prevqdayipchlqtytrnval
# sed -n 60000000,122080000p prevqdayipchlqtytrn >> prevqdayipchlqtytrnval
# sed -n 1,1p prevqdayipchlqtytrn > prevqdayipchlqtytstval
# sed -n 144710000,152400000p prevqdayipchlqtytrn >> prevqdayipchlqtytstval
# sed -n 162000000,168300000p  prevqdayipchlqtytrn >> prevqdayipchlqtytstval
# sed -n 175000000,181880000p  prevqdayipchlqtytrn >> prevqdayipchlqtytstval
# gzip prevqdayipchlqtytrnval
# gzip prevqdayipchlqtytstval
# gzip prevqdayipchlqtytrn

