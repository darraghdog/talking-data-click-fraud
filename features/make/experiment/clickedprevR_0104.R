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

getchlDf = function(fname , keepcols_){
  df = fread(paste0(path, fname))[,keepcols_,with=F]
  #samp = sample(nrow(df), 10000000)
  #df = df[samp]
  gc()
  df[, click_time := fasttime::fastPOSIXct(click_time)]
  df[ , click_day  := as.numeric(format(click_time, "%d"))]
  df[ , hour       := as.numeric(format(click_time, "%H"))]
  df[, click_time := NULL];
  gc()
  return(df)
}

tst_dataset = fread(paste0(path, "testfull.csv"))$dataset

keepcols_ = c("ip", "device", "os", "app", "click_time", "is_attributed")
trndf = getchlDf('train.csv', keepcols_)
keepcols_ = c("ip", "device", "os", "app", "click_time")
trndf[,is_train:=1]
tstdf = getchlDf('test.csv', keepcols_)
tstdf[,is_attributed:=0]
tstdf[,is_train:=0]
train_rows = nrow(trndf)
alldf = rbind(trndf, tstdf)
rm(trndf, tstdf)
#alldf = getchlDf('train.csv')
gc();gc()
alldf[,index:=1:.N]

# Rotare the hours to get them all starting at 12AM
alldf[,`:=`(hourclip= hour-16, dayclip = click_day)]
alldf[hourclip<0, `:=`(hourclip=hourclip+24, dayclip=dayclip-1)]
alldf[hourclip>=12, dayclip := dayclip+0.5]
alldf[, dayclip := dayclip * 2]  # min dayclip -- 13, maxdayclip -- 18

# Get a sample to check things
samp = sample(nrow(alldf), 10000000)
# table(alldf[samp]$hourclip, alldf[samp]$dayclip)
table(alldf[samp]$hourclip, alldf[samp]$dayclip, alldf[samp]$is_train)

# Aggregate and get the quantity for the same hour on the prev day
# Day 0 gets a -1, an dno occurrence in same hour prev day gets 0. 
aggdf = alldf[,.(.N, as.integer(sum(is_attributed)>0)), by=.(ip, app, device, os, dayclip)]
setnames(aggdf, "N", "ct")
setnames(aggdf, "V2", "is_case")
aggdf = aggdf[order(ip, device, os, app, dayclip)]
aggdf[, is_caseprev := shift( is_case, 2, type="lag")] 
aggdf[,`:=` (is_caseprev_shift1 = shift( is_case, 1, type="lag"), shift1 = 0)] 
aggdf[(dayclip-2 != shift(dayclip, 2, type="lag")) & !(dayclip-2 == shift(dayclip, 1, type="lag")),  is_caseprev := -1]
aggdf[ dayclip-2  == shift(dayclip, 1, type="lag"),  `:=`(is_caseprev = is_caseprev_shift1, shift1 = 1)]
aggdf[(app != shift(app, 2, type="lag")  & shift1 == 0),  is_caseprev := -2]
aggdf[(ip  != shift(ip, 2, type="lag")   & shift1 == 0),  is_caseprev := -2]
aggdf[(os  != shift(os, 2, type="lag")   & shift1 == 0),  is_caseprev := -2]
aggdf[(device != shift(device, 2, type="lag") & shift1 == 0),  is_caseprev := -2]
aggdf[(app != shift(app, 1, type="lag")  & shift1 == 1),  is_caseprev := -2]
aggdf[(ip  != shift(ip, 1, type="lag")   & shift1 == 1),  is_caseprev := -2]
aggdf[(os  != shift(os, 1, type="lag")   & shift1 == 1),  is_caseprev := -2]
aggdf[(device != shift(device, 1, type="lag") & shift1 == 1),  is_caseprev := -2]
# Make first two days, where we have no info, a different value
aggdf[dayclip %in% 1:13, is_caseprev := -3]
# -3 is no info form two periods before; -2 is there is not any record from 2 periods before
# -1 is we have earlier clicks but not on this day, 0 is two periods before was no is_attr; 1 is we have and is_attr
View(aggdf[ip==5147])
aggdf = aggdf[, .(ip, app, device, os, dayclip, is_caseprev)]
gc(); gc()
aggdf = aggdf[,lapply(.SD,function(x){ifelse(is.na(x),-1,x)})]
setnames(aggdf, "is_caseprev", "prev_app_hday_case")
alldf = merge(alldf, aggdf, by = c("ip", "device", "os", "dayclip", "app"), all.x=T, all.y=F)
rm(aggdf)
#table(alldf[samp]$prev_app_hday_case, alldf[samp]$is_attributed)

# Aggregate and get the quantity for the same hour on the prev day
# Day 0 gets a -1, an dno occurrence in same hour prev day gets 0. 
aggdf = alldf[,.(.N, as.integer(sum(is_attributed)>0)), by=.(ip,  device, os, dayclip)]
setnames(aggdf, "N", "ct")
setnames(aggdf, "V2", "is_case")
aggdf = aggdf[order(ip, device, os, dayclip)]
aggdf[, is_caseprev := shift( is_case, 2, type="lag")] 
aggdf[,`:=` (is_caseprev_shift1 = shift( is_case, 1, type="lag"), shift1 = 0)] 
aggdf[(dayclip-2 != shift(dayclip, 2, type="lag")) & !(dayclip-2 == shift(dayclip, 1, type="lag")),  is_caseprev := -1]
aggdf[dayclip-2 == shift(dayclip, 1, type="lag"),  `:=`(is_caseprev = is_caseprev_shift1, shift1 = 1)]
aggdf[(ip != shift(ip, 2, type="lag")   & shift1 == 0),  is_caseprev := -2]
aggdf[(os != shift(os, 2, type="lag")   & shift1 == 0),  is_caseprev := -2]
aggdf[(device != shift(device, 2, type="lag") & shift1 == 0),  is_caseprev := -2]
aggdf[(ip != shift(ip, 1, type="lag")   & shift1 == 1),  is_caseprev := -2]
aggdf[(os != shift(os, 1, type="lag")   & shift1 == 1),  is_caseprev := -2]
aggdf[(device != shift(device, 1, type="lag") & shift1 == 1),  is_caseprev := -2]
# Make first two days, where we have no info, a different value
aggdf[dayclip %in% 1:13, is_caseprev := -3]
# -3 is no info form two periods before; -2 is there is not any record from 2 periods before
# -1 is we have earlier clicks but not on this day, 0 is two periods before was no is_attr; 1 is we have and is_attr
View(aggdf[ip==5147])
aggdf = aggdf[, .(ip, device, os, dayclip, is_caseprev)]
gc(); gc()
aggdf = aggdf[,lapply(.SD,function(x){ifelse(is.na(x),-1,x)})]
setnames(aggdf, "is_caseprev", "prev_ipdevos_hday_case")
alldf = merge(alldf, aggdf, by = c("ip", "device", "os", "dayclip"), all.x=T, all.y=F)
rm(aggdf)
#table(alldf[samp]$prev_ipdevos_hday_case, alldf[samp]$is_attributed)



gc();gc();gc();gc()
alldf = alldf[order(index)]
alldf = alldf[,.(prev_ipdevos_hday_case, prev_app_hday_case)]
gc();gc();gc();gc()
feattrn = alldf[1:train_rows]
feattst = alldf[(1+train_rows):nrow(alldf)]

writeme(feattrn, "prev_hday_clicks_trn")
writeme(feattst, "prev_hday_clicks_tst")
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

