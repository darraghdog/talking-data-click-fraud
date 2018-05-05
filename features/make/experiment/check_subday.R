#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)
library(Metrics)
path = '~/tdata/data/'
path = '/Users/dhanley2/Documents/tdata/data/'

rnkr = function(vec){
  dt = data.table(vec)
  dt[, idx := 1:.N]
  dt = dt[order(vec)]
  dt[, rank := (1:.N)/.N]
  rank = dt[order(idx)]$rank
  return(rank)
}

removeDupeId = function(df){
  # identify the dupes, and the dupes with an is_attributed value
  df[, ct:= .N, by = cols ]
  df[ct>1, maxattr:= max(is_attributed), by = cols ]
  df[ct == 1, maxattr:= 0 ]
  df[ct>1, seq := .N:1 , by = cols ]
  df[ct == 1, seq := 0 ]
  # Extract the dupes with a case
  removeDupe = rep(0, nrow(df))
  removeDupe[(df$maxattr==1) & (df$is_attributed==0)] = 1
  # Extract the final of the dupes wihout a case
  removeDupe[(df$maxattr==0) & (df$seq>1)] = 1
  return(removeDupe)
}

cols = c("ip", "app", "device", "os", "channel", "click_time")
trndf = fread(paste0(path, 'trainval.csv'))
tstdf = fread(paste0(path, 'testval.csv'))
trnfdf = fread(paste0(path, 'train.csv'))
tstfdf = fread(paste0(path, 'test.csv'))[order(click_id)]
tstdf[,click_time := fasttime::fastPOSIXct(click_time)]
tstdf[,hour   := as.numeric(format(click_time, "%H"))]


sublg = fread(paste0(path, "../sub/sub_lgb2404val_difftime_thresh35.csv"), skip = 1)
subnn = fread(paste0(path, "../sub/sub_nnet2104val.csv"))
sublg1 = fread(paste0(path, "../sub/sub_lgb1704val.csv"), skip = 1)
subst = fread(paste0(path, "../sub/blend_0503_1.postproc1.csv"))[order(click_id)]


auc(tstdf$is_attributed, sublg$V1)
# [1] 0.9832104
auc(tstdf$is_attributed, subnn$is_attributed)
# [1] 0.982526
auc(tstdf$is_attributed, sublg1$V1)
# [1] 0.9836714

table(tstdf$hour)
idx = tstdf$hour == 4 
auc(tstdf[idx]$is_attributed, sublg[idx]$V1)
# [1] 0.9808486
auc(tstdf[idx]$is_attributed, subnn[idx]$is_attributed)
# [1] 0.9804621
auc(tstdf[idx]$is_attributed, sublg1[idx]$V1)
# [1] 0.9814477





tstdf[, pred := rnkr(sublg1$V1)]
# [1] 0.9836714
auc(tstdf$is_attributed, tstdf$pred1)
tstdf[, pred1 := pred]
tstdf[!ip %in% trndf$ip, pred1 := pred*1.05]

tstdf[, ipct := .N, by = ip]
tstdf[, devct := .N, by = device]
devs = unique(tstdf[devct>30000]$device)

View(tstdf[,.(mean(is_attributed), .N), by = app])

idx = tstdf$device == devs[3]
table(idx)
tstdf[, pred := rnkr(sublg1$V1)]
tstfdf[, pred := rnkr(subst$is_attributed)]

auc(tstdf[idx]$is_attributed, tstdf[idx]$pred)
# [1] 0.9836714
tstdf[, pred1 := pred ]
tstdf[device %in% c(4, 56), pred1 := 0.8*pred]
auc(tstdf$is_attributed, tstdf$pred1)


3543, 3866
hist(tstdf[device == 3866]$pred)
hist(tstfdf[device == 3866]$pred)
mean(trnfdf[app == 56]$is_attributed)

4, 56

tstdf[!os %in% trndf$device]
tstfdf[!device %in% trnfdf$device]

View(tstdf[(is_attributed == 1)&(0.5>pred) ])



