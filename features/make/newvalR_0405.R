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


# Write out the <ip, device, os, channel> level -- shift2
cols_ = c("ip", "device")
keep  = c(cols_, "click_time")
trndf = fread(paste0(path, 'train.csv'))
trndf = trndf[, c(keep, "is_attributed"), with = F]
tstdf = fread(paste0(path, 'testfull.csv'))
tstdf = tstdf[, c(keep, "dataset") , with = F]

# combine the files
y = trndf$is_attributed
trndf[,is_attributed:=NULL]
test_rows = tstdf$dataset
tstdf[,dataset:=NULL]
alldf = rbind(trndf, tstdf)
rm(trndf, tstdf);gc();gc()

# Get periods
alldf[, click_time:= fasttime::fastPOSIXct(click_time)]
alldf[, day    := as.numeric(format(click_time, "%d"))]
alldf[day == 6, day    := 7]
alldf[, index  := 1:.N]
gc()

# Check for new values, or values not seen previous day
notSeenFunc = function(alldf, col){
  setkeyv(alldf, c("day", col))
  aggdf = alldf[,.N, by = c("day", col)]
  setnames(aggdf, col, "grp")
  setkeyv(aggdf, c("grp", "day"))
  aggdf[, shiftN := shift(N, 1, type = "lag") ]
  aggdf[, shiftDay := shift(day, 1, type = "lag") ]
  aggdf[, shiftGrp := shift(grp, 1, type = "lag") ]
  aggdf[, prevDay := 1]
  aggdf[day == 7, prevDay := 0]
  aggdf[N   < 30, prevDay := 0]
  aggdf[(day-shiftDay==1) & (grp==shiftGrp), prevDay := 0]
  aggdf = aggdf[,.(day, grp, prevDay)]
  setnames(aggdf, "prevDay", paste0("not_seen_", col))
  setnames(aggdf, "grp", col)
  setkeyv(aggdf, c("day", col))
  alldf = alldf[aggdf][order(index)]
  return(alldf)
}

alldf = notSeenFunc(alldf, "ip")
alldf = notSeenFunc(alldf, "device")


# Visualise the feature
alldf[, act := -1]
alldf[1:length(y), act := y]
alldf[act>-1 , .(mean(act), sum(act), .N) , by = not_seen_ip]
alldf[act>-1 , .(mean(act), sum(act), .N) , by = not_seen_device]


# Write files
featall = alldf[,.(not_seen_ip, not_seen_device)]
feattrn = featall[1:length(y)]
feattst = featall[(1+length(y)):nrow(featall)][test_rows==1]
writeme(feattrn, "not_seen_prev_trn")
writeme(feattst, "not_seen_prev_tst")
rm(feattrn, feattst)
gc(); gc(); gc()

sum(feattst$not_seen_ip)
