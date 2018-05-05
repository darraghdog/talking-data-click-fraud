#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)

source(paste0(path, "../features/make/utils.R"))
path = '~/tdata/data/'
path = '/Users/dhanley2/Documents/tdata/data/'

# Write out the <ip, device, os, channel> level -- shift2
cols_ = c("ip", "app")
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

# Check for new values, or values not seen previous day
col = "app"
setkeyv(alldf, c("day", col))
aggdf = alldf[,.N, by = c("day", col)]
setnames(aggdf, col, "grp")
setkeyv(aggdf, c("grp", "day"))
aggdf[, shiftN := shift(N, 1, type = "lag") ]
aggdf[, shiftDay := shift(day, 1, type = "lag") ]
aggdf[, shiftGrp := shift(grp, 1, type = "lag") ]
aggdf[, prevDay := 1]
aggdf[day == 7, prevDay := 0]
aggdf[N   < 40, prevDay := 0]
aggdf[(day-shiftDay==1) & (grp==shiftGrp), prevDay := 0]
aggdf = aggdf[,.(day, grp, prevDay)]
setnames(aggdf, "prevDay", paste0("not_seen_", col))
setkeyv(aggdf, c("day", col))

table(aggdf$prevDay)

aggdf


# check the feature
set.seed(10)
samp = sample(nrow(featall[1:length(y)]), 2000000)
table(cut2(featall[samp]$click_sec_lead_split_sec, g = 20), y[samp])

# Write files
feattrn = featall[1:length(y)]
feattst = featall[(1+length(y)):nrow(featall)][test_rows==1]
writeme(feattrn, "lead_split_sec_trn_ip")
writeme(feattst, "lead_split_sec_tst_ip")
rm(feattrn, feattst)
gc(); gc(); gc()