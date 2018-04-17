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
cols_ = c("ip", "device", "os", "app")
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

# Set up time for click counts
alldf$click_sec = as.numeric(fasttime::fastPOSIXct(alldf$click_time))
alldf[, click_sec := click_sec - min(alldf$click_sec)]
alldf[, click_hour := ((round(click_sec/3600) - min(round(alldf$click_sec/3600)))-5)%%24] # Make the hour -5 for TZ
alldf[, click_day  := 1 +  round((click_sec-(3600*5))/(3600*24)) - min(round(alldf$click_sec/(3600*24)))] 

# Visualise set up
samp = sample(1:nrow(alldf), 1000000)
table(alldf[samp]$click_hour, alldf[samp]$click_day)

# Create cumulative count ratio, for big hitters; also only do it for the hours we have in test
idx = (alldf$click_day>1) & (alldf$click_hour %in% 13:20)
alldf[idx, ct := .N, by = .(ip, device, os, app, click_day)]
alldf[, `:=`(cumsum50 = 0, cumsum10=0)]
alldf[(ct > 50) & idx, cumsum50 := 2+(((.N-1):0)/(.N-1)) , by = .(ip, device, os, app)]
alldf[(ct > 10) & idx, cumsum10 := 2+(((.N-1):0)/(.N-1)) , by = .(ip, device, os, app)]
featall = alldf[,.(cumsum10, cumsum50)]

# Check a table of the cumsum feature
table(cut2(alldf[1:length(y)][cumsum>1]$cumsum, g = 100), y[alldf[1:length(y)]$cumsum>1] )



# Write files
feattrn = featall[1:length(y)]
feattst = featall[(1+length(y)):nrow(featall)][test_rows==1]
idx = feattrn$cumsum10!=0
table(cut2(feattrn[idx]$cumsum10, g=20), y[idx])
idx = feattrn$cumsum50!=0
table(cut2(feattrn[idx]$cumsum50, g=10), y[idx])

writeme(feattrn, "cumsumday_trn")
writeme(feattst, "cumsumday_tst")

rm(feattrn, feattst)
gc(); gc(); gc()

