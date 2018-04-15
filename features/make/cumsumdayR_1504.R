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
alldf[, click_hour := round(click_sec/3600) - min(round(alldf$click_sec/3600))]
alldf[, click_hour := click_hour%%24]
alldf[, click_easthour := (click_hour-5)%%24]
alldf[, click_day  := 1 +  round((click_sec-(3600*5))/(3600*24)) - min(round(alldf$click_sec/(3600*24)))] 

# Visualise set up
samp = sample(1:nrow(alldf), 10000000)
table(alldf[samp]$click_easthour, alldf[samp]$click_day)
hist(alldf[samp]$click_easthour)

# Create cumulative count ratio, for big hitters; also only do it for the hours we have in test
alldf[(click_day>1) & (click_easthour %in% 13:20), ct := .N, by = .(ip, device, os, app, click_day)]
alldf[, cumsum:= 0]
alldf[(click_day>1) & (click_easthour %in% 13:20), cumsum:=1]
alldf[(ct > 30) & (cumsum==1), cumsum := 2+(((.N-1):0)/(.N-1)) , by = .(ip, device, os, app)]
alldf[cumsum == 1, cumsum:= 0]
featall = alldf[,.(cumsum)]

# Check a table of the cumsum feature
table(cut2(alldf[1:length(y)][cumsum>1]$cumsum, g = 100), y[alldf[1:length(y)]$cumsum>1] )



# Write files
feattrn = featall[1:length(y)]
feattst = featall[(1+length(y)):nrow(featall)][test_rows==1]
writeme(feattrn, "cumsumday_trn")
writeme(feattst, "cumsumday_tst")

table(cut2(feattrn$cumsum, g = 20), y)

rm(feattrn, feattst)
gc(); gc(); gc()

# check the feature
set.seed(10)
samp = sample(nrow(featall[1:length(y)]), 10000000)
table(cut2(featall[samp]$count_in_next_ipdevos_ten_sec, g = 50), y[samp])
table(cut2(featall[samp]$count_in_next_ipdevos_ten_min, g = 50), y[samp])


table(featall[samp]$count_in_next_ipdevos_ten_sec>100, y[samp])


