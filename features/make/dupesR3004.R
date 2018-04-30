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

trndf = fread(paste0(path, 'train.csv'))
tstdf = fread(paste0(path, 'test.csv'))
trndf = trndf[, .(ip, device, os, app, channel, click_time)]
tstdf = tstdf[, .(ip, device, os, app, channel, click_time)]
train_rows = nrow(trndf)
alldf = rbind(trndf, tstdf)
rm(trndf, tstdf)
gc(); gc(); gc()

alldf[, click_sec := as.numeric(fasttime::fastPOSIXct(click_time))]
alldf[, index := 1:.N]
gc(); gc(); gc()
setkeyv(alldf, c("ip", "device", "os", "app", "channel", "click_sec"))
aggdf = unique(alldf[,.(ip, device, os, app, channel, click_sec)])
aggdf[, group1 := 1:.N]
alldf = merge(alldf, aggdf, by = c("ip", "device", "os", "app", "channel", "click_sec"))
alldf[, click_sec4 := as.integer(click_sec/4)]
aggdf = unique(alldf[,.(ip, device, os, app, channel, click_sec4)])
aggdf[, group2 := 1:.N]
alldf = merge(alldf, aggdf, by = c("ip", "device", "os", "app", "channel", "click_sec4"))
rm(aggdf)
gc(); gc(); gc()
setkeyv(alldf, c("index"))
alldf[, dupeseq1:= 1:.N , by = group1]
alldf[, dupeseq2:= 1:.N , by = group2]
alldf[, group1ct:= .N , by = group1]
alldf[, group2ct:= .N , by = group2]
alldf[group1ct==1, dupeseq1 := 99 ]
alldf[group2ct==1, dupeseq2 := 99 ]

feattrn = alldf[1:train_rows , .(dupeseq1, dupeseq2)]
feattst = alldf[(1+train_rows:nrow(alldf)) , .(dupeseq1, dupeseq2)]
feattst[is.na(feattst)] <- 99


fname = "dupeseqtrn"
write.csv(feattrn,  gzfile(paste0(path, fname)), row.names = F, quote = F)
fname = "dupeseqtst"
write.csv(feattst,  gzfile(paste0(path, fname)), row.names = F, quote = F)

