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
trndf[, click_sec := as.numeric(fasttime::fastPOSIXct(click_time))]
trndf[, index := 1:.N]
trndf = trndf[, .(ip, device, os, app, channel, click_sec, is_attributed, index)]
gc(); gc(); gc()
aggdf = unique(trndf[,.(ip, device, os, app, channel, click_sec)])
aggdf[, group1 := 1:.N]
trndf = merge(trndf, aggdf, by = c("ip", "device", "os", "app", "channel", "click_sec"))
trndf[, click_sec4 := as.integer(click_sec/4)]
aggdf = unique(trndf[,.(ip, device, os, app, channel, click_sec4)])
aggdf[, group2 := 1:.N]
trndf = merge(trndf, aggdf, by = c("ip", "device", "os", "app", "channel", "click_sec4"))
trndf = trndf[order(index)]
trndf[, dupeseq1:= 1:.N , by = group1]
trndf[, dupeseq2:= 1:.N , by = group2]
trndf[, group1ct:= .N , by = group1]
trndf[, group2ct:= .N , by = group2]
trndf[group1ct==1, dupeseq1 := 0 ]
trndf[group2ct==1, dupeseq2 := 0 ]
trndf[, .(mean(is_attributed), sum(is_attributed), .N), by = dupeseq1]
trndf[, .(mean(is_attributed), sum(is_attributed), .N), by = dupeseq2]


