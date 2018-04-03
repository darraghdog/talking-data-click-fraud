#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(R.utils)
library(fasttime)
library(Hmisc)

path = '~/tdata/data/'
#path = '/Users/dhanley2/Documents/tdata/data/'
source(paste0(path, "../features/make/utils.R"))

# Load in the two files
cols_ = c("ip", "device", "os", "app")
trndf = fread(paste0(path, 'train.csv'))
trndf = trndf[,c(cols_, "click_time"), with=F]
train_rows = nrow(trndf)
gc();gc()
tstdf = fread(paste0(path, 'testfull.csv'))
sub_rows = tstdf$dataset
tstdf = tstdf[,c(cols_, "click_time"), with=F]
alldf = rbind(trndf, tstdf)
rm(trndf, tstdf)
gc();gc()

alldf[, click_sec := as.numeric(fasttime::fastPOSIXct(click_time))]
alldf[, click_sec :=  click_sec - min( click_sec)] 
alldf[, split_sec := round((0:(.N-1))/.N, 8), by = click_time]
alldf[, click_split_sec := click_sec + split_sec]
alldf[,`:=`(split_sec=NULL, click_sec=NULL, click_time=NULL)]
alldf[, index := 1:nrow(alldf)]
alldf
gc();gc();gc();gc();gc()

alldf = alldf[order(ip, device, os, app, click_split_sec)]
gc();gc();gc();gc();gc()
alldf
alldf[,seq_lead := .N:1, by = cols_ ]
alldf[,click_sec_shift_lead5 := - click_split_sec + shift(click_split_sec, 5, type = "lead")]
alldf[seq_lead %in% 1:5, click_sec_shift_lead5 := -1]
alldf[,click_sec_shift_lead1 := - click_split_sec + shift(click_split_sec, 1, type = "lead")]
alldf[seq_lead %in% 1:1, click_sec_shift_lead1 := -1]
alldf[,click_sec_shift_leaddiff5 := click_sec_shift_lead5 - click_sec_shift_lead1]
alldf[seq_lead %in% 1:5, click_sec_shift_leaddiff5 := -1]

# We won't overfit, so anything over 6 hours in the future we remove. 
alldf[click_sec_shift_lead5 > ( 3600*6 ) ,  click_sec_shift_lead5 := -1]
alldf[click_sec_shift_lead1 > ( 3600*6 ) ,  click_sec_shift_lead1 := -1]
alldf[click_sec_shift_lead5==-1 ,  click_sec_shift_leaddiff5 := -1]

alldf = alldf[,.(click_sec_shift_lead5, click_sec_shift_leaddiff5)]
gc();gc();gc();gc();gc()
feattrn = alldf[1:train_rows]
feattst = alldf[(1+train_rows):nrow(alldf)][sub_rows==1]

# Write out the files
fname = "lead5_split_sec_trn_ip_device_os_app"
writeme(feattrn, "lead5_split_sec_trn_ip_device_os_app")
writeme(feattst, "lead5_split_sec_tst_ip_device_os_app")

