#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)


path = '~/tdata/data/'
path = '/Users/dhanley2/Documents/tdata/data/'

source(paste0(path, '../features/make/utils.R'))

# Write out the <ip, device, os, channel> level -- shift2
keep  = c("ip", "device", "os", "app", "channel", "click_time") 
basecols_ = c("ip", "device", "os")
trndf = fread(paste0(path, 'train.csv'))[,keep,with=F]
fname = "lead_split_sec_trn_ip_device_os_appchl"
feattrn1  = getSplitLead2(trndf, c(basecols_, "app"), "app", path, TRUE)
feattrn2  = getSplitLead2(trndf, c(basecols_, "channel"), "chl", path, TRUE)
feattrn3  = getSplitLead2(trndf, c(basecols_, "channel", "app"), "appchl", path, TRUE)
feattrn   = cbind(feattrn1, feattrn2, feattrn3)
writeme(feattrn, fname)
rm(feattrn1, feattrn2, feattrn3, feattrn, trndf)
gc()
# table(cut2(feattrn$click_sec_lead_split_sec[1:10000000], g = 100), trndf$is_attributed[1:10000000])

tstdf = fread(paste0(path, 'testfull.csv'))
setidx = tstdf$dataset
fname = "lead_split_sec_tst_ip_device_os_appchl"
feattst1  = getSplitLead2(tstdf, c(basecols_, "app"), "app", path, TRUE)
feattst2  = getSplitLead2(tstdf, c(basecols_, "channel"), "chl", path, TRUE)
feattst3  = getSplitLead2(tstdf, c(basecols_, "channel", "app"), "appchl", path, TRUE)
feattst   = cbind(feattst1, feattst2, feattst3)
writeme(feattst, fname)
rm(feattst1, feattst2, feattst3, feattst, tstdf)
gc()


