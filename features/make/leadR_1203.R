#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)

path = '~/tdata/data/'
trndf = fread(paste0(path, 'trainvalsmall.csv'))
trndf = trndf[,.(ip, device, os, click_time)]
gc();gc();gc()
trndf$click_sec = as.numeric(fasttime::fastPOSIXct(trndf$click_time))
trndf$click_time = NULL
gc();gc();gc()
train_rows = nrow(trndf)

tstdf = fread(paste0(path, 'testvalsmall.csv'))
tstdf = tstdf[,.(ip, device, os, click_time)]
gc();gc();gc()
tstdf$click_sec = as.numeric(fasttime::fastPOSIXct(tstdf$click_time))
tstdf$click_time = NULL
gc();gc();gc()

alldf = rbind(trndf, tstdf)
rm(trndf, tstdf)
gc();gc();gc()
alldf[, index := 1:nrow(alldf)]
alldf = alldf[order(ip, device, os, click_sec)]
alldf[,click_sec_shift := shift(click_sec, 1, type = "lead")]
alldf[,seq := .N:1, by = .(ip, device, os) ]
alldf[,click_sec_lead := click_sec_shift - click_sec]
alldf[seq==1, click_sec_lead := -1]
alldf = alldf[order(index)]
alldf

leadtrn = alldf[,.(click_sec_lead)][1:train_rows]
leadtst = alldf[,.(click_sec_lead)][(train_rows+1): nrow(alldf)]

trndf = fread(paste0(path, 'trainvalsmall.csv'))
trndf[,lead_time:= leadtrn$click_sec_lead]


table(cut2(trndf$lead_time, g = 10), trndf$is_attributed)

write.csv(leadtrn, paste0(path, '../features/leadtrnsmall.csv'), row.names = FALSE, quote = F)
write.csv(leadtst, paste0(path, '../features/leadtstsmall.csv'), row.names = FALSE, quote = F)
