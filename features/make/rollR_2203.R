#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)


path = '~/tdata/data/'
path = '/Users/dhanley2/Documents/tdata/data/'

# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F, na = '')
}



trndf = fread(paste0(path, 'train.csv'))
tstdf = fread(paste0(path, 'testfull.csv'))
trndf$is_train = 1
tstdf$is_train = 0
trndf$dataset  = 0
trndf$attributed_time = tstdf$click_id = NULL
tstdf$is_attributed =   -1
alldf = rbind(trndf,tstdf)
rm(trndf,tstdf);gc();gc()

alldf$click_time = fasttime::fastPOSIXct(alldf$click_time)
alldf$click_sec = as.numeric(alldf$click_time)
alldf[, click_day  := as.numeric(format(click_time, "%d"))]
cols_ = c("ip", "device", "os")
alldf = alldf[,c(cols_, "click_sec", "is_attributed", "dataset", "is_train", "click_day"), with = F]
alldf[, index := 1:nrow(alldf)]
gc();gc();gc();gc();gc();gc()
setorderv(alldf, c(cols_, "click_sec"))
alldf[,click_sec_shift_lead := shift(click_sec, 1, type = "lead")]
alldf[,click_sec_shift_lag  := shift(click_sec, 1, type = "lag")]
alldf[,seq_lead := .N:1, by = c(cols_, "click_day") ]
alldf[,seq_lag  := 1:.N, by = c(cols_, "click_day") ]
alldf[,click_sec_lead := click_sec_shift_lead - click_sec]
alldf[,click_sec_lag := click_sec - click_sec_shift_lag]
alldf[seq_lead==1, click_sec_lead := NA]
alldf[seq_lag==1,  click_sec_lag :=  NA]
alldf[, roll_mean_five := as.integer(round(roll_mean(click_sec_lead, 5, na.rm = TRUE, fill = -1)*2)), by = cols_]
alldf[, roll_max_five := as.integer(round(roll_max(click_sec_lead, 5, na.rm = TRUE, fill = -1))), by = cols_]
alldf[, roll_min_five := as.integer(round(roll_min(click_sec_lead, 5, na.rm = TRUE, fill = -1))), by = cols_]
alldf[, roll_var_five := roll_var(click_sec_lead, 5, na.rm = TRUE, fill = -1), by = cols_]
setorderv(alldf, "index")

table(cut2(alldf[1:10000000]$roll_mean_five, g = 20), alldf[1:10000000]$is_attributed)
table(cut2(alldf[1:10000000]$roll_max_five, g = 20), alldf[1:10000000]$is_attributed)
table(cut2(alldf[1:10000000]$roll_min_five, g = 20), alldf[1:10000000]$is_attributed)
table(cut2(alldf[1:10000000]$roll_var_five, g = 20), alldf[1:10000000]$is_attributed)

writeme(alldf[is_train==1,.(roll_mean_five, roll_min_five, roll_max_five, roll_var_five)], 'roll_five_trn', na = -1)
writeme(alldf[dataset==1 ,.(roll_mean_five, roll_min_five, roll_max_five, roll_var_five)], 'roll_five_tst', na = -1)



trndf = fread(paste0(path, 'trainvalsmall.csv'))
tstdf = fread(paste0(path, 'testvalsmall.csv'))
trndf$is_train = 1
tstdf$is_train = 0
trndf$attributed_time = tstdf$click_id = NULL
tstdf$is_attributed =   -1
alldf = rbind(trndf,tstdf)
rm(trndf,tstdf);gc();gc()

alldf$click_time = fasttime::fastPOSIXct(alldf$click_time)
alldf$click_sec = as.numeric(alldf$click_time)
alldf[, click_day  := as.numeric(format(click_time, "%d"))]
cols_ = c("ip", "device", "os")
alldf = alldf[,c(cols_, "click_sec", "is_attributed", "is_train", "click_day"), with = F]
alldf[, index := 1:nrow(alldf)]
gc();gc();gc();gc();gc();gc()
setorderv(alldf, c(cols_, "click_sec"))
alldf[,click_sec_shift_lead := shift(click_sec, 1, type = "lead")]
alldf[,click_sec_shift_lag  := shift(click_sec, 1, type = "lag")]
alldf[,seq_lead := .N:1, by = c(cols_, "click_day") ]
alldf[,seq_lag  := 1:.N, by = c(cols_, "click_day") ]
alldf[,click_sec_lead := click_sec_shift_lead - click_sec]
alldf[,click_sec_lag := click_sec - click_sec_shift_lag]
alldf[seq_lead==1, click_sec_lead := NA]
alldf[seq_lag==1,  click_sec_lag :=  NA]
alldf[, roll_mean_five := as.integer(round(roll_mean(click_sec_lead, 5, na.rm = TRUE, fill = -1)*2)), by = cols_]
alldf[, roll_max_five := as.integer(round(roll_max(click_sec_lead, 5, na.rm = TRUE, fill = -1))), by = cols_]
alldf[, roll_min_five := as.integer(round(roll_min(click_sec_lead, 5, na.rm = TRUE, fill = -1))), by = cols_]
alldf[, roll_var_five := roll_var(click_sec_lead, 5, na.rm = TRUE, fill = -1), by = cols_]
setorderv(alldf, "index")
alldf[, roll_var_five := round(roll_var_five, 5)]


table(cut2(alldf[1:10000000]$roll_mean_five, g = 20), alldf[1:10000000]$is_attributed)
table(cut2(alldf[1:10000000]$roll_max_five, g = 20), alldf[1:10000000]$is_attributed)
table(cut2(alldf[1:10000000]$roll_min_five, g = 20), alldf[1:10000000]$is_attributed)
table(cut2(alldf[1:10000000]$roll_var_five, g = 20), alldf[1:10000000]$is_attributed)
writeme(alldf[is_train==1][ ,.(roll_mean_five, roll_min_five, roll_max_five, roll_var_five)], 'roll_five_trnvalsmall')
writeme(alldf[is_train==0][ ,.(roll_mean_five, roll_min_five, roll_max_five, roll_var_five)], 'roll_five_tstvalsmall')

