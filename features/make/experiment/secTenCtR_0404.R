#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)

getPeriodCt = function(df, cols_, count_period, intervals){
  df = df[,c(cols_, "split_sec"), with = F]
  df[, index := 1:nrow(df)]
  setorderv(df, c(cols_, "split_sec"))
  df[,next_ten := 0]
  df[,seq_lead := .N:1, by = cols_ ]
  for (shift_n in intervals){
    print(shift_n)
    df[,click_sec_shift_lead := shift(split_sec, shift_n, type = "lead")]
    df[(seq_lead>shift_n) & ((click_sec_shift_lead - split_sec) < (count_period*10000)), next_ten := shift_n]
    gc() 
  }
  setorderv(df, "index")
  setnames(df, "next_ten", new_name)
  df = df[,new_name,with=F]
  return(df)
}


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
alldf[, split_sec := round((0:(.N-1))/.N, 4), by = click_time]
alldf[, split_sec := click_sec*10000+split_sec*10000]
alldf[, `:=`(click_time=NULL, click_sec=NULL)]
gc();gc();gc();gc();gc();gc();gc()

# Create the feature
count_period = 3600 # how many seconds in the future we count
new_name = "count_in_next_hr"
intervals = c(1:20, 2*(11:24), 5*(10:20),10*(11:20), 20*(11:20), 40*(11:20))
featall1  = getPeriodCt(alldf, cols_, count_period, intervals) 
count_period = 600 # how many seconds in the future we count
intervals = c(1:15, 2*(8:15), 5*(7:16),10*(9:20))
new_name = "count_in_next_ten_mins"
featall2  = getPeriodCt(alldf, cols_, count_period, intervals) 

# Merge the features
featall = cbind(featall1, featall2)
rm(featall1, featall2); gc();gc()

# Write files
feattrn = featall[1:length(y)]
feattst = featall[(1+length(y)):nrow(featall)][test_rows==1]
writeme(feattrn, "lead_count_next_ipdevosapp_trn")
writeme(feattst, "lead_count_next_ipdevosapp_tst")
rm(feattrn, feattst, alldf)
gc(); gc(); gc()

# check the feature
set.seed(10)
samp = sample(nrow(featall[1:length(y)]), 2000000)
table(cut2(featall[samp]$count_in_next_hr, g = 20), y[samp])
table(cut2(featall[samp]$count_in_next_ten_mins, g = 20), y[samp])



