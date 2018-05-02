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



removeDupeId = function(df){
  # identify the dupes, and the dupes with an is_attributed value
  df[, ct:= .N, by = cols ]
  df[ct>1, maxattr:= max(is_attributed), by = cols ]
  df[ct == 1, maxattr:= 0 ]
  df[ct>1, seq := .N:1 , by = cols ]
  df[ct == 1, seq := 0 ]
  # Extract the dupes with a case
  removeDupe = rep(0, nrow(df))
  removeDupe[(df$maxattr==1) & (df$is_attributed==0)] = 1
  # Extract the final of the dupes wihout a case
  removeDupe[(df$maxattr==0) & (df$seq>1)] = 1
  return(removeDupe)
}

trndf = fread(paste0(path, 'trainval.csv'))
cols = c("ip", "app", "device", "os", "channel", "click_time")

tstdf = fread(paste0(path, 'testval.csv'))
cols = c("ip", "app", "device", "os", "channel", "click_time")

removeDupeIdtrn = removeDupeId(trndf)
removeDupeIdtst = removeDupeId(tstdf)

table(removeDupeIdtst)

fname = "removedupetrnval.gz"
write.csv(removeDupeIdtrn, 
          gzfile(paste0(path, '../features/', fname)), 
          row.names = F, quote = F)

fname = "removedupetstval.gz"
write.csv(removeDupeIdtst, 
          gzfile(paste0(path, '../features/', fname)), 
          row.names = F, quote = F)
