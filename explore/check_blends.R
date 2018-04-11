#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)
library(Metrics)

rnkr = function(vec){
  dt = data.table(vec)
  dt[, idx := 1:.N]
  dt = dt[order(vec)]
  dt[, rank := (1:.N)/.N]
  rank = dt[order(idx)]$rank
  return(rank)
}


path = '~/tdata/data/'
path = '/Users/dhanley2/Documents/tdata/data/'
source(paste0(path, "../features/make/utils.R"))
y = fread(paste0(path, '../sub/y_val.csv'))$V1
y[1:100]

subl = fread(paste0(path, '../sub/sub_lgb0604Cval.csv'), skip = 1)$V1
auc(y, subl)
# [1] 0.9830236
# 
#./ffm-train -s 8 -t 15 -r 0.05 --auto-stop -p ../weights/test_dfval.ffm ../weights/train_dfval.ffm model
subo = fread(paste0(path, '../libffm/output'))$V1
auc(y, subo)
# [1] 0.9560032
# [1] 0.9688788 ... 5x oversampling
# [1] 0.9683376 ... 10x oversampling

rsubo = rnkr(subo)
rsubl = rnkr(subl)

auc(y, (0.999*rsubl)+ (0.001*rsubo))
