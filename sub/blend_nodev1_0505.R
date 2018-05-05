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

rnkr = function(vec){
  dt = data.table(vec)
  dt[, idx := 1:.N]
  dt = dt[order(vec)]
  dt[, rank := (1:.N)/.N]
  rank = dt[order(idx)]$rank
  return(rank)
}

tstdf = fread(paste0(path, 'test.csv'))
sub  = fread(paste0(path, '../sub/sub_lgb2404_orig.csv'))
subd1 = fread(paste0(path, '../sub/sub_lgb0505_nodev1.csv'), skip = 1)
tstdf[, pred := sub$is_attributed]
idx  = tstdf$device != 1
plot(tstdf[idx]$pred, subd1$V2)
tstdf[idx]$pred = tstdf[idx]$pred*0.5 + subd1$V2*0.5
subout = tstdf[,.(click_id, pred)]
setnames(subout, c("click_id", "is_attributed"))
fwrite(subout, paste0(path, "../sub/sub_lgb2404_with_nodev1.csv"), row.names = F)



length(tstdf[idx]$click_id)
table(tstdf[idx]$click_id - subd1$V1)
plot(tstdf[idx]$pred, subd1$V2)

# tstdf = fread(paste0(path, 'testval.csv'))
# sub  = fread(paste0(path, '../sub/sub_lgb2404val_difftime_thresh35.csv'), skip = 1)
# subd1 = fread(paste0(path, '../sub/sub_0505_nodev1val.csv'), skip = 1)
# tstdf[, pred := sub$V1]
# 
# auc(tstdf$is_attributed, sub$V1)
# # [1] 0.9832104
# idx  = tstdf$device != 1
# auc(tstdf$is_attributed[idx], tstdf$pred[idx])
# # [1] 0.9865915
# auc(tstdf$is_attributed[idx], subd1$V1)
# # [1] 0.9882552
# tstdf[idx]$pred = tstdf[idx]$pred*0.5 + subd1$V1*0.5
# auc(tstdf$is_attributed, tstdf$pred)
# # [1] 0.9832766

