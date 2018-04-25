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

subnn = fread(paste0(path, "../sub/sub_nnet2104.csv"))
subnn[, click_id := as.integer(click_id)]
#sub[, is_attributed := format(is_attributed, scientific = F)]
#fwrite(sub, paste0(path, "../sub/sub_nnet2104_proc.csv"))
subnn[ , is_attributed := rnkr(is_attributed)]
subnn = subnn[order(click_id)]

sublg = fread(paste0(path, "../sub/blend_last_3subs_FULL_plus_kernel_plus_giba_2.csv"))
sublg[, click_id := as.integer(click_id)]
sublg[, is_attributed := rnkr(is_attributed)]
sublg = sublg[order(click_id)]

subnn
sublg
all.equal(subnn$click_id, sublg$click_id)
idx = subnn$click_id != sublg$click_id
table(idx)

cor(subnn$is_attributed, sublg$is_attributed)

subbl = sublg
subbl[, is_attributed := (0.2*subnn$is_attributed) + (0.8*sublg$is_attributed)]
subbl[, is_attributed := format(is_attributed, scientific = F) ]
fwrite(subbl, paste0(path, "../sub/sub_nnet2104_proc_<2>_____blend_last_3subs_FULL_plus_kernel_plus_giba_2_<8>.csv"), row.names = F)
