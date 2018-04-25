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


sub1 = fread(paste0(path, "../sub/sub_lgb2404.csv"))[order(click_id)]
sub2 = fread(paste0(path, "../sub/blend_lgb_nn_2subs_FULL_2 2.csv"))[order(click_id)]
sub3 = fread(paste0(path, "../sub/sub_lgb2404_difftime_thresh35.csv"))[order(click_id)]
sub1[, is_attributed := rnkr(is_attributed)]
sub2[, is_attributed := rnkr(is_attributed)]
sub3[, is_attributed := rnkr(is_attributed)]
#sublg[, is_attributed := rnkr(is_attributed)]
cor(sub1$is_attributed, sub2$is_attributed)
# [1] 0.9850251
cor(sub1$is_attributed, sub3$is_attributed)
# [1] 0.9916751
cor(sub2$is_attributed, sub3$is_attributed)
# [1] 0.9823914


sub1
sub2
sub3
all.equal(sub1$click_id, sub2$click_id)

sub = sub1
sub[, is_attributed := (0.5*sub2$is_attributed) + (0.3*sub1$is_attributed) + (0.2*sub3$is_attributed)]
sub[, is_attributed := format(is_attributed, scientific = F) ]
sub
fwrite(sub, paste0(path, "../sub/sub_lgb2404__X30___sub_lgb2404_thresh35__X20____blend_lgb_nn_2subs_FULL_2__X50.csv"), row.names = F)
