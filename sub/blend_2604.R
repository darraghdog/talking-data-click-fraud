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

subbl = fread(paste0(path, "../sub/blend_0426_1.csv"))[order(click_id)]
subwb = fread(paste0(path, "../sub/wordbatch_fm_ftrl_public.csv"))[order(click_id)]
subbl[, is_attributed := rnkr(is_attributed)]
subwb[, is_attributed := rnkr(is_attributed)]

cor(subbl$is_attributed, subwb$is_attributed)

sub = subwb
sub[, is_attributed := (0.05*subwb$is_attributed + 0.95*subbl$is_attributed)]

fwrite(sub, paste0(path, "../sub/blend0426__X95__wbpublic___X05.csv"), row.names = F)
