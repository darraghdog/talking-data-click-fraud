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


sub99 = fread(paste0(path, "../sub/sub_lgb1704.csv"))[order(click_id)]
sub40 = fread(paste0(path, "../sub/sub_lgb2404.csv"))[order(click_id)]
sub35 = fread(paste0(path, "../sub/sub_lgb2404_difftime_thresh35.csv"))[order(click_id)]
sub30 = fread(paste0(path, "../sub/sub_lgb2404_difftime_thresh30.csv"))[order(click_id)]
sub25 = fread(paste0(path, "../sub/sub_lgb2404_difftime_thresh25.csv"))[order(click_id)]
sub10 = fread(paste0(path, "../sub/sub_lgb2404_difftime_thresh10.csv"))[order(click_id)]
subnn = fread(paste0(path, "../sub/sub_nnet2104.csv"))[order(click_id)]
subwb = fread(paste0(path, "../sub/wordbatch_fm_ftrl.csv"))[order(click_id)]

sublg = copy(sub99)
sublg[, is_attributed := ((sub99$is_attributed^(1/6)) *
                          (sub40$is_attributed^(1/6)) *
                          (sub10$is_attributed^(1/6)) *
                          (sub35$is_attributed^(1/6)) * 
                          (sub30$is_attributed^(1/6)) *
                          (sub25$is_attributed^(1/6)))]

sublg[, is_attributed := rnkr(is_attributed)]
subnn[, is_attributed := rnkr(is_attributed)]
subwb[, is_attributed := rnkr(is_attributed)]
sub99[, is_attributed := rnkr(is_attributed)]

cor(subnn$is_attributed, sub99$is_attributed)
cor(subwb$is_attributed, sub99$is_attributed)
cor(sublg$is_attributed, sub99$is_attributed)

sub = copy(sub99)
sub[, is_attributed := (0.7*sublg$is_attributed +
                          0.25*subnn$is_attributed + 
                          0.05*subwb$is_attributed)]

cor(sub$is_attributed, subnn$is_attributed)
cor(sub$is_attributed, subwb$is_attributed)
cor(sub$is_attributed, sublg$is_attributed)

fwrite(sub, paste0(path, "../sub/blend_lgb_nn_wb_3004.csv"), row.names = F)
