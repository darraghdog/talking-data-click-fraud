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
subnn = fread(paste0(path, "../sub/sub_nnet2104.csv"))[order(click_id)]
subwb = fread(paste0(path, "../sub/wordbatch_fm_ftrl.csv"))[order(click_id)]

sub99[, is_attributed := rnkr(is_attributed)]
sub40[, is_attributed := rnkr(is_attributed)]
sub35[, is_attributed := rnkr(is_attributed)]
sub30[, is_attributed := rnkr(is_attributed)]
sub25[, is_attributed := rnkr(is_attributed)]
subnn[, is_attributed := rnkr(is_attributed)]
subwb[, is_attributed := rnkr(is_attributed)]

cor(sub25$is_attributed, sub99$is_attributed)
cor(sub30$is_attributed, sub99$is_attributed)
cor(sub35$is_attributed, sub99$is_attributed)
cor(sub40$is_attributed, sub99$is_attributed)
cor(subnn$is_attributed, sub99$is_attributed)
cor(subwb$is_attributed, sub99$is_attributed)

sub = sub99
sub[, is_attributed := (0.2*sub99$is_attributed +
                          0.2*sub40$is_attributed + 
                          0.2*sub35$is_attributed + 
                          0.2*sub30$is_attributed +
                          0.2*sub25$is_attributed)]

cor(sub$is_attributed, subnn$is_attributed)
cor(sub$is_attributed, subwb$is_attributed)

fwrite(sub, paste0(path, "../sub/blend_lgb_thresh_25K_30K_35K_40K_100K.csv"), row.names = F)
