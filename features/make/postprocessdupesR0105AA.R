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

tstdf = fread(paste0(path, 'test.csv'))[order(click_id)]
subbl = fread(paste0(path, "../sub/blend_0426_1.csv"))[order(click_id)]
tstdf[, is_attributed := rnkr(subbl$is_attributed)]

cols = c("ip", "app", "device", "os", "channel", "click_time")
tstdf[, ct:= .N, by = cols ]
#tstdf[ct>1, seq:= .N:1, by = cols ]
#tstdf[ct==1, seq:= 0]

tstdf[ct>1, seq:= 1:.N, by = cols ]
tstdf[ct==1, seq:= 0]
tstdf[seq>1, is_attributed := 0 ]

# Make all dupes except last to zero
#tstdf[seq>1, is_attributed := 0 ]
# Write out
fwrite(tstdf[, .(click_id, is_attributed)], paste0(path, "../sub/blend0426___post_process_dupesAA.csv"), row.names = F)
