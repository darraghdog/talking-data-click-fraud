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
subbl = fread(paste0(path, "../sub/sub_lgb0405.csv"))[order(click_id)]
tstdf[, is_attributed1 := subbl$is_attributed]
tstdf[, is_attributed := subbl$is_attributed]

#tstdf[, is_attributed := rnkr(subbl$is_attributed)]

cols = c("ip", "app", "device", "os", "channel", "click_time")
tstdf[, ct:= .N, by = cols ]
tstdf[ct>1, seq:= .N:1, by = cols ]
tstdf[ct==1, seq:= 0]
table(tstdf$seq)
# Make all dupes except last to zero, and make last the max of all the dupes
tstdf[seq>0, is_attributed := max(is_attributed), by = cols ]
tstdf[seq>1, is_attributed := 0 ]

plot(tstdf[seq==1]$is_attributed, tstdf[seq==1]$is_attributed1) 

# Write out
fwrite(tstdf[, .(click_id, is_attributed)], paste0(path, "../sub/sub_lgb_0405___post_processed.csv"), row.names = F)
