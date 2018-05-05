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

cols = c("ip", "app", "device", "os", "channel", "click_time")
trndf = fread(paste0(path, 'trainval.csv'))
tstdf = fread(paste0(path, 'testval.csv'))
trnalldf = fread(paste0(path, 'train.csv'))
tstalldf = fread(paste0(path, 'test.csv'))

removeDupeIdtrn = removeDupeId(trndf)
removeDupeIdalltrn = removeDupeId(trnalldf)
removeDupeIdtst = removeDupeId(tstdf)
removeDupeIdalltst = rep(0, nrow(tstalldf))

sum(removeDupeIdtst)/length(removeDupeIdtst)
sum(removeDupeIdtrn)/length(removeDupeIdtrn)
sum(removeDupeIdalltrn)/length(removeDupeIdalltrn)

fname = "removedupetrnval.gz"
write.csv(removeDupeIdtrn, 
          gzfile(paste0(path, '../features/', fname)), 
          row.names = F, quote = F)

fname = "removedupetstval.gz"
write.csv(removeDupeIdtst, 
          gzfile(paste0(path, '../features/', fname)), 
          row.names = F, quote = F)

fname = "removedupetrn.gz"
write.csv(removeDupeIdalltrn, 
          gzfile(paste0(path, '../features/', fname)), 
          row.names = F, quote = F)

fname = "removedupetst.gz"
write.csv(removeDupeIdalltst, 
          gzfile(paste0(path, '../features/', fname)), 
          row.names = F, quote = F)


tstalldf[, ct:= .N, by = cols ]
tstalldf[ct>1, seq:= .N:1, by = cols ]
tstalldf[ct==1, seq:= 0]
# Make all dupes except last to zero
removeDupeIdalltst[tstalldf$seq>1] = 1
tstalldf[, rdupe := removeDupeIdalltst ]
View(tstalldf[1:10000])

fname = "removedupetstfilt.gz"
write.csv(removeDupeIdalltst, 
          gzfile(paste0(path, '../features/', fname)), 
          row.names = F, quote = F)

sub = fread(paste0(path, "../sub/sub_lgb0205.csv"))
sub[removeDupeIdalltst==1, is_attributed := 0]
fname = "sub_lgb0205_nodupe.csv.gz"
write.csv(sub, 
          gzfile(paste0(path, '../sub/', fname)), 
          row.names = F, quote = F)

sublgn = fread(paste0(path, "../sub/sub_lgb0205.csv"))
sublg = fread(paste0(path, "../sub/sub_lgb2404.csv"))
subnn = fread(paste0(path, "../sub/sub_nnet2104.csv"))
sublg[removeDupeIdalltst==1, is_attributed := 0]
sublgn[removeDupeIdalltst==1, is_attributed := 0]
subnn[removeDupeIdalltst==1, is_attributed := 0]
cor(rnkr(sublgn$is_attributed), rnkr(sublg$is_attributed))
cor(rnkr(subnn$is_attributed), rnkr(sublg$is_attributed))
cor(rnkr(subnn$is_attributed), rnkr(sublgn$is_attributed))


rnkr = function(vec){
  dt = data.table(vec)
  dt[, idx := 1:.N]
  dt = dt[order(vec)]
  dt[, rank := (1:.N)/.N]
  rank = dt[order(idx)]$rank
  return(rank)
}
