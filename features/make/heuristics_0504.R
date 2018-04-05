#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)

path = '~/tdata/data/'
path = '/Users/dhanley2/Documents/tdata/data/'

chk_app = 23
keep  = c("app", "device", "click_time", "is_attributed")
trndf = fread(paste0(path, 'train.csv'))
keep  = c("app", "device", "click_time")
tstdf = fread(paste0(path, 'test.csv'))
trndf = trndf[, keep, with = F]
gc(); gc()
trndf[, ct:= .N, by = app]
nrow(trndf[device==3032]) # 692891
trndf[, click_sec := as.numeric(fasttime::fastPOSIXct(click_time))]
trndf[, click_sec := click_sec - min(trndf$click_sec)]
trndf[, click_day := round((click_sec)/(24*3600))]
aggdf = trndf[ct>100000, .(mean(is_attributed), sum(is_attributed), .N), by = app]
setnames(aggdf, c("app", "ymean", "ysum", "count"))
aggdf = aggdf[order(ymean)]
head(aggdf, 30)


near_zero_apps = c(151, 56, 4, 23)
table(trndf[app %in% 23]$click_day)

mean_nz_trn = mean(trndf[app %in% near_zero_apps]$is_attributed)
# 1.776085e-05
1/mean_nz_trn

tstdf = tstdf[order(click_id)]
idx = tstdf$app %in% near_zero_apps
nrow(tstdf[idx])/nrow(tstdf)

sub = fread(paste0(path, "../sub/sub_lgb0304C.csv"))[order(click_id)]
subg = fread(paste0(path, "../sub/blend_lgb0304C_giba_lgb0404_FULL.csv"))[order(click_id)]

all.equal(sub$click_id, tstdf$click_id)
all.equal(subg$click_id, tstdf$click_id)
mean_nz_trn
mean(sub[idx]$is_attributed) 
mean(subg[idx]$is_attributed) 
mean(sub$is_attributed)

mean(sub[idx]$is_attributed) /mean(mean_nz_trn)
mean(subg[idx]$is_attributed) /mean(mean_nz_trn)
mean(sub$is_attributed)/mean(trndf$is_attributed)

mean(as.numeric(subg$is_attributed>0.05))
mean(as.numeric(subg[idx]$is_attributed>0.05))
sum (as.numeric(subg[idx]$is_attributed>0.05))

hist(subg[idx]$is_attributed, breaks = 1000, ylim = c(0, 100))

subh = subg
subh[idx]$is_attributed = subg[idx]$is_attributed / 5
fwrite(subh, file = paste0(path, "../sub/sub_app_heuristics.csv"), row.names = F)

head(subh[idx]$is_attributed)
head(subg[idx]$is_attributed)


