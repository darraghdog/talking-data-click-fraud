#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)

path = '~/tdata/data/'
path = '/Users/dhanley2/Documents/tdata/data/'

#compute entropy by group, over subgrp
calc_entropy <- function(df, group, subgrp, tgt_vn_prefix) {
  sum1 <- df[, .N, by=list(df[[group]], df[[subgrp]])]
  setnames(sum1, c(group, subgrp, 'subgrpcnt'))
  sum2 <- df[, .N, by=list(df[[group]])]
  setnames(sum2, c(group, 'cnt'))
  sum3 <- merge(sum2, sum1, by=c(group))
  sum3[, entropy := - log(subgrpcnt * 1.0 / cnt) * subgrpcnt * 1.0 / cnt]
  sum3[is.na(entropy), entropy := 0]
  sum4 <- sum3[, sum(entropy), by=list(sum3[[group]])]
  setnames(sum4, c(group, paste(tgt_vn_prefix, 'entropy', sep='_')))
  return(sum4)
}

# Write out the <ip, device, os> level
keepcols = c('ip', 'os', 'device', 'app', 'channel','click_time')
trndf = fread(paste0(path, 'train.csv'))
#trndf = fread(paste0(path, 'trainvalsmall.csv'))
trndf = trndf[, keepcols, with=F]
tstdf = fread(paste0(path, 'testfull.csv'))
#tstdf = fread(paste0(path, 'testvalsmall.csv'))
tstdf = tstdf[, keepcols, with=F]
gc(); gc()

# Make the full training data
alldf = rbind(trndf, tstdf)
rm(tstdf, trndf)
gc();gc()
alldf[,click_time := fasttime::fastPOSIXct(click_time)]
alldf[,click_hr   := as.numeric(format(click_time, "%H"))]
alldf[,click_min   := as.numeric(format(click_time, "%M"))]
rm(tstdf, trndf)
gc();gc()


# get the entropy features per ip
entropyip  <- calc_entropy(alldf, "ip", 'device', 'ip_device')
entropyip  <- merge(entropyip, calc_entropy(alldf, "ip", 'os', 'ip_os'), by = 'ip', how="all")
entropyip  <- merge(entropyip, calc_entropy(alldf, "ip", 'app', 'ip_app'), by = 'ip', how="all")
entropyip  <- merge(entropyip, calc_entropy(alldf, "ip", 'channel', 'ip_channel'), by = 'ip', how="all")
entropyip  <- merge(entropyip, calc_entropy(alldf, "ip", 'click_hr', 'ip_click_hr'), by = 'ip', how="all")
entropyip  <- merge(entropyip, calc_entropy(alldf, "ip", 'click_min', 'ip_click_min'), by = 'ip', how="all")

entropydev <- calc_entropy(alldf, "device", 'os', 'device_os')
entropydev  <- merge(entropydev, calc_entropy(alldf, "device", 'channel', 'device_channel'), by = 'device', how="all")

entropyapp  <- calc_entropy(alldf, "app", 'channel', 'app_channel')
entropychl  <- calc_entropy(alldf, "channel", 'app', 'channel_app')

# get the entropy features per ip, dev, os
ido = alldf[,.(length(click_hr)), by = c("ip", "device", "os")]
ido[,ipdevosId:=1:nrow(ido)]
ido[,V1:=NULL]
alldf = merge(alldf, ido, by = c("ip", "device", "os"), how = "all")
gc(); gc(); gc(); gc(); gc(); gc()
entropyipdevos  <- calc_entropy(alldf, "ipdevosId", 'app', 'ipdevos_app')
entropyipdevos  <- merge(entropyipdevos, calc_entropy(alldf, "ipdevosId", 'click_min', 'ipdevos_click_min'), 
                         by = "ipdevosId", how="all")
entropyipdevos  <- merge(entropyipdevos, calc_entropy(alldf, "ipdevosId", 'channel', 'ipdevos_chl'), 
                         by = "ipdevosId", how="all")
entropyipdevos = merge(entropyipdevos, ido, by = "ipdevosId", how = "all")
entropyipdevos[,ipdevosId:=NULL]

# get the entropy features per ip, dev, os, app
idoa = alldf[,.(length(click_hr)), by = c("ip", "device", "os", "app")]
idoa[,ipdevosappId:=1:nrow(idoa)]
idoa[,V1:=NULL]
alldf = merge(alldf, idoa, by = c("ip", "device", "os", "app"), how = "all")
gc(); gc(); gc(); gc(); gc(); gc()
entropyipdevosapp  <- calc_entropy(alldf, "ipdevosappId", 'channel', 'ipdevosapp_chl')
entropyipdevosapp  <- merge(entropyipdevosapp, calc_entropy(alldf, "ipdevosappId", 'click_min', 'ipdevosapp_click_min'), 
                         by = "ipdevosappId", how="all")
entropyipdevosapp = merge(entropyipdevosapp, idoa, by = "ipdevosappId", how = "all")
entropyipdevosapp[,ipdevosappId:=NULL]


# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(entropyip, 'entropyip')
writeme(entropydev, 'entropydev')
writeme(entropyapp, 'entropyapp')
writeme(entropychl, 'entropychl')
writeme(entropyipdevos, 'entropyipdevos')
writeme(entropyipdevosapp, 'entropyipdevosapp')
