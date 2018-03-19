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
<<<<<<< HEAD
trndf = fread(paste0(path, 'train.csv'))
#trndf = fread(paste0(path, 'trainvalsmall.csv'))
trndf = trndf[, keepcols, with=F]
tstdf = fread(paste0(path, 'old/test.csv'))
#tstdf = fread(paste0(path, 'testvalsmall.csv'))
=======
#trndf = fread(paste0(path, 'train.csv'))
trndf = fread(paste0(path, 'trainvalsmall.csv'))
trndf = trndf[, keepcols, with=F]
#tstdf = fread(paste0(path, 'old/test.csv'))
tstdf = fread(paste0(path, 'testvalsmall.csv'))
>>>>>>> 2961af6dd2a130f6d3c60204ff2d0bdf84a7aded
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


# get the entropy features
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

# trndf = merge(trndf, outipdevos, by="ip", how = "left")
<<<<<<< HEAD
# table(cut2(trndf$ip_devos_entropy, g= 20), trndf$is_attributed)
=======
# table(cut2(trndf$ip_devos_entropy, g= 20), trndf$is_attributed)
>>>>>>> 2961af6dd2a130f6d3c60204ff2d0bdf84a7aded
