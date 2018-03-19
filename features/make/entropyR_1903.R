#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)

path = '~/tdata/data/'
#path = '/Users/dhanley2/Documents/tdata/data/'

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

entropyHrMin = function(df, user, add_string){
  ip_visit_hr_entropy <- calc_entropy(df, user,  'click_hr', paste0(add_string, '_visit_hr'))
  ip_visit_min_entropy <- calc_entropy(df, user , 'click_min', paste0(add_string, '_visit_min'))
  outdf = merge(ip_visit_hr_entropy, ip_visit_min_entropy, by = user, how="all")
  return(outdf)
}

# Write out the <ip, device, os> level
keepcols = c('ip', 'os', 'device', 'click_time', 'is_attributed')
trndf = fread(paste0(path, 'trainvalsmall.csv'))
trndf = trndf[, keepcols, with=F]
tstdf = fread(paste0(path, 'testvalsmall.csv'))
tstdf = tstdf[, keepcols[1:4], with=F]
tstdf$is_attributed = NA
gc(); gc()

# Make the full training data
alldf = rbind(trndf, tstdf)
alldf[,click_time := fasttime::fastPOSIXct(click_time)]
alldf[,click_hr   := as.numeric(format(click_time, "%H"))]
alldf[,click_min   := as.numeric(format(click_time, "%M"))]

# get the entropy features
outiphrmin = entropyHrMin(alldf, "ip", add_string = 'ip')
outipdevos <- calc_entropy(alldf, "ip", c('device'), 'ip_devos')

# Write out the files
write.csv(outiphrmin, 
          gzfile(paste0(path, '../features/entropy_ip_hr_minvalsmall.gz')), 
          row.names = F, quote = F)
write.csv(outipdevos, 
          gzfile(paste0(path, '../features/entropy_ip_devosvalsmall.gz')), 
          row.names = F, quote = F)


trndf = merge(trndf, outipdevos, by="ip", how = "left")
table(cut2(trndf$ip_devos_entropy, g= 20), trndf$is_attributed)
