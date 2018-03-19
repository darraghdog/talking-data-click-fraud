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
  #sum2[, dummy:=1]
  #dowdf <- data.table(x=unique(df[[subgrp]]), dummy=1)
  #setnames(dowdf, c(subgrp, 'dummy'))
  #sum2a <- merge(sum2, dowdf, by='dummy', allow.cartesian=T)
  sum3 <- merge(sum2, sum1, by=c(group))
  #sum3[is.na(subgrpcnt), subgrpcnt:=0]
  sum3[, entropy := - log(subgrpcnt * 1.0 / cnt) * subgrpcnt * 1.0 / cnt]
  sum3[is.na(entropy), entropy := 0]
  sum4 <- sum3[, sum(entropy), by=list(sum3[[group]])]
  setnames(sum4, c(group, paste(tgt_vn_prefix, 'entropy', sep='_')))
  return(sum4)
}

# Write out the <ip, device, os> level
trndf = fread(paste0(path, 'trainvalsmall.csv'))
trndf