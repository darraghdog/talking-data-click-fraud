#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)

path = '~/tdata/data/'
#path = '/Users/dhanley2/Documents/tdata/data/'

trndf = fread(paste0(path, 'trainval.csv'))
tstdf = fread(paste0(path, 'testval.csv'))




