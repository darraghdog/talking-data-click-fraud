#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)


path = '~/tdata/data/'
#path = '/Users/dhanley2/Documents/tdata/data/'

source(paste0(path, '../features/make/utils.R'))

# Write out the <ip, device, os> level
keepcols = c("ip", "os", "device", "is_attributed")
trndf = fread(paste0(path, 'train.csv'))
trndf = trndf[,keepcols,with=F]
trndf[,is_train:=1]
gc(); gc()
tstdf = fread(paste0(path, 'test.csv'))
tstdf = tstdf[,keepcols[1:(length(keepcols)-1)],with=F]
tstdf[, is_train:=0]
tstdf[, is_attributed := NA]
gc(); gc()
alldf = rbind(trndf, tstdf)
rm(trndf, tstdf)
gc(); gc()


#reponse encoding with leave-one-out and credibility adjustment
base_ft <- alldf$is_train ==1
mean_y0 = mean(alldf$is_attributed, na.rm = T)
calc_exp2(alldf, base_ft, 'is_attributed', c('ip'), 'exp2_ip', k = 1000, mean_y0 = mean_y0, verbose=T)
calc_exp2(alldf, base_ft, 'is_attributed', c('ip', 'device', 'os'), 'exp2_ipdevos', k = 500, mean_y0 = mean_y0, verbose=T)

hist(alldf$exp2_ip)
hist(alldf$exp2_ipdevos)



