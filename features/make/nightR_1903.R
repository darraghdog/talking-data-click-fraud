#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)

path = '~/tdata/data/'
path = '/Users/dhanley2/Documents/tdata/data/'

# Write out the <ip, device, os> level
keepcols = c("ip", "os", "device", "click_time")
trndf = fread(paste0(path, 'train.csv'))
trndf = trndf[,keepcols,with=F]
gc(); gc()
tstdf = fread(paste0(path, 'old/test.csv'))
tstdf = tstdf[,keepcols,with=F]
gc(); gc()
alldf = rbind(trndf, tstdf)
rm(trndf, tstdf)
gc(); gc()

# Make the full training data
alldf[, click_time := fasttime::fastPOSIXct(click_time)]
alldf[, click_hr   := as.numeric(format(click_time, "%H"))]
alldf[, click_time := NULL ]
gc();gc()

niteClicksProp = function(df, prior = 40, cols_ = c("ip", "os", "device")){
  clknite = df[click_hr %in% 19:21, .(.N), by=cols_]
  clkday = df[click_hr %in% c(2:16), .(.N), by=cols_]
  setnames(clknite, c(cols_, "ctnite"))
  setnames(clkday, c(cols_, "ctday"))
  clk = merge(clkday, clknite, by = cols_, type = "all")
  glob_mean = sum(clk$ctnite)/sum(clk$ctday)
  prior_niteday = prior
  clk[,bmean_nite_clicks := (((ctday+ctnite)*(ctnite/ctday)) + (prior_niteday*glob_mean))/((ctday+ctnite)+prior_niteday)]
  return(clk[,c(cols_, "bmean_nite_clicks"), with=F])
}

nitedf1 = niteClicksProp(alldf, 40, cols_ = c("ip", "os", "device"))
nitedf2 = niteClicksProp(alldf, 40, cols_ = c("ip"))
names(nitedf1)[4] = paste0(names(nitedf1)[4], "_ipdevos")
names(nitedf2)[2] = paste0(names(nitedf2)[2], "_ip")

# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(nitedf1, 'niteratio_ipdevos')
writeme(nitedf2, 'niteratio_ip')
