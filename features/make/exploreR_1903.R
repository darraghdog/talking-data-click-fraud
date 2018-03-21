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
trndf = fread(paste0(path, 'trainvalsmall.csv'))
gc(); gc()

# Make the full training data
trndf[,click_time := fasttime::fastPOSIXct(click_time)]
trndf[,click_hr   := as.numeric(format(click_time, "%H"))]
gc();gc()


clknite = trndf[click_hr %in% 19:21, .(sum(is_attributed), .N), by=.(ip)]
clkday = trndf[click_hr %in% c(2:16), .(sum(is_attributed), .N), by=.(ip)]
setnames(clknite, c("ip", "ysumday", "ctnite"))
setnames(clkday, c("ip", "ysumnite", "ctday"))
clk = merge(clkday, clknite, by = "ip", type = "all")
clk

glob_mean = sum(clk$ctnite)/sum(clk$ctday)
glob_mean_click = (sum(clk$ysumday)+sum(clk$ysumnite))/(sum(clk$ctnite)+sum(clk$ctday))

prior_niteday = 20
prior_clicks = 100
clk[,bmean := (((ctday+ctnite)*(ctnite/ctday)) + (prior_niteday*glob_mean))/((ctday+ctnite)+prior_niteday)]
clk[,bmean_click := ((ysumnite+ysumday) + (prior_clicks*glob_mean_click))/( prior_clicks+(ctday+ctnite))]
clk[order(bmean)]
clk
plot(clk$bmean, clk$bmean_click, ylim = c(0,.005))
abline(h=(glob_mean_click), col="red")


