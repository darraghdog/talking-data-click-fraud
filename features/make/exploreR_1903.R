#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)

path = '~/tdata/data/'
#path = '/Users/dhanley2/Documents/tdata/data/'

# Write out the <ip, device, os> level
trndf = fread(paste0(path, 'train.csv'))
trndf[,attributed_time:=NULL]
trndf[,channel:=NULL]
gc(); gc()

# Make the full training data
trndf[,click_time := as.numeric(fasttime::fastPOSIXct(click_time))]
trndf[,click_hr   := round(click_time/3600) %%24]
trndf[,click_hr   := click_hr]
trndf[,click_day   := round(click_time/(24*3600))]
trndf[,click_time:=NULL]
gc(); gc()

aggdf = trndf[,.(.N, sum(is_attributed)), by = .(ip, device, os, click_hr, click_day)]
rm(trndf)
gc(); gc()
setnames(aggdf, "N", "ct_click")
setnames(aggdf, "V2", "y_click")
aggdf[,ipdevosapp_ct :=sum(ct_click), by = .(ip, device, os)]
aggdf = aggdf[ipdevosapp_ct>5000]

aggdf = aggdf[order(ip, device, os, click_hr, click_day)]
aggdf[,seq := 1:.N, by = .(ip, device, os, click_hr)]
aggdf[, ct_click_lag := shift(ct_click, 1, type="lag")]
aggdf[, y_click_lag := shift( y_click, 1, type="lag")]
glob_mean = sum(aggdf$y_click)/sum(aggdf$ct_click)
aggdf[seq>1, bmean_lag := ((y_click_lag)+(glob_mean*2000))/(ct_click_lag+2000)]
aggdf[seq>1, bmean := ((y_click)+(glob_mean*2000))/(ct_click+2000)]


plot(table(cut2(round(aggdf$bmean,5), g=10), cut2(round(aggdf$bmean_lag,5), g=10)))

View(aggdf[1:2000]) 

#########################################################################

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


