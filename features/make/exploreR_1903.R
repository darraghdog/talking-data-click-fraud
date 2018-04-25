#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)

path = '~/tdata/data/'
path = '/Users/dhanley2/Documents/tdata/data/'

rnkr = function(vec){
  dt = data.table(vec)
  dt[, idx := 1:.N]
  dt = dt[order(vec)]
  dt[, rank := (1:.N)/.N]
  rank = dt[order(idx)]$rank
  return(rank)
}

tstdf = fread(paste0(path, 'test.csv'))
tstdf = tstdf[, .(device)]
sub = fread(paste0("/Users/dhanley2/Downloads/", "blend_lgb_nn_2subs_FULL_2.csv"))
sub[, is_attributed:= rnkr(is_attributed)]
hist(sub$is_attributed)
hist(sub[tstdf$device==3]$is_attributed)
sub[tstdf$device==3, is_attributed := (is_attributed/5)]
hist(sub[tstdf$device==3]$is_attributed)
hist(sub$is_attributed)



chk_app = 23
keep  = c('ip', 'os', "app", "device", "channel", "click_time", "is_attributed")
trndf = fread(paste0(path, 'train.csv'))
tstdf = fread(paste0(path, 'test.csv'))
trndf = trndf[, keep, with = F]
gc(); gc()


aggdf = trndf[, .(.N, sum(is_attributed)/.N), by =.(app,channel)]
aggdf = aggdf[N>20][V2>0.5]

tstdf$pred = subl$V2


tsagg = merge(tstdf, aggdf, by.x = c("app", "channel"), by.y = c("app", "channel"), "inner")
tsagg = tsagg[, .(.N, sum(V2)/.N), by =.(app,channel)]
hist(tsagg $V2)

ips_ = aggdf[N>10][V2>0.7]$ip

377*0.973

aggdf[ip==118524]
trndf[ip==118524]
tstdf[ip==118524]

subl = fread(paste0(path, '../sub/sub_lgb0304C.csv'), skip = 1)
hist(subl[tstdf$os==61]$V2)
sum(subl$V2>0.95)

for (ip_ in ips_) {
  if(ip_ %in% tstdf$ip){
    print(ip_)
    idx1 = tstdf$ip==ip_
    idx2 = trndf$ip==ip_
    print(sum(subl$V2>mean(subl[idx1]$V2)) / mean(trndf$is_attributed))
    print(nrow(subl[idx1]))
    print(nrow(trndf[idx2]))
    print(mean(subl[idx1]$V2))
    print('--------------------------')
  }
}

(sum(trndf$is_attributed)/10)/21


trndf[, ct:= .N, by = app]
nrow(trndf[device==3032]) # 692891
trndf[, click_sec := as.numeric(fasttime::fastPOSIXct(click_time))]
trndf[, click_sec := click_sec - min(trndf$click_sec)]
trndf[, click_day := round((click_sec)/(24*3600))]
aggdf = trndf[ct>100000, .(mean(is_attributed), sum(is_attributed), .N), by = app]
setnames(aggdf, c("app", "ymean", "ysum", "count"))
aggdf = aggdf[order(ymean)]



idx = 1:1000000
trndf = fread(paste0(path, 'trainval.csv'))
trndf[, sec_seq := 1:.N , by = click_time]
cols_ = c("ip", "device", "os", "app")
trndf$click_sec = as.numeric(fasttime::fastPOSIXct(trndf$click_time))
trndf[, click_sec :=  click_sec - min( click_sec)] 
trndf[,ct := .N, by = click_time]
trndf[, split_sec := round((0:(.N-1))/.N, 8), by = click_time]
trndf[, ct_sec := .N , by = click_time]
trndf[, click_split_sec := click_sec + split_sec]
trndf[, index := 1:nrow(trndf)]
trndf = trndf[order(ip, os, device, app, click_split_sec)]
trndf
trndf[,click_sec_shift_lead := - click_split_sec + shift(click_split_sec, 1, type = "lead")]
trndf[,seq_lead := .N:1, by = cols_ ]
trndf[seq_lead == 1, click_sec_shift_lead := -1]

trndf[(ip==6) & (app==19) & (device == 16) & (os==0)]

idx = 1:3000000
idx = idx[trndf[idx]$click_sec_shift_lead!=-1]
table(cut2(round(1000*trndf[idx]$click_sec_shift_lead), g = 50), trndf[idx]$is_attributed)


trndf[click_sec_shift_lead==-1][1:1000]

# 0        1
# FALSE 51121554    34771
# TRUE  10807505   116171


getSplitLead = function(df, cols_, fname, path, shift_n = 1){
  df$click_sec = as.numeric(fasttime::fastPOSIXct(df$click_time))
  df[, split_sec := round((0:(.N-1))/.N, 4), by = click_time]
  df = df[,c(cols_, "click_sec", "split_sec"), with = F]
  df[, index := 1:nrow(df)]
  setorderv(df, c(cols_, "click_sec", "split_sec"))
  df[,click_sec_shift_lead := shift(click_sec+split_sec, shift_n, type = "lead")]
  df[,seq_lead := .N:1, by = cols_ ]
  df[,click_sec_lead := click_sec_shift_lead - (click_sec + split_sec)]
  df[,click_sec_lead := round(click_sec_lead, 4)]
  df[seq_lead %in% 1:shift_n, click_sec_lead := 999999]
  setorderv(df, "index")
  new_name = "click_sec_lead_split_sec"
  setnames(df, "click_sec_lead", new_name)
  df = df[,new_name,with=F]
  return(df)
}



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


