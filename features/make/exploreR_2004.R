#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)
library(Metrics)

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

tstdf = fread(paste0(path, 'testval.csv'))
tstdf[, click_sec := as.numeric(fasttime::fastPOSIXct(click_time))]
tstdf[, attributed_sec := as.numeric(fasttime::fastPOSIXct(attributed_time))]
tstdf[, click_hr := round(click_sec/3600)%%24]
tstdf[, difftime := attributed_sec - click_sec]
tstdf[, difftimecut := 0]
tstdf[!is.na(difftime), difftimecut := 1+as.integer(difftime>25000)]
table(tstdf$click_hr, tstdf$difftimecut)

tstdf[!is.na(difftime), difftime<5000 := 2]
tstdf[, max(difftime, na.rm = T), by = click_hr]








View(tstdf[,.(.N, as.integer(mean(is_attributed)*10000)), by = device])
View(tstdf[,.(.N, as.integer(mean(is_attributed)*10000)), by = os])
View(tstdf[,.(.N, as.integer(mean(is_attributed)*10000)), by = device])

trndf = fread(paste0(path, 'trainval.csv'))
sublg = fread(paste0(path, '../sub/sub_lgb1704val.csv'))
subnn = fread(paste0(path, '../sub/sub_nnet1404Aval.csv'))
subnn = rnkr(subnn$is_attributed)
sublg = rnkr(sublg$V1)[-1]
cor(sublg, subnn)
# [1] 0.9273761
auc(tstdf$is_attributed, sublg)
# [1] 0.9836714
auc(tstdf$is_attributed, subnn)
# [1] 0.9821893

auc(tstdf$is_attributed, (subnn*0.2)+(sublg*0.8))
# [1] 0.9838299

auc(tstdf$is_attributed, (subnn*0.1)+(sublg*0.9))
# [1] 0.9838299 
y_act = tstdf$is_attributed

idx = 1:10000000
tstdf
subdiff = (subnn - sublg)
lmod = lm(y_act[idx] ~ subnn[idx] + sublg[idx] + subdiff[idx])














keep = c("ip", "device", "os", "app")
tstdf = tstdf[,keep, with=F]
trndf = trndf[,c(keep, "is_attributed"), with=F]
tstdf[, pred := rnkr(sub$is_attributed)]

tot_attr = sum(trndf$is_attributed)
aggtrndf = trndf[,.(.N, sum(is_attributed), as.integer(mean(is_attributed)*10000),  as.integer((sum(is_attributed)/tot_attr)*10000)), by = .(ip,os,device)]
setnames(aggtrndf, c("ip", "os","device", "1_total_ct", "1_is_attr_ct", "1_mean_attr_10k", "1_is_attr_tot_ratio_10k"))
tot_attr = sum(tstdf$is_attributed)
aggdf = tstdf[,.(.N, as.integer(mean(pred)*10000)), by = .(ip,os,device)]
setnames(aggdf, c("ip", "os","device", "2_total_ct", "2_mean_pred"))

aggdf = merge(aggdf, aggtrndf, by = c("ip", "os","device"), all.x = T)
View(aggdf)





table(trndf$channel, trndf$is_attributed)







table(trndf[(!(device %in% unique(tstdf$device)) & !(ip %in% unique(tstdf$ip)) & !(os %in% unique(tstdf$os)) )])

idx = ((trndf$channel %in% unique(tstdf$channel)) )
table(trndf[idx]$channel, trndf[idx]$is_attributed)

table(trndf[device == 3866]$is_attributed)

table(trndf[device %in% unique(trndf[idx]$device)]$device, trndf[device %in% unique(trndf[idx]$device)]$is_attributed)

table(trndf[device==3032]$os, trndf[device==3032]$is_attributed)

table(trndf[is_attributed==1][(!(device %in% unique(tstdf$device)) & !(os %in% unique(tstdf$os)) )]$device)
table(trndf[is_attributed==1][(!(device %in% unique(tstdf$device)) & !(os %in% unique(tstdf$os)) )]$os)





tstdf = fread(paste0(path, 'testval.csv'))
trndf = fread(paste0(path, 'trainval.csv'))
sub   = fread(paste0(path, '../sub/sub_lgb0604Cval.csv'), skip = 1)
tstdf[, pred := rnkr(sub$V1)]

hist(tstdf[is_attributed==1]$pred)
hist(tstdf[is_attributed==0]$pred)

tot_attr = sum(trndf$is_attributed)
aggtrndf = trndf[,.(.N, sum(is_attributed), as.integer(mean(is_attributed)*10000),  as.integer((sum(is_attributed)/tot_attr)*10000)), by = channel]
setnames(aggtrndf, c("channel", "1_total_ct", "1_is_attr_ct", "1_mean_attr_10k", "1_is_attr_tot_ratio_10k"))


tot_attr = sum(tstdf$is_attributed)
aggdf = tstdf[,.(.N, sum(is_attributed), as.integer(mean(is_attributed)*10000),  as.integer((sum(is_attributed)/tot_attr)*10000), as.integer(mean(pred)*10000)), by = channel]
setnames(aggdf, c("channel", "2_total_ct", "2_is_attr_ct", "2_mean_attr_10k", "2_is_attr_tot_ratio_10k", "2_mean_pred"))

aggdf = merge(aggdf, aggtrndf, by = "channel", all = T)
View(aggdf)









#################################################


chk_app = 23
keep  = c('ip', 'os', "app", "device", "channel", "click_time", "is_attributed")
trndf = fread(paste0(path, 'train.csv'))
tstdf = fread(paste0(path, 'test.csv'))
trndf = trndf[, keep, with = F]
gc(); gc()

tot_attr = sum(trndf$is_attributed)
aggdf = trndf[,.(.N, sum(is_attributed), as.integer(mean(is_attributed)*10000),  as.integer((sum(is_attributed)/tot_attr)*10000)), by = device]
setnames(aggdf, c("device", "total_ct", "is_attr_ct", "mean_attr_10k", "is_attr_tot_ratio_10k"))
aggtstdf = tstdf[,.(.N), by = device]
setnames(aggtstdf, c("device", "total_tst_ct"))
aggdf = merge(aggdf, aggtstdf, by = "device", all.x = T, all.y = T)
aggdf[, tst_ct_ratio_10k := as.integer(10000*(total_tst_ct/total_ct))]
View(aggdf)


tot_attr = sum(trndf[device==1]$is_attributed)
aggdf = trndf[device==1,.(.N, sum(is_attributed), as.integer(mean(is_attributed)*10000),  as.integer((sum(is_attributed)/tot_attr)*10000)), by = os]
setnames(aggdf, c("os", "total_ct", "is_attr_ct", "mean_attr_10k", "is_attr_tot_ratio_10k"))
aggtstdf = tstdf[device==1,.(.N), by = os]
setnames(aggtstdf, c("os", "total_tst_ct"))
aggdf = merge(aggdf, aggtstdf, by = "os", all.x = T, all.y = T)
aggdf[, tst_ct_ratio_10k := as.integer(10000*(total_tst_ct/total_ct))]
View(aggdf)


tot_attr = sum(trndf$is_attributed)
aggdf = trndf[,.(.N, sum(is_attributed), as.integer(mean(is_attributed)*10000),  as.integer((sum(is_attributed)/tot_attr)*10000)), by = device]
setnames(aggdf, c("device", "total_ct", "is_attr_ct", "mean_attr_10k", "is_attr_tot_ratio_10k"))
aggtstdf = tstdf[,.(.N), by = device]
setnames(aggtstdf, c("device", "total_tst_ct"))
aggdf = merge(aggdf, aggtstdf, by = "device", all.x = T, all.y = T)
aggdf[, tst_ct_ratio_10k := as.integer(10000*(total_tst_ct/total_ct))]
View(aggdf)


trndf[os %in% c(39), .(.N, mean(is_attributed), sum(is_attributed)), by = device]
tstdf[os %in% c(39), .(.N), by = device]

nrow(trndf[device==3032])


#####################################

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
hist(subl[(tstdf$device==1)&(tstdf$os %in% c(39, 56, 70))]$V2)

table(subl[(tstdf$device==1)&(tstdf$os %in% c(39, 56, 70))]$V2>0.01)
table(subl$V2>0.01)

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


