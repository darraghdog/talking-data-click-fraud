#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
gc();gc();gc()
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)

getSplitLead = function(df, cols_, new_name, shift_n = 1){
  df$click_sec = as.numeric(df$click_time)
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
  setnames(df, "click_sec_lead", new_name)
  df = df[,new_name,with=F]
  return(df)
}

path = '~/tdata/data/'
path = '/Users/dhanley2/Documents/tdata/data/'
source(paste0(path, "../features/make/utils.R"))

# Write out the <ip, device, os, channel> level 
cols_ = c("ip", 'os', "app", "channel")
keep  = c(cols_, "click_time")
trndf = fread(paste0(path, 'train.csv'))
trndf = trndf[, c(keep, "is_attributed"), with = F]
tstdf = fread(paste0(path, 'testfull.csv'))
tstdf = tstdf[, c(keep, "dataset") , with = F]


# combine the files
y = trndf$is_attributed
trndf[,is_attributed:=NULL]
test_rows = tstdf$dataset
tstdf[,dataset:=NULL]
alldf = rbind(trndf, tstdf)
rm(trndf, tstdf);gc();gc()


# Create the lead features
alldf[,click_time := fasttime::fastPOSIXct(click_time)]
alldf[,day  := (round(as.numeric(click_time)/(24*3600)))%%7]
alldf[,hour   := as.numeric(format(click_time, "%H"))]
alldf[,minute  := as.numeric(format(click_time, "%M"))]
featall1  = getSplitLead(alldf, c("ip", "os"), "click_split_lead_ipos")
featall1[click_split_lead_ipos>(2*3600) , click_split_lead_ipos := 9999]
featall2  = getSplitLead(alldf, c("ip", "app"), "click_split_lead_ipapp")
featall2[click_split_lead_ipapp>(1.5*3600) , click_split_lead_ipapp := 9999]
alldf[, lu1 := length(unique(channel)) , by=c("ip") ]
alldf[, lu2 := length(unique(channel)) , by=c("day","hour","minute") ]
alldf[, lu3 := length(unique(app)), by=c("ip") ]
alldf[, N1 := .N , by=c("app","day","hour") ]
alldf[, N2 := .N , by=c("app","day","hour","minute") ]
alldf[, RN1N2 := N2/N1 ]


## check the feature
#set.seed(10)
#samp = sample(nrow(featall1[1:nrow(alldf[is_attributed<2])]), 5000000)
#table(cut2(featall1[samp][[1]], g = 20), alldf$is_attributed[samp])
#table(cut2(featall2[samp][[1]], g = 20), alldf$is_attributed[samp])

# Write files
featall1[[1]] = as.integer(featall1[[1]]*100000)
featall2[[1]] = as.integer(featall2[[1]]*100000)
alldf$RN1N2   = as.integer(alldf$RN1N2*100000)
featall = cbind(alldf[,.(lu1, lu2, lu3, RN1N2)], featall1, featall2)

feattrn = featall[1:length(y)]
feattst = featall[(1+length(y)):nrow(featall)][test_rows==1]
writeme(feattrn, "leads_ratios_trn")
writeme(feattst, "leads_ratios_tst")
rm(feattrn, feattst)
gc(); gc(); gc()
gc(); gc(); gc()
