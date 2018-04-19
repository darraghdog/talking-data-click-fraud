#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
gc();gc();gc()
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)


path = '~/tdata/data/'
path = '/Users/dhanley2/Documents/tdata/data/'
source(paste0(path, "../features/make/utils.R"))

# Write out the <ip, device, os, channel> level 
cols_ = c("ip", "device", 'os', "app", "click_time")
keep  = cols_
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

# 
cols_ = c("ip", "device", 'os', "split_sec")
alldf[,click_sec := fasttime::fastPOSIXct(click_time)]
alldf[,click_sec := click_sec - min(alldf$click_sec)]
alldf[, split_sec := round((0:(.N-1))/.N, 4) + click_sec, by = click_sec]
alldf[, index := 1:nrow(alldf)]
alldf[,click_sec := NULL]
alldf[,click_time := NULL]
gc()
# Make and id for the values 
setkeyv(alldf, c("ip", "os", "device") , physical = T)
key_ipdevos = alldf[,1, by = c("ip", "os", "device")]
key_ipdevos[,ipdevos:= 1:.N]
setkeyv(key_ipdevos, c("ip", "os", "device") , physical = T)
alldf = merge(alldf, key_ipdevos, by =  c("ip", "os", "device"), how = 'inner')
alldf

###########################################
# Now lets get the count in the next 20
###########################################
alldf = alldf[,.(ipdevos, index, split_sec, app)]
setkeyv(alldf, c("ipdevos") , physical = T)
alldf[,ct_ipdevos := .N, by = ipdevos]
steps = 20
for (i in 1:steps){
  cn = paste0("appshift", i)
  expr = bquote(.(as.name(cn)):=shift(app, i, type = "lead"))
  alldf[,eval(expr), by = ipdevos]
  gc()
}

# Set the baseline for the count of the app
alldf[,count_same_in_next_seq:= 0]
for (i in 1:steps){
  print(i)
  cn = paste0("appshift", i)
  expr = bquote(count_same_in_next_seq := count_same_in_next_seq + as.integer(app == .(as.name(cn))))
  alldf[,eval(expr)]
  gc()
}
alldf[ct_ipdevos<steps, count_same_in_next_seq:= 99]
# Anything that does not have an hour out of in for we make to 
alldf[, split_sec_shift_steps := shift(as.numeric(split_sec), steps, type = "lead"), by = ipdevos]
alldf[ split_sec_shift_steps  - as.numeric(split_sec) > 3600, count_same_in_next_seq := 99]

###########################################
# Now lets get the count in the next 20
###########################################
alldf[, count_same_in_next_seq20:= count_same_in_next_seq]
alldf = alldf[,.(ipdevos, index, split_sec, app, count_same_in_next_seq20)]
gc();gc()
alldf[,ct_ipdevos := .N, by = ipdevos]
steps = 5
for (i in 1:steps){
  cn = paste0("appshift", i)
  expr = bquote(.(as.name(cn)):=shift(app, i, type = "lead"))
  alldf[,eval(expr), by = ipdevos]
  gc()
}

# Set the baseline for the count of the app
alldf[,count_same_in_next_seq:= 0]
for (i in 1:steps){
  print(i)
  cn = paste0("appshift", i)
  expr = bquote(count_same_in_next_seq := count_same_in_next_seq + as.integer(app == .(as.name(cn))))
  alldf[,eval(expr)]
  gc()
}
alldf[ct_ipdevos<steps, count_same_in_next_seq:= 99]
# Anything that does not have an hour out of in for we make to 
alldf[, split_sec_shift_steps := shift(as.numeric(split_sec), steps, type = "lead"), by = ipdevos]
alldf[ split_sec_shift_steps  - as.numeric(split_sec) > 3600, count_same_in_next_seq := 99]



alldf[, count_same_in_next_seq5:= count_same_in_next_seq]
alldf = alldf[,.(ipdevos, index, split_sec, app, count_same_in_next_seq20, count_same_in_next_seq5)]


# Reset the data table to the right order
samp = sample(1:nrow(alldf), 100000)
table(alldf$count_same_in_next_seq20[samp])
table(alldf$count_same_in_next_seq5[samp])
gc();gc()

# Fill the NA
alldf[is.na(count_same_in_next_seq20), count_same_in_next_seq20 := 99]
alldf[is.na(count_same_in_next_seq5), count_same_in_next_seq5 := 99]

# Reorder to correct sequence
alldf = alldf[order(index)]
samp = sample(1:length(y), 10000000)
table(alldf$count_same_in_next_seq20[samp], y[samp])
table(alldf$count_same_in_next_seq5[samp], y[samp])


# Write files
featall = alldf[,.(count_same_in_next_seq20, count_same_in_next_seq5)]
feattrn = featall[1:length(y)]
feattst = featall[(1+length(y)):nrow(featall)][test_rows==1]
writeme(feattrn, "count_same_in_next_trn")
writeme(feattst, "count_same_in_next_tst")
rm(feattrn, feattst)
gc(); gc(); gc()
gc(); gc(); gc()
