#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)

path = '~/tdata/data/'
path = '/Users/dhanley2/Documents/tdata/data/'
source(paste0(path, '../features/make/utils.R'))

train_path <- paste0(path, "train.csv")
test_path  <- paste0(path, "testfull.csv")

tr_col_names <- c("ip", "app", "device", "os", "channel", "click_time", "attributed_time", 
                  "is_attributed")
#truncated hours: 12pm-14pm, 17pm-19pm, 21pm-23pm (UTC + 8)
#including actual hours
most_freq_hours_in_test_data  <- c(4,5,9,10,13,14)
least_freq_hours_in_test_data <- c(6,11,15)
dev_group1 <- c(1,2)
dev_group2 <- c(0,3,5,6,40,59)


# Write out the <ip, device, os> level
keepcols = c('ip', 'os', 'device', 'app', 'channel','click_time')
trndf = fread(paste0(path, 'train.csv'))
trndf = trndf[, keepcols, with=F]
train_rows = nrow(trndf)
tstdf = fread(paste0(path, 'testfull.csv'))
tstdf = tstdf[, keepcols, with=F]
gc(); gc()

# Make the full training data
alldf = rbind(trndf, tstdf)
rm(tstdf, trndf)
gc();gc()

alldf[ , click_time := fasttime::fastPOSIXct(click_time)]
alldf[ , click_day  := as.numeric(format(click_time, "%d"))]
alldf[ , hour       := as.numeric(format(click_time, "%H"))]
alldf[ , ctr_in_test_hh   := 3]
alldf[ hour %in% least_freq_hours_in_test_data, ctr_in_test_hh   := 2]
alldf[ hour %in% most_freq_hours_in_test_data , ctr_in_test_hh   := 1]
alldf[ , ctr_in_test_dev   := 3]
alldf[ hour %in% dev_group2, ctr_in_test_dev   := 2]
alldf[ hour %in% dev_group1, ctr_in_test_dev   := 1]
alldf[ , ctr_hh_dev:=.N , .(ip, hour, device)]
alldf = alldf[ , .(ctr_in_test_hh, ctr_in_test_dev, ctr_hh_dev)]
gc(); gc()

# Write out the data
tst_rows = fread(paste0(path, 'testfull.csv'))$dataset
writeme(alldf[1:train_rows], "ctr_test_hours_trn")
writeme(alldf[(1+train_rows):nrow(alldf)][tst_rows==1], "ctr_test_hours_tst")



# |Feature                 |   Gain|  Cover| Frequency|
#   |:-----------------------|------:|------:|---------:|
#   |app                     | 0.6829| 0.2273|    0.2428|
#   |channel                 | 0.1212| 0.2175|    0.2423|
#   |nip_hh_dev              | 0.0679| 0.0469|    0.0392|
#   |nip_test_hours_set1     | 0.0390| 0.0896|    0.0742|
#   |nip_day_test_hours_set2 | 0.0292| 0.0609|    0.0621|
#   |nip_hh_app              | 0.0250| 0.0877|    0.0686|
#   |os                      | 0.0185| 0.1472|    0.1797|
#   |nip_hh_os               | 0.0093| 0.0774|    0.0621|
#   |device                  | 0.0069| 0.0456|    0.0289|