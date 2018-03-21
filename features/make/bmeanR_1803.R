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
keepcols = c("ip", "os", "device", "click_time", "is_attributed")
trndf = fread(paste0(path, 'train.csv'))
trndf = trndf[,keepcols,with=F]
trndf[,is_train:=1]
gc(); gc()
tstdf = fread(paste0(path, 'old/test.csv'))
tstdf = tstdf[,keepcols[1:4],with=F]
tstdf[, is_train:=0]
tstdf[, is_attributed := NA]
gc(); gc()
alldf = rbind(trndf, tstdf)
rm(trndf, tstdf)
gc(); gc()

# Make the full training data
alldf[, click_time := fasttime::fastPOSIXct(click_time)]
alldf[, click_day   := as.numeric(format(click_time, "%w"))]
alldf
alldf[, click_time := NULL ]
gc();gc()

alldf[,.N, by= click_day]
alldf[,mean(is_attributed, na.rm = T), by = click_day]

# Function for out of fold Bayes mean
oofBMean = function(df, fold, cols_, allrows_ = FALSE){
  glob_mean = mean(df[(click_day != fold) & (is_train == 1)]$is_attributed)
  glob_ct   = nrow(df[(click_day != fold) & (is_train == 1)])
  prior = 200
  # Aggregate the infold data
  indf = df[(click_day != fold) & (is_train == 1),.(.N, mean(is_attributed)), by = c(cols_)]
  if (allrows_){
    outdf = unique(df[(is_train == 0), c(cols_), with=F])
  }else{
    outdf = unique(df[(click_day == fold) & (is_train == 1), c(cols_, "click_day"), with=F])
  }
  setnames(indf, c(cols_, "count", "mean"))
  indf[, bmean := ((mean*count)+(glob_mean*prior))/(count+prior)]
  indf[,mean := NULL]
  indf[,count := NULL]
  outdf = merge(outdf, indf, by = cols_, how = "left")
  outdf[ is.na(bmean), bmean:= glob_mean]
  return(outdf)
}

# Get bayes mean ip using days as folds
cols_ = "ip"

# Get global mean and count
days_ = unique(alldf[(is_train == 1)]$click_day)
outdfls = list()
for (fold in days_){
  print(paste0("Calculating day ...", fold))
  outdfls[[length(outdfls)+1]] = oofBMean(alldf, fold, "ip" )
}

train_bmean = do.call(rbind, outdfls)
test_bmean  = oofBMean(alldf, 99, "ip", allrows_ = TRUE)

# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(train_bmean, 'bmeantrn_ip')
writeme(test_bmean , 'bmeantst_ip')
hist(train_bmean$bmean, breaks = 1000, xlim = c(0, 0.05))
hist(test_bmean$bmean, breaks = 1000, xlim = c(0, 0.05))
