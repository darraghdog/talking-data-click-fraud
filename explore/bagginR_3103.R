#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)

path = '~/tdata/data/'
path = '/Users/dhanley2/Documents/tdata/data/'

sub1 = fread(paste0(path, '../sub/sub_lgb3003C.csv'))
sub2 = fread(paste0(path, '../sub/sub_lgb3003B.csv'))
sub1
sub2
sub1 = sub1[order(is_attributed)]
sub2 = sub2[order(is_attributed)]
sub1[,bag := (1:nrow(sub1))/nrow(sub1)]
sub2[,bag := (1:nrow(sub2))/nrow(sub2)]
sub1 = sub1[order(click_id)]
sub2 = sub2[order(click_id)]
sub3 = sub1[,.(click_id, is_attributed)]
sub3[, is_attributed := (sub1$bag*0.3) + (0.7*sub2$bag)]
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../sub/', name,'.gz')), 
            row.names = F, quote = F)
}

writeme(sub3, 'sub_lgb3003bag.csv')