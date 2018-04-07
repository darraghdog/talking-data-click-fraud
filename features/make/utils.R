library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)

writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}


getPeriodCt = function(df, cols_, count_period, intervals){
  df = df[,c(cols_, "split_sec"), with = F]
  df[, index := 1:nrow(df)]
  setorderv(df, c(cols_, "split_sec"))
  df[,next_ten := 0]
  df[,seq_lead := .N:1, by = cols_ ]
  for (shift_n in intervals){
    print(shift_n)
    df[,click_sec_shift_lead := shift(split_sec, shift_n, type = "lead")]
    df[(seq_lead>shift_n) & ((click_sec_shift_lead - split_sec) < (count_period*10000)), next_ten := shift_n]
    gc() 
  }
  setorderv(df, "index")
  setnames(df, "next_ten", new_name)
  df = df[,new_name,with=F]
  return(df)
}


getSplitLead2 = function(df, cols_, fname, path, shift_n = 1){
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
  new_name = paste0("click_sec_lead_split_sec_", fname)
  setnames(df, "click_sec_lead", new_name)
  df = df[,new_name,with=F]
  return(df)
}

#response encoder for categorical features, with credibility adjustment and leave-one-out
calc_exp2 <- function(dt, ft, vn_y, by, tgt_vn, k, mean_y0=NULL, verbose=F) {
  dt[, tmp_y := dt[[vn_y]]]
  tmp <- dt[ft, list(.N, sum(tmp_y), mean(tmp_y)), by=by]
  if(verbose) print(paste("dim of summary :", dim(tmp)))
  setnames(tmp, c(by, 'tmp_cnt', 'tmp_sumy', 'tmp_mean_y'))
  if(is.null(mean_y0)) mean_y0 <- mean(tmp$tmp_mean_y)
  if(verbose) print(paste("mean_y0 = ", mean_y0))
  tmp[, tmp_mean_y := NULL]
  left_merge_inplace(dt, tmp, by=by, verbose=verbose)
  dt[is.na(tmp_cnt), tmp_cnt := 0]
  dt[is.na(tmp_sumy), tmp_sumy := 0]
  dt[ft, tmp_cnt := tmp_cnt - 1L]
  dt[ft, tmp_sumy := tmp_sumy - tmp_y]
  dt[, paste(tgt_vn, sep='') := (tmp_sumy + mean_y0 * k) / (tmp_cnt + k)]
  dt[, tmp_y := NULL]
  dt[, tmp_sumy := NULL]
  dt[, tmp_cnt := NULL]
  return(0)
}


#merge a small dataset to a large data.table without copying the large data.table
left_merge_inplace <- function(dt1, dt2, by, verbose=F, fill.na=NA) {
  st <- proc.time()
  dt1a <- copy(dt1[, by, with=F])
  dt2a <- copy(dt2[, by, with=F])
  if (verbose) {
    print('small datasets created')
    print(proc.time() - st)
  }
  dt1a[, tmp_idx1 := c(1:dim(dt1a)[1])]
  dt2a[, tmp_idx2 := c(1:dim(dt2a)[1])]
  if (verbose) {
    print('row index created')
    print(proc.time() - st)
  }
  dt3 <- merge(dt1a, dt2a, by=by, all.x=T)
  if (verbose) {
    print('small datasets merged')
    print(proc.time() - st)
  }
  dt3 <- dt3[order(tmp_idx1), ]
  if (verbose) {
    print('merged dataset reordered')
    print(proc.time() - st)
  }
  dt2_idx_map <- dt3$tmp_idx2
  if (verbose) {
    print('row index generated')
    print(proc.time() - st)
  }
  
  for(vn in names(dt2)) {
    if(!(vn %in% by)) {
      dt1[, paste(vn, sep='') := dt2[[vn]][dt2_idx_map]]
      if(!is.na(fill.na)) dt1[is.na(dt1[[vn]]), paste(vn, sep=''):=fill.na]
      if (verbose) {
        print(paste('assigned variable ', vn))
        print(proc.time() - st)
      }
    }
  }
}
