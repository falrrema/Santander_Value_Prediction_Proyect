##################################
# EDA Santander Kaggle Challenge #
##################################
# Data loading ------------------------------------------------------------
options(scipen=999)
source("~/Kaggle_Santander_Value_Prediction_Challenge/Helper_Santander.R")
library(parallel)
santander_libs()
df <- fread("Kaggle_Santander_Value_Prediction_Challenge/Data/train.csv") %>% as.tibble()

# QuickLook ---------------------------------------------------------------
glimpse(df)
dim(df)
table_na(df) %>% as.tibble # check missingness
names(df) <- paste0("x", tolower(names(df))) # Add X before each columns to format columns correctly

# Column Removal ----------------------------------------------------------
# Measure of sparseness
sparsity <- table_0(df) %>% as.tibble
sparsity %>% count(percent == 1) # 256 columns are completly empty sparsity
sparsity %>% count(percent >= 0.99) # 2364 columns have over 99% sparsity
sparsity %>% count(percent >= 0.98) # 4952 columns have over 98% sparsity
sparsity %>% count(percent <= 0.90) # 41 columns have less than 90% sparsity

sparsity %>% count(count_1 == 1) # 233 columns have just 1 data rows
sparsity %>% count(count_1 == 2) # 140 columns have just 2 data rows
sparsity %>% count(count_1 >= 10) # 3729 columns have over 10 data rows

# Remove total sparse columns 
df <- rm_sparse_terms(df, upper_bound = 1) 

# Remove duplicated columns
df <- as.data.frame(as.list(df)[!duplicated(as.list(df))]) # 5 duplicated columns

# To matrix
id <- df$xid
df$xid <- NULL
dim(df)

# Check zero variance columns
sum(apply(df, 2, var) == 0) # 8 zero variance columns
df <- df[, apply(df, 2, var) != 0] # remove these columns
dim(df)

# LogTransform data -------------------------------------------------------
df <- as_data_frame(df)
df <- sapply(df, log10p) # applies log10(x + 1) trasformation to all columns
glimpse(df)

# Check Outliers ----------------------------------------------------------
df$row_sums <- rowSums(df[,2:ncol(df)])
df$row_means <- rowMeans(df[,2:ncol(df)])
df$row_sd <- apply(df[,2:ncol(df)], 1, sd)

dim(df)
glimpse(df)

# rows empty 
df %>% count(row_sums == 0) # 1 row completly empty
target_0 <- df %>% filter(row_sums == 0) %>% pull(xtarget)

ss <- df %>% filter(xtarget == target_0) # 12 columns share the same target
sp_ss <- table_0(ss)
cols_X0 <- sp_ss %>% filter(count_1 > 1) %>% arrange(percent) %>% pull(column)
ss %>% select(cols_X0) %>% View

# Duplicated target
count_target <- df %>% count(xtarget) %>% arrange(desc(n)) 
count_target # there are many duplicated targets

col_6.30 <- df %>% filter(xtarget == count_target$xtarget[1]) %>% 
    table_0 %>% filter(count_1 > 1) %>% arrange(percent) %>% pull(column)

df_6.30 <- df %>% filter(xtarget == count_target$xtarget[1]) %>% select(col_6.30)
df_6.30 %>% count(row_sums, sort = TRUE) # there is 5 rows that have duplicated rowsums
df_6.30 %>% filter(row_sums %in% c(7.16, 61.2, 82.7, 144, 152)) %>% 
    arrange(total) %>% View

# get the columns of th top 10 duplicated target
col_dupl <- vector(mode = "list", 10)

for (i in 1:10) {
    col_dupl[[i]] <- df %>% filter(xtarget == count_target$xtarget[i]) %>% select(-xtarget, -row_sums, -row_means, -row_sd) %>% 
        table_0 %>% filter(count_1 > 1) %>% arrange(percent) %>% pull(column)
    cat("\nDuplicated Target", count_target$target[i], "has", length(col_dupl[[i]]), "columns")
}

# Which columns intersect?
# Intersected columns aremost likely to be just noise because they don't add any information to target
col_intersect <- Reduce(intersect, col_dupl)

df %>% table_0() %>% filter(column %in% col_intersect) %>% as.data.table
col_cross <- names(df_6.30)[!names(df_6.30) %in% col_intersect]
df_6.30 %>% select(col_cross) %>% mutate(total2 = rowSums(.[3:ncol(.)])) %>% 
    select(xtarget, total, total2, Xdc10234ae:X7e293fbaf) %>% View

# Target distribution -----------------------------------------------------
summary(df$xtarget) # target summary
gghist(df, xtarget) # targets distribution
df_fil1 %>% mutate(target_log = log10p(target)) %>% 
    gghist(target_log, bin = 50) # log10 transformation on target

setDT(df)
df <- df[, lapply(.SD, mean), by=xtarget] # fast aggregarion by target
dim(df)

# Segmentation  -----------------------------------------------------------
df <- preProc_data(df, add_Xcols = TRUE, logTransform = TRUE, sparsity = 1.1, ceroVariance = FALSE, removeDuplicates = FALSE)
cuts <- df %>% pull(target) %>% quantile(probs = c(0.5))

df_gather <- df %>% mutate(value_type = cut(target, breaks = c(0,cuts,10), labels = c(0, 1))) %>% 
    gather(predictor, value, starts_with("x")) %>% filter(value != 0) %>% arrange(id)

value_type_pred <- df_gather %>% group_by(value_type, predictor) %>% 
    summarise(n = n(),
              mean_pred = mean(value),
              mean_target = mean(target),
              median_pred = median(value),
              median_target = median(target)) %>% 
    ungroup %>% group_by(value_type) %>% 
    top_n(30, n)

value_type_pred %>% arrange(value_type, desc(n)) %>% View

cols_intersect <- value_type_pred %>% split(.$value_type) %>% 
    map(~ .x$predictor) %>% 
    reduce(intersect)

sp_preds <- df_gather %>% filter(predictor %in% cols_intersect) %>% 
    group_by(id, target, value_type) %>% 
    summarise(mean_preds = mean(value),
              median_preds = median(value),
              max_preds = max(value),
              min_preds = min(value),
              sd_preds = sd(value)) %>% 
    mutate_at(vars(mean_preds:sd_preds), function(t) coalesce(t, 0))

library(glmnet)
sp <- sample(sp_preds$id, 1000)
train <- sp_preds[!sp_preds$id %in% sp,]
test <- sp_preds[sp_preds$id %in% sp,]
Y_true <- as.numeric(test$value_type) - 1
Y <- as.numeric(train$value_type) - 1 
train$id <- train$target <- train$value_type <- NULL
test$id <- test$target <- test$value_type <- NULL

glmnet_classifier <- cv.glmnet(x = as.matrix(train), y = Y, alpha = 1, nfolds = 10)
plot(glmnet_classifier)

preds <- predict(glmnet_classifier, newx = as.matrix(test))[,1]
MLmetrics::AUC(preds, y_true = Y_true)
MLmetrics::RMSLE(y_pred = exp10p(preds), y_true = exp10p(Y_true))

# Looking into duplicate target as features in train ----------------------
df <- preProc_data(df, add_Xcols = TRUE, logTransform = FALSE, sparsity = 1.1, ceroVariance = FALSE, removeDuplicates = FALSE)
df_gather <- df %>%  
    gather(predictor, value, starts_with("x")) %>% 
    group_by(id) %>% 
    arrange(id, desc(value))
df_gather <- as.data.table(df_gather)

cl <- makeCluster(7) # using 7 cores
clusterEvalQ(cl = cl, {library(data.table)})
clusterExport(cl = cl, varlist = list("df_gather"))
df_res <- pbapply::pblapply(unique(df_gather$id), function(t) {
    zeros <- sum(df_gather[id == t]$value == 0)
    values <- sum(df_gather[id == t]$value != 0)
    dupl <- sum(duplicated(df_gather[id == t & value > 0]$value))
    mv_dupl <- mean(df_gather[id == t & value > 0]$value)
    mv <- mean(unique(df_gather[id == t & value > 0]$value))
    mdv_dupl <- median(df_gather[id == t & value > 0]$value)
    mdv <- median(unique(df_gather[id == t & value > 0]$value))
    min_v <- min(df_gather[id == t & value > 0]$value)
    max_v <- max(df_gather[id == t & value > 0]$value)
    
    data.table(id = t, 
               target = unique(df_gather[id == t]$target),
               mean_values_dupl = mv_dupl, 
               mean_values = mv,
               median_values_dupl = mdv_dupl,
               median_values = mdv,
               min_value = min_v,
               max_value = max_v,
               zeros = zeros, 
               values = values,
               dupl = dupl)
}, cl = cl) %>% bind_rows() 
stopCluster(cl=cl)


# data leakage -----------------------------------------------------------
library(data.table)
df <- as.data.table(df)
df2 <- df[,c("ID","target","f190486d6","58e2e02e6","eeb9cd3aa","9fd594eec","6eef030c1","15ace8c9f","fb0f5dbfe","58e056e12","20aa07010","024c577b9","d6bb78916","b43a7cfd5","58232a6fb"),with=F]
df2 <- df2[ c(2072,3493,379,2972,2367,4415,2791,3980,194,1190,3517,811,4444) ]

df3 <- df2 %>% as.tibble %>% gather(key, value, "target":"58232a6fb") %>% arrange(value)
df3 %>% spread(key, value) %>% View


