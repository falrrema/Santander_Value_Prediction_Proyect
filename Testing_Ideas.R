#################
# Testing Ideas #
#################
# Data loading ------------------------------------------------------------
options(scipen=999)
source("~/Kaggle_Santander_Value_Prediction_Challenge/Helper_Santander.R")
library(parallel)
library(ranger)
santander_libs()
train <- fread("Kaggle_Santander_Value_Prediction_Challenge/Data/train.csv") %>% as.tibble()
test <- fread("Kaggle_Santander_Value_Prediction_Challenge/Data/test.csv") %>% as.tibble()

# Finding relevant features -----------------------------------------------
train <- preProc_data(train, add_Xcols = TRUE, logTransform = TRUE, sparsity = 1.1, ceroVariance = FALSE, removeDuplicates = FALSE)
test <- preProc_data(test, add_Xcols = TRUE, logTransform = TRUE, sparsity = 1.1, ceroVariance = FALSE, removeDuplicates = FALSE)

rf <- ranger(target ~ ., data = train[, -1], importance = "permutation")
rf_importance <- data_frame(predictors = names(importance(rf)), importance_value = importance(rf)) %>% 
    arrange(desc(importance_value)) %>% filter(importance_value > 0)
rf_importance <- rf_importance[1:110,]
saveRDS(rf_importance, "Kaggle_Santander_Value_Prediction_Challenge/Data/rf_importance.RDS")

# Row Aggregation ---------------------------------------------------------
train_gather <- train %>%  
    gather(predictor, value, starts_with("x")) %>% 
    group_by(id) %>% 
    filter(value > 0)
test_gather <- test %>%  
    gather(predictor, value, starts_with("x")) %>% 
    group_by(id) %>% 
    filter(value > 0)
train_gather <- as.data.table(train_gather)
test_gather <- as.data.table(test_gather)

cl <- makeCluster(7) # using 7 cores
clusterEvalQ(cl = cl, {library(data.table)})
clusterExport(cl = cl, varlist = list("train_gather"))
train_res <- pbapply::pblapply(unique(train_gather$id), function(t) {
    values <- sum(train_gather[id == t]$value != 0)
    dupl <- sum(duplicated(train_gather[id == t]$value))
    mv_dupl <- mean(train_gather[id == t]$value)
    mv <- mean(unique(train_gather[id == t]$value))
    mdv_dupl <- median(train_gather[id == t]$value)
    mdv <- median(unique(train_gather[id == t]$value))
    min_v <- min(train_gather[id == t]$value)
    max_v <- max(train_gather[id == t]$value)
    
    data.table(id = t, 
               target = unique(train_gather[id == t]$target),
               mean_values_dupl = mv_dupl, 
               mean_values = mv,
               median_values_dupl = mdv_dupl,
               median_values = mdv,
               min_value = min_v,
               max_value = max_v,
               values = values,
               dupl = dupl)
}, cl = cl) %>% bind_rows() 
stopCluster(cl=cl)

cl <- makeCluster(7) # using 7 cores
clusterEvalQ(cl = cl, {library(data.table)})
clusterExport(cl = cl, varlist = list("test_gather"))
test_res <- pbapply::pblapply(unique(test_gather$id), function(t) {
    values <- sum(test_gather[id == t]$value != 0)
    dupl <- sum(duplicated(test_gather[id == t]$value))
    mv_dupl <- mean(test_gather[id == t]$value)
    mv <- mean(unique(test_gather[id == t]$value))
    mdv_dupl <- median(test_gather[id == t]$value)
    mdv <- median(unique(test_gather[id == t]$value))
    min_v <- min(test_gather[id == t]$value)
    max_v <- max(test_gather[id == t]$value)
    
    data.table(id = t, 
               mean_values_dupl = mv_dupl, 
               mean_values = mv,
               median_values_dupl = mdv_dupl,
               median_values = mdv,
               min_value = min_v,
               max_value = max_v,
               values = values,
               dupl = dupl)
}, cl = cl) %>% bind_rows() 
stopCluster(cl=cl)

saveRDS(list(train_res = train_res, test_res = test_res), 
        "Kaggle_Santander_Value_Prediction_Challenge/Data/Data_Row_Agg.RDS")





# Modelling ---------------------------------------------------------------
rf_importance <- readRDS("Kaggle_Santander_Value_Prediction_Challenge/Data/rf_importance.RDS")
train <- fread("Kaggle_Santander_Value_Prediction_Challenge/Data/train.csv") %>% as.tibble()
test <- fread("Kaggle_Santander_Value_Prediction_Challenge/Data/test.csv") %>% as.tibble()
train <- preProc_data(train, add_Xcols = TRUE, logTransform = TRUE, sparsity = 1.1, ceroVariance = FALSE, removeDuplicates = FALSE)
test <- preProc_data(test, add_Xcols = TRUE, logTransform = TRUE, sparsity = 1.1, ceroVariance = FALSE, removeDuplicates = FALSE)
train <- train %>% select(id, target, rf_importance$predictors)
test <- test %>% select(id, rf_importance$predictors)

df <- readRDS("Kaggle_Santander_Value_Prediction_Challenge/Data/Data_Row_Agg.RDS")
train <- train %>% left_join(df$train_res, by = c("id", "target"))
test <- test %>% left_join(df$test_res, by = c("id"))

adversarial_val <- readRDS("Kaggle_Santander_Value_Prediction_Challenge/Data/Adversarial_Training_Validation.RDS")
train_id <- adversarial_val$train_id
val_id <- adversarial_val$val_id
tr <- train %>% filter(id %in% train_id)
val <- train %>% filter(id %in% val_id)

library(lightgbm)
to_train <- tr
to_test <- val
to_test <- to_test[, names(to_train)]

Y_tr <- to_train$target
train_id <- to_train$id
to_train$target <-  to_train$id <- NULL
Y_true <- to_test$target
val_id <- to_test$id
to_test$target <- to_test$id <- NULL

dtrain <- lgb.Dataset(data = as.matrix(to_train), label = Y_tr)
dtest <- lgb.Dataset(data = as.matrix(to_test), label = Y_true)
valids <- list(train = dtrain, test = dtest)

evalerror <- function(preds, dtrain) {
    labels <- lightgbm::getinfo(dtrain, "label")
    err <- MLmetrics::RMSLE(y_pred = exp10p(preds), y_true = exp10p(labels))
    return(list(name = "RMSLE", value = err, higher_better = FALSE))
}

params <- list(boosting = "gbdt", objective = "regression", learning_rate=0.005, max_depth=4, 
               feature_fraction = 0.5, bagging_fraction = 0.8, bagging_freq = 10, 
               num_leaves = 20, min_data_in_bin = 30, min_gain_to_split = 1, min_data_in_leaf = 50)
lgbcv <- lgb.cv(params = params, data = dtrain, nrounds = 1500, nfold = 3,
                early_stopping_rounds = 20, nthread = 5, verbose = 1, eval = evalerror)

lgb.model <- lgb.train(params = params, data = dtrain, num_threads = 7, 
                       nrounds = 1050, eval_freq = 20, eval = evalerror)

val$pred <- predict(lgb.model, data = as.matrix(to_test))
val <- val %>% select(id, target, pred, starts_with("x"), mean_values_dupl:dupl)
MLmetrics::RMSLE(y_pred = exp10p(val$pred), y_true = exp10p(val$target))


