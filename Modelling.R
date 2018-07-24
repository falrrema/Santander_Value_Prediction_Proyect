################################
# Modelling with Base Learners #
################################

# Setup -------------------------------------------------------------------
options(java.parameters = "-Xmx10g")
source("~/Kaggle_Santander_Value_Prediction_Challenge/Helper_Santander.R")
source("Kaggle_Santander_Value_Prediction_Challenge/Backup/regr.lightgbm_MLR_implementation.R")
library(lightgbm)
santander_libs()

# Getting data ------------------------------------------------------------
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

# Creating Modelling Tasks ------------------------------------------------
train_task <- tr %>% select(-id) %>% as.data.frame %>% makeRegrTask(data = ., target = "target")
val_task <- val %>% select(-id) %>% as.data.frame %>% makeRegrTask(data = ., target = "target")

# Benchmark Modelling -----------------------------------------------------
cv <- makeResampleDesc("CV", iters = 3L)

# Compare models
cat("\nComparando Modelos...")
lrns_name <- c("regr.featureless", # base model
               "regr.lightgbm",
               "regr.glmnet", # GLM with Lasso Regularization
               "regr.earth", # Multivariate Adaptive Regression Splines
               "regr.ranger",
               "regr.xgboost") # eXtreme Gradient Boosting
lrns <-  lapply(lrns_name, function(t) makeLearner(t))
names(lrns) <- lrns_name
lrns[["regr.xgboost"]]$par.vals <- list(booster = "gbtree", objective = "reg:linear", alpha = 0,
                                        eta=0.01, max_depth=5, gamma = 8, min_child_weight = 10, nrounds = 700)
lrns[["regr.lightgbm"]]$par.vals <- list(learning_rate=0.01, max_depth=7, feature_fraction = 0.7,min_data_in_leaf=100,
                                         bagging_fraction = 1,bagging_freq = 10, num_leaves = 20, num_iterations = 700)
lrns[["regr.ranger"]]$par.vals <- list(num.trees = 500, min.node.size = 10)

# Bagging
earth.lrn <- makeBaggingWrapper(lrns[["regr.earth"]], bw.iters = 50, bw.replace = TRUE, bw.size = 0.6, bw.feats = 3/4)
bag.lrn <- list(earth.bagged = earth.lrn)
lrns_tot <- c(lrns, bag.lrn)

# Stacking
to_stack <- lrns_tot[names(lrns_tot) %in% c("earth.bagged", "glmnet.bagged", "regr.lightgbm", "regr.ranger")]
mean.stack <- makeStackedLearner(base.learners = to_stack,  predict.type = "response", method = "average")
hill.stack <- makeStackedLearner(base.learners = to_stack, predict.type = "response", method = "hill.climb")
stack.lrns <- list(mean.stack = mean.stack, hill.stack = hill.stack)
stack.lrns <- map2(stack.lrns, names(stack.lrns), function(a,b) {a$id = b;a})

# ### Benchmark
lrns_bagged_stack <- c(lrns, bag.lrn, stack.lrns)
# set.seed(34)
bmr_base <- benchmark(lrns_bagged_stack, train_task, resamplings = cv, measures = RMSLE_kaggle)

# Summary results training and testing set
to_pred <- lrns_bagged_stack[names(lrns_bagged_stack) %in% c("regr.lightgbm", "regr.xgboost", "regr.ranger", 
                                                    "mean.stack", "regr.earth.bagged", "hill.stack")]
models <- map2(to_pred, names(to_pred), function(x, y) {
    cat("\nConstruyendo Modelo:", y)
    train(x, task = train_task)
})
preds_train <- lapply(models, function(t) predict(t, task = train_task))
preds_test <- lapply(models, function(t) predict(t, task = val_task))
perf_train <- lapply(preds_train, function(t) performance(t, RMSLE_kaggle)) # performance en el training set
perf_test <- lapply(preds_test, function(t) performance(t, RMSLE_kaggle)) # performance en el testing set

perf <- bind_rows(perf_train, perf_test) %>% 
    mutate(data = c("train", "test")) %>% 
    select(data, regr.lightgbm:mean.stack) %>% 
    gather(lrns, values, regr.lightgbm:mean.stack) %>% 
    spread(data, values) %>% 
    mutate(diff = train-test) %>% 
    select(lrns, train, test, diff) %>% 
    arrange(test)
perf 

# PostProcessing  ---------------------------------------------------------
cols_selected <- getTaskData(train_task) %>% select(-target) %>% colnames
id <- test$id
test <- test %>% select(cols_selected)
df_train <- bind_rows(getTaskData(train_task), getTaskData(val_task))

to_train_task <- makeRegrTask(data = df_train, target = "target") 
models <- map2(to_pred, names(to_pred), function(x, y) {
    cat("\nConstruyendo Modelo:", y)
    train(x, task = to_train_task)
})

# Prediction on test data -------------------------------------------------
preds <- map2(models, names(models), function(x,y) {
    cat("\nPredicciones:", y)
    
    predict(x, newdata = as.data.frame(test))
})

## Submission
df_submission <- data.frame(ID = id, target = exp10p(preds$regr.ranger$data$response))
fwrite(df_submission, file = "Kaggle_Santander_Value_Prediction_Challenge/Data/ranger.csv")

# Xgboost -----------------------------------------------------------------
# Con el fin de tener los paràmetros tuneados
library(xgboost)
library(lightgbm)
to_train <- getTaskData(train_task)
to_test <- getTaskData(val_task)
to_test <- to_test[, names(to_train)]

Y_tr <- to_train$target
to_train$target <- NULL
Y_true <- to_test$target
to_test$target <- NULL

dtrain <- xgb.DMatrix(data = as.matrix(to_train), label = Y_tr)
params <- list(booster = "gbtree", objective = "reg:linear", alpha = 0,
               eta=0.01, max_depth=5, gamma = 8, min_child_weight = 10)
xgbcv <- xgb.cv(params = params, data = dtrain, nrounds = 1500, nfold = 3,
                early_stopping_rounds = 10)

dtrain <- lgb.Dataset(data = as.matrix(to_train), label = Y_tr)
dtest <- lgb.Dataset(data = as.matrix(to_test), label = Y_true)
valids <- list(train = dtrain, test = dtest)

evalerror <- function(preds, dtrain) {
    labels <- lightgbm::getinfo(dtrain, "label")
    err <- MLmetrics::RMSLE(y_pred = exp10p(preds), y_true = exp10p(labels))
    return(list(name = "RMSLE", value = err, higher_better = FALSE))
}

params <- list(boosting = "gbdt", objective = "regression", metric = "rmse", 
               learning_rate=0.01, max_depth=7, feature_fraction = 0.7,min_data_in_leaf=100,
               bagging_fraction = 1,bagging_freq = 10, num_leaves = 20, valids = valids)
lgbcv <- lgb.cv(params = params, data = dtrain, nrounds = 1500, nfold = 3,
                early_stopping_rounds = 20, nthread = 5, verbose = 1, eval = evalerror)

