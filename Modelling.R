################################
# Modelling with Base Learners #
################################

# Setup -------------------------------------------------------------------
options(java.parameters = "-Xmx10g")
source("~/Santander_Value_Prediction_Proyect//Helper_Santander.R")
source("Santander_Value_Prediction_Proyect/Backup/regr.lightgbm_MLR_implementation.R")
library(lightgbm)
santander_libs()

# Getting data ------------------------------------------------------------
df <- readRDS("Santander_Value_Prediction_Proyect/Data/Data_Clustered.RDS")
df_train <- df$train_cluster
test <- df$test_cluster

set.seed(55)
val <- df_train %>% sample_n(size = 1000)
train <- df_train %>% filter(!id %in% val$id)

# Creating Modelling Tasks ------------------------------------------------
train_task <- train %>% select(-id) %>% as.data.frame %>% makeRegrTask(data = ., target = "target")
val_task <- val %>% select(-id) %>% as.data.frame %>% makeRegrTask(data = ., target = "target")

# Benchmark Modelling -----------------------------------------------------
cv <- makeResampleDesc("CV", iters = 3L)

# Compare models
cat("\nComparando Modelos...")
lrns_name <- c("regr.featureless", # base model
               "regr.lightgbm",
               "regr.glmnet", # GLM with Lasso Regularization
               "regr.earth", # Multivariate Adaptive Regression Splines
               "regr.cubist",
               "regr.ksvm",
               "regr.ranger",
               "regr.xgboost") # eXtreme Gradient Boosting
lrns <-  lapply(lrns_name, function(t) makeLearner(t))
names(lrns) <- lrns_name
lrns[["regr.xgboost"]]$par.vals <- list(booster = "gbtree", objective = "reg:linear", lambda = 0.8, alpha = 0, verbose = FALSE,
                                        eta=0.08, max_depth=8, gamma = 5, nrounds = 150, early_stopping_rounds = 50)
lrns[["regr.lightgbm"]]$par.vals <- list(boosting = "gbdt", objective = "regression", metric = "rmse", max_bin = 200,lambda_l2 = 1,
                                         learning_rate=0.01, max_depth=7, min_data_in_leaf=100, num_leaves = 2^8, lambda_l1 = 0, nrounds = 500)
lrns[["regr.ranger"]]$par.vals <- list(num.trees = 100, min.node.size = 100)
lrns[["regr.ksvm"]]$par.vals <- list(kernel = "rbfdot")

# Bagging
earth.lrn <- makeBaggingWrapper(lrns[["regr.earth"]], bw.iters = 50, bw.replace = TRUE, bw.size = 0.6, bw.feats = 3/4)
bag.lrn <- list(earth.bagged = earth.lrn)
lrns_tot <- c(lrns, bag.lrn)

# Stacking
to_stack <- lrns_tot[names(lrns_tot) %in% c("earth.bagged", "regr.xgboost", "regr.lightgbm", "regr.ranger")]
mean.stack <- makeStackedLearner(base.learners = to_stack,  predict.type = "response", method = "average")
light.stack <- makeStackedLearner(base.learners = to_stack, super.learner = "regr.lightgbm", predict.type = "response", use.feat = TRUE)
hill.stack <- makeStackedLearner(base.learners = to_stack, predict.type = "response", method = "hill.climb")
stack.lrns <- list(mean.stack = mean.stack, hill.stack = hill.stack, light.stack = light.stack)
stack.lrns <- map2(stack.lrns, names(stack.lrns), function(a,b) {a$id = b;a})

# ### Benchmark
lrns_bagged_stack <- c(lrns, bag.lrn, stack.lrns)
# set.seed(34)
bmr_base <- benchmark(lrns_bagged_stack, train_task, resamplings = cv, measures = RMSLE_kaggle)

# Summary results training and testing set
to_pred <- lrns_bagged_stack[names(lrns_bagged_stack) %in% c("regr.lightgbm", "regr.xgboost", "regr.ranger", "regr.ksvm", "regr.cubist",
                                                    "mean.stack", "regr.earth.bagged", "hill.stack", "light.stack")]
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
test <- test[, cols_selected]
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
df_submission <- data.frame(ID = id, target = exp10p(preds$mean.stack$data$response))
fwrite(df_submission, file = "Santander_Value_Prediction_Proyect//Data/Mean_Stack_clustered.csv")

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
params <- list(booster = "gbtree", objective = "reg:linear", lambda = 0.8, num_leaves = 2^8, alpha = 0,
               eta=0.08, max_depth=8, gamma = 5, min_data_in_leaf = 500)
xgbcv <- xgb.cv(params = params, data = dtrain, nrounds = 1500, nfold = 3,
                early_stopping_rounds = 10)

dtrain <- lgb.Dataset(data = as.matrix(to_train), label = Y_tr)
dtest <- lgb.Dataset(data = as.matrix(to_test), label = Y_true)

evalerror <- function(preds, dtrain) {
    labels <- lightgbm::getinfo(dtrain, "label")
    err <- MLmetrics::RMSLE(y_pred = exp10p(preds), y_true = exp10p(labels))
    return(list(name = "RMSLE", value = err, higher_better = FALSE))
}

params <- list(boosting = "gbdt", objective = "regression", metric = "rmse", max_bin = 200,lambda_l2 = 1,
               learning_rate=0.01, max_depth=7, min_data_in_leaf=100, num_leaves = 2^8, lambda_l1 = 0)
lgbcv <- lgb.cv(params = params, data = dtrain, nrounds = 3000, nfold = 3,
                early_stopping_rounds = 20, nthread = 5, verbose = 1, eval = evalerror)

