#########################
# Tunning Base Learners #
#########################

# Setup -------------------------------------------------------------------
options(java.parameters = "-Xmx10g")
source("~/Kaggle_Santander_Value_Prediction_Challenge/Helper_Santander.R")
santander_libs()

proc_task <- readRDS("Kaggle_Santander_Value_Prediction_Challenge/Data/APSF_top_Tasks.RDS")
train_task_fil <- proc_task$train_task_fil
val_task_fil <- proc_task$val_task_fil

# Tunning -----------------------------------------------------------------
library(mlrHyperopt)

# KSVM
tune_ksvm <- hyperopt(train_task_fil, "regr.ksvm")

# Xgboost
lrn_xgboost <- makeLearner("regr.xgboost", par.vals = list(booster = "gbtree", objective = "reg:linear",
                                                           eta=0.05, gamma=5, max_depth=4, nrounds = 110))
tune_xgboost <- hyperopt(train_task_fil, lrn_xgboost)
