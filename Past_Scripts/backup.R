
# Backup ------------------------------------------------------------------
### Setting tune wrappers
ctrl <- makeTuneControlRandom(maxit = 100L)
inner <- cv <- makeResampleDesc("CV", iters = 3L, predict = "both")
outer <- cv <- makeResampleDesc("CV", iters = 5L, predict = "both")
meas <- list(rmse, mae) 

# Random Forest

ps <- makeParamSet(
    makeIntegerParam("ntree",lower = 50, upper = 500),
    makeIntegerParam("mtry", lower = 3, upper = 10),
    makeIntegerParam("nodesize", lower = 10, upper = 50)
)

rf.leaner <- makeTuneWrapper(lrns[["regr.randomForest"]], resampling = inner, 
                             par.set = ps, control = ctrl, measures = meas)

# SVM
ps <- makeParamSet(
    makeDiscreteParam("C", values = 2^c(-8,-4,-2,0)), #cost parameters
    makeDiscreteParam("sigma", values = 2^c(-8,-4,0,4)) #RBF Kernel Parameter
)

svm.leaner <- makeTuneWrapper(lrns[["regr.ksvm"]], resampling = inner, 
                              par.set = ps, control = ctrl, measures = meas)

# Xgboost
ps <- makeParamSet(
    makeDiscreteParam("booster", values = c("gbtree", "gblinear")),
    makeNumericParam("eta", lower = 0.01, upper = 0.1),
    makeIntegerParam("nrounds", lower = 50, upper = 200),
    makeIntegerParam("gamma", lower = 3, upper = 10),
    makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),
    makeIntegerParam("max_depth", lower = 5, upper = 8),
    makeIntegerParam("min_child_weight", lower = 1, upper = 10)
)

xgb.leaner <- makeTuneWrapper(lrns[["regr.xgboost"]], resampling = inner, 
                              par.set = ps, control = ctrl, measures = meas)

lrns_tot <- c(lrns, list(xgb.leaner = xgb.leaner, svm.leaner = svm.leaner, rf.leaner = rf.leaner))


