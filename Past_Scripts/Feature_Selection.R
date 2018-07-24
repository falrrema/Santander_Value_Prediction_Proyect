#######################################
# Feature drift and Feature Selection #
#######################################

# Data loading ------------------------------------------------------------
options(java.parameters = "-Xmx10g")
source("~/Kaggle_Santander_Value_Prediction_Challenge/Helper_Santander.R")
santander_libs()
train <- fread("Kaggle_Santander_Value_Prediction_Challenge/Data/train.csv") %>% as.tibble()
test <- fread("Kaggle_Santander_Value_Prediction_Challenge/Data/test.csv") %>% as.tibble()

# Preprocessing -----------------------------------------------------------
train <- preProc_data(train, add_Xcols = TRUE, logTransform = TRUE, sparsity = 0.98, ceroVariance = TRUE, removeDuplicates = TRUE)
test <- preProc_data(test, logTransform = TRUE, sparsity = 0, ceroVariance = FALSE, removeDuplicates = FALSE)
cols_preserve <- colnames(train)[!colnames(train) %in% "target"]
test <- test %>% select(cols_preserve) # replicate columns preserve in train

# Feature Drift -----------------------------------------------------------
# The idea behind this analysis is that features between train and test differ significantly in their distributions
# Thus they are not suited as predictors in this regression task
# This section will identify those variables and eliminate them from data sets
# Create an equivalent train and testing samples (~4000) with an origin target columns
# set.seed(50)
# train_sample <- train %>% 
#     mutate(isTrain = 1) %>% 
#     select(isTrain, id, starts_with("x")) %>% 
#     sample_n(4000)
# 
# set.seed(50)
# test_sample <- test %>% 
#     mutate(isTrain = 0) %>% 
#     select(isTrain, id, starts_with("x")) %>% 
#     sample_n(4000)
# 
# df_sample <- bind_rows(train_sample, test_sample) %>% 
#     mutate(isTrain = as.factor(isTrain)) %>% 
#     slice(sample(1:n()))
# 
# # RandomForest classifier by column
# # AUC will be calculated as the mean of 3 CV
# # Parallelization using 7 cores for speed
# library(parallel)
# cols <- df_sample %>% select(starts_with("x")) %>% colnames
# rf_lrn <- makeLearner("classif.ranger", predict.type = "prob", # making a ranger classifier
#                       par.vals = list(num.threads = 7, seed = 50, num.trees = 100))
# 
# cl <- makeCluster(7) # using 7 cores
# clusterEvalQ(cl = cl, {library(dplyr); library(mlr)})
# clusterExport(cl = cl, varlist = list("df_sample", "rf_lrn"))
# df_drift <- pbapply::pblapply(cols, function(v) {
#     tr_task <- df_sample %>% select(isTrain, v) %>% as.data.frame() %>% 
#         makeClassifTask(data = ., target = "isTrain", positive = 1)
#     rs <- resample(learner = rf_lrn, task = tr_task, resampling = cv3, measures = auc, show.info = FALSE)
#     data_frame(predictors = v, auc_score = rs$aggr)
# }, cl = cl) %>% bind_rows() %>% arrange(desc(auc_score))
# stopCluster(cl=cl)
# 
# # Eliminate features with AUC scores above 0.6
# cols_preserve <- df_drift %>% filter(auc_score < 0.6) %>% pull(predictors)
# saveRDS(cols_preserve, "Kaggle_Santander_Value_Prediction_Challenge/Data/Cols_preserve_drift.RDS")
cols_preserve <- readRDS("Kaggle_Santander_Value_Prediction_Challenge/Data/Cols_preserve_drift.RDS")
train <- train %>% select(id, target, cols_preserve)
test <- test %>% select(id, cols_preserve)

# Creating row agregates features -------------------------------------------
# Considering all columns except empty ones eliminated above
train_agg <- row_aggregates(train)
test_agg <- row_aggregates(test)

# Dimensionality Reduction Features ---------------------------------------
# PCA
# set.seed(50)
# PCA_res <- principal_component_fit(train, nComp = 500, center = TRUE, scale = TRUE) # get 80% of variance cols
# test_pca <- principal_component_transform(test_df = test, pca_object = PCA_res$pc_irlba)
# 
# # tSVD
# tSVD_res <- tSVD_fit(train, n_vectors = 300)
# test_svd <- tSVD_transform(test_df = test, SVD_object = tSVD_res$tSVD_res)

# Target Row Aggregation --------------------------------------------------
train_complete <- train %>% select(id, target, starts_with("x")) %>% 
    left_join(train_agg, by = "id") %>% 
    # left_join(PCA_res$train_pc, by = "id") %>% 
    # left_join(tSVD_res$train_svd, by = "id") %>% 
    mutate_at(vars(sum_zeros, sum_values), function(t) (t - mean(t))/sd(t))

test_complete <- test %>% select(id, starts_with("x")) %>% 
    left_join(test_agg, by = "id") %>% 
    # left_join(test_pca, by = "id") %>% 
    # left_join(test_svd, by = "id") %>% 
    mutate_at(vars(sum_zeros, sum_values), function(t) (t - mean(t))/sd(t))

# Feature Selection -------------------------------------------------------
# In this section I will eliminate redundant variables generated in the feature engineering process
# Highly Correlated variables will be eliminated
correlationMatrix <- train_complete %>% 
    select(starts_with("x"), starts_with("row"), starts_with("sum")) %>% cor
highlyCorrelated <- caret::findCorrelation(correlationMatrix, cutoff = 0.9, names = TRUE)
train_complete <- train_complete %>% select(-highlyCorrelated)
test_complete <- test_complete %>% select(-highlyCorrelated)

# Variable hunting by var.select method
# In this section I will use Variable hunting method of randomForestSRC package to find top predictors
set.seed(50)
vh_train <- train_complete %>% select(-id) %>% as.data.frame() %>% 
    randomForestSRC::var.select(target ~., ., method = "vh", nrep = 3, ntree = 200, fast = TRUE)
cols_preserve <- vh_train$topvars

# Variable importance by Chi.squared
train_task <- train_complete %>% select(-id) %>% as.data.frame() %>% 
    makeRegrTask(data = ., target = "target")
fs <- generateFilterValuesData(train_task, method = "chi.squared")
cols_preserve2 <- fs$data %>% filter(chi.squared > 0) %>% arrange(desc(chi.squared)) %>% pull(name)
cols_preserve3 <- c(cols_preserve, cols_preserve2) %>% unique

train_feat <- train_complete %>% select(id, target, cols_preserve3) 
test_feat <- test_complete %>% select(id, cols_preserve3) 

# Saving Data -------------------------------------------------------------
fwrite(train_complete, "Kaggle_Santander_Value_Prediction_Challenge/Data/train_processed.csv")
fwrite(test_complete, "Kaggle_Santander_Value_Prediction_Challenge/Data/test_processed.csv")

fwrite(train_feat, "Kaggle_Santander_Value_Prediction_Challenge/Data/train_processed_featsel.csv")
fwrite(test_feat, "Kaggle_Santander_Value_Prediction_Challenge/Data/test_processed_featsel.csv")
