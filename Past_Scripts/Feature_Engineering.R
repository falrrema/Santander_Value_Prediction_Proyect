#######################
# Feature Engineering #
#######################
set.seed(50)

# Data loading ------------------------------------------------------------
options(java.parameters = "-Xmx10g")
source("~/Kaggle_Santander_Value_Prediction_Challenge/Helper_Santander.R")
santander_libs()
train <- fread("Kaggle_Santander_Value_Prediction_Challenge/Data/train.csv") %>% as.tibble()
test <- fread("Kaggle_Santander_Value_Prediction_Challenge/Data/test.csv") %>% as.tibble()

# Preprocessing -----------------------------------------------------------
train <- preProc_data(train, add_Xcols = TRUE, logTransform = TRUE, sparsity = 1, ceroVariance = TRUE, removeDuplicates = TRUE)
test <- preProc_data(test, logTransform = TRUE, sparsity = 0, ceroVariance = FALSE, removeDuplicates = FALSE)

# Creating row agregates features -------------------------------------------
# Considering all columns except empty ones eliminated above
train_agg <- row_aggregates(train)
test_agg <- row_aggregates(test)

# Further sparsity reduction ----------------------------------------------
# Aggregation features done, continue column prunning
train <- preProc_data(train, add_Xcols = FALSE, logTransform = FALSE, sparsity = 0.98, ceroVariance = FALSE, removeDuplicates = FALSE)
col_preserve <- colnames(train)[!colnames(train) %in% "target"]
test <- test %>% select(col_preserve) # replicate columns preserve in train

# Dimensionality Reduction Features ---------------------------------------
# PCA
set.seed(50)
PCA_res <- principal_component_fit(train, nComp = 500, center = TRUE, scale = TRUE) # get 80% of variance cols
test_pca <- principal_component_transform(test_df = test, pca_object = PCA_res$pc_irlba)

# tSVD
tSVD_res <- tSVD_fit(train, n_vectors = 300)
test_svd <- tSVD_transform(test_df = test, SVD_object = tSVD_res$tSVD_res)

# Target Row Aggregation --------------------------------------------------
train_complete <- train %>% select(id, target) %>% 
    left_join(train_agg, by = "id") %>% 
    left_join(PCA_res$train_pc, by = "id") %>% 
    left_join(tSVD_res$train_svd, by = "id") %>% 
    mutate_at(vars(sum_zeros, sum_values), function(t) (t - mean(t))/sd(t))

test_complete <- test %>% select(id) %>% 
    left_join(test_agg, by = "id") %>% 
    left_join(test_pca, by = "id") %>% 
    left_join(test_svd, by = "id") %>% 
    mutate_at(vars(sum_zeros, sum_values), function(t) (t - mean(t))/sd(t))

# Saving Data -------------------------------------------------------------
fwrite(train_complete, "Kaggle_Santander_Value_Prediction_Challenge/Data/train_processed.csv")
fwrite(test_complete, "Kaggle_Santander_Value_Prediction_Challenge/Data/test_processed.csv")

