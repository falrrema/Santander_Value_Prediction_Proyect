#################
# Testing Ideas #
#################
# Data loading ------------------------------------------------------------
options(scipen=999)
source("~/Santander_Value_Prediction_Proyect/Helper_Santander.R")
library(parallel)
library(ranger)
santander_libs()
train <- fread("Santander_Value_Prediction_Proyect/Data/train.csv") %>% as.tibble()
test <- fread("Santander_Value_Prediction_Proyect/Data/test.csv") %>% as.tibble()

# Finding relevant features -----------------------------------------------
rf <- ranger(target ~ ., data = train[, -1], importance = "permutation")
rf_importance <- data_frame(predictors = names(importance(rf)), importance_value = importance(rf)) %>% 
    arrange(desc(importance_value)) %>% filter(importance_value > 0)
rf_importance <- rf_importance[1:110,]
saveRDS(rf_importance, "Santander_Value_Prediction_Proyect/Data/rf_importance.RDS")

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
        "Santander_Value_Prediction_Proyect/Data/Data_Row_Agg.RDS")

# Dimensionality Reduction Features ---------------------------------------
# PCA
train <- preProc_data(train, add_Xcols = TRUE, logTransform = TRUE, sparsity = 1, ceroVariance = TRUE, removeDuplicates = TRUE)
test <- preProc_data(test, add_Xcols = TRUE, logTransform = TRUE, sparsity = 1, ceroVariance = TRUE, removeDuplicates = TRUE)
train_cols <- train %>% select(id, starts_with("x")) %>% names
test <- test[, train_cols]

train$target <- NULL
set.seed(50)
PCA_res <- principal_component_fit(train, nComp = 20, center = TRUE, scale = TRUE) # top 20 components
test_pca <- principal_component_transform(test_df = test, pca_object = PCA_res$pc_irlba)

# tSVD
tSVD_res <- tSVD_fit(train, n_vectors = 20)
test_svd <- tSVD_transform(test_df = test, SVD_object = tSVD_res$tSVD_res)

train_dr <- PCA_res$train_pc %>% left_join(tSVD_res$train_svd, by = "id")
test_dr <- test_pca %>% left_join(test_svd, by = "id")

saveRDS(list(train_dr = train_dr, test_dr = test_dr), 
        "Santander_Value_Prediction_Proyect/Data/Data_Dim_Red.RDS")

# Modelling ---------------------------------------------------------------
rf_importance <- readRDS("Santander_Value_Prediction_Proyect/Data/rf_importance.RDS")
train <- fread("Santander_Value_Prediction_Proyect/Data/train.csv") %>% as.tibble()
test <- fread("Santander_Value_Prediction_Proyect/Data/test.csv") %>% as.tibble()
train <- preProc_data(train, add_Xcols = TRUE, logTransform = TRUE, sparsity = 1.1, ceroVariance = FALSE, removeDuplicates = FALSE)
test <- preProc_data(test, add_Xcols = TRUE, logTransform = TRUE, sparsity = 1.1, ceroVariance = FALSE, removeDuplicates = FALSE)
train <- train %>% select(id, target, rf_importance$predictors[1:50])
test <- test %>% select(id, rf_importance$predictors[1:50])

df <- readRDS("Santander_Value_Prediction_Proyect/Data/Data_Row_Agg.RDS")
train <- train %>% left_join(df$train_res, by = c("id", "target"))
test <- test %>% left_join(df$test_res, by = c("id"))

df <- readRDS("Santander_Value_Prediction_Proyect/Data/Data_Dim_Red.RDS")
train <- train %>% left_join(df$train_dr, by = c("id"))
test <- test %>% left_join(df$test_dr, by = c("id"))

# Correlation Analysis ----------------------------------------------------
correlationMatrix <- cor(train[,3:100])
highlyCorrelated <- caret::findCorrelation(correlationMatrix, cutoff =0.8, verbose = TRUE, names = TRUE) # cutoff over 0.8
train <- train %>% select(-highlyCorrelated) # removing highly correlated variables
test <- test %>% select(-highlyCorrelated) # removing highly correlated variables

# Clustering --------------------------------------------------------------
library(dendextend)
library(cluster) 

train_norm <- train %>% select(-target, -id) %>% scale %>%  
    as.data.frame()
rownames(train_norm) <- train$id

# Silhouette Analysis
sil_width <- map_dbl(2:20, function(k) { 
    cat("\nClustering with K = ", k)
    model <- pam(x = train_norm, k = k) 
    model$silinfo$avg.width
})

sil_df <- data.frame(k = 2:20,  sil_width = sil_width)
ggplot(sil_df, aes(x = k, y = sil_width)) + geom_line() +
    scale_x_continuous(breaks = 2:20)

# Get Clusters and predict on test set
cluster_model <- pam(x = train_norm, k = 2, keep.data = TRUE)
plot(silhouette(cluster_model))
cluster_model$silinfo$widths %>% View
train$cluster <- cluster_model$clustering

library(FNN)
pred_knn <- get.knnx(cluster_model$medoids, scale(test[, -1]), 1)
test$cluster <- pred_knn$nn.index

saveRDS(list(train_cluster = train, test_cluster= test),
     "Santander_Value_Prediction_Proyect/Data/Data_Clustered.RDS")

