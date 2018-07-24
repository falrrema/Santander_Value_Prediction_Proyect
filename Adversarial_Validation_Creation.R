###################################
# Adversarial validation creation #
###################################

# Data loading ------------------------------------------------------------
source("~/Kaggle_Santander_Value_Prediction_Challenge/Helper_Santander.R")
santander_libs()
train <- fread("Kaggle_Santander_Value_Prediction_Challenge/Data/train.csv") %>% as.tibble()
test <- fread("Kaggle_Santander_Value_Prediction_Challenge/Data/test.csv") %>% as.tibble()

# Adversarial Validation --------------------------------------------------
# Test set differs from training
# To have a better validation set I am construct a randomforest classifier to distinguish train from test samples
# those that are unable to distinguish are going to be use as validation samples
train <- preProc_data(train, add_Xcols = TRUE, logTransform = TRUE, sparsity = 1.1, ceroVariance = FALSE, removeDuplicates = FALSE)
test <- preProc_data(test, add_Xcols = TRUE, logTransform = TRUE, sparsity = 1.1, ceroVariance = FALSE, removeDuplicates = FALSE)
col_preserve <- colnames(train)[!colnames(train) %in% "target"]
test <- test %>% select(col_preserve) # replicate columns preserve in train

set.seed(50)
train_sample <- train %>% 
    mutate(isTest = 0) %>% 
    select(isTest, id, starts_with("x")) %>% 
    sample_n(4000)

set.seed(50)
test_sample <- test %>% 
    mutate(isTest = 1) %>% 
    select(isTest, id, starts_with("x")) %>% 
    sample_n(4000)

df_task <- bind_rows(train_sample, test_sample) %>% 
    mutate(isTest = as.factor(isTest)) %>% 
    dplyr::slice(sample(1:n())) %>% select(-id) %>% as.data.frame() %>% 
    makeClassifTask(data = ., target = "isTest", positive = 1)

# Training a ranger object using MLR
rf_lrn <- makeLearner("classif.ranger", predict.type = "prob", # making a ranger classifier
                      par.vals = list(num.threads = 7, seed = 50, num.trees = 200))
rf_mod <- train(rf_lrn, task = df_task)

# Making predictions in the Training set
preds <- predict(rf_mod, newdata = as.data.frame(train))
train$isTest <- getPredictionProbabilities(preds) 

# Selecting 25% of the most similar rows of training to testing set to make validation set
# Get IDs to separate future validation and training sets
top_rows <- round(nrow(train) * 0.25)
val_id <- train %>% arrange(isTest) %>% slice(1:top_rows) %>% pull(id)
train_id <- train %>% arrange(isTest) %>% slice((top_rows + 1):nrow(train)) %>% pull(id)

# Saving data -------------------------------------------------------------
saveRDS(list(train_id = train_id, val_id = val_id),
        "Kaggle_Santander_Value_Prediction_Challenge/Data/Adversarial_Training_Validation.RDS")

