library(data.table)
library(Matrix)
library(xgboost)
train = fread("/Users/xiaofeifei/I/Kaggle/Benz/train_start.csv")
test = fread("/Users/xiaofeifei/I/Kaggle/Benz/test_start.csv")


train[,X02:= sum(X0,X2), by = ID]

train <- lapply(train, as.numeric)
train = as.data.table(train)

test <- lapply(test, as.numeric)
test = as.data.table(test)

folds <- Laurae::kfold(y, 5)

model <- CascadeForest(training_data = train, # Training data
                       validation_data = NULL, # Validation data
                       training_labels = y, # Training labels
                       validation_labels = NULL, # Validation labels
                       folds = folds, # Folds for cross-validation
                       boosting = F, # Do not touch this unless you are expert
                       nthread = 6, # Change this to use more threads
                       cascade_lr = 1, # Do not touch this unless you are expert
                       training_start = NULL, # Do not touch this unless you are expert
                       validation_start = NULL, # Do not touch this unless you are expert
                       cascade_forests = c(rep(2, 2), 0), # Number of forest models
                       cascade_trees = 100, # Number of trees per forest
                       cascade_rf = 4, # Number of Random Forest in models
                       cascade_seeds = 1:5, # Seed per layer
                       objective = "reg:linear",
                       eval_metric = Laurae::df_rmse,
                       multi_class = FALSE, # Modify this for multiclass problems
                       early_stopping = 2, # stop after 2 bad combos of forests
                       maximize = FALSE, # not a maximization task
                       verbose = TRUE, # print information during training
                       low_memory = TRUE)

pred = CascadeForest_pred(model, test)

Id = test$ID
pred = cbind(Id,pred)

colnames(pred) = c("ID", "y")
write.csv(pred, file = "pred.csv", row.names = F)
