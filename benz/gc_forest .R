library(data.table)
library(Matrix)
library(xgboost)
library(Laurae)
train = fread("/Users/xiaofeifei/I/Kaggle/Benz/train.csv")
test = fread("/Users/xiaofeifei/I/Kaggle/Benz/test.csv")
y = train$y

train <- lapply(train, as.numeric)
train = as.data.table(train)

test <- lapply(test, as.numeric)
test = as.data.table(test)


folds <- Laurae::kfold(y, 5)
train[,y:=NULL]
model <- CRTreeForest(training_data = train, # Training data
                       validation_data = NULL, # Validation data
                       training_labels = y, # Training labels
                       validation_labels = NULL, # Validation labels
                        nthread = 2,
                      folds = folds,
                      lr = 0.02,
                      n_forest = 20,
                      n_trees = 1000,
                      random_forest = 0,
                      seed = 0,
                      objective = "reg:linear",
                      eval_metric = Laurae::df_rmse
)

pred = CRTreeForest_pred(model, test)

Id = test$ID
pred = cbind(Id,pred)

init_mod <- lm(y ~ 1, data = train)
biggest <- formula(lm(y~(X0 + X1 + X2 + X3 + X4 + X5 + X6  + X8)^2,train))
step(init_mod, scope = biggest, direction = 'forward')

colnames(pred) = c("ID", "y")
write.csv(pred, file = "pred.csv", row.names = F)


