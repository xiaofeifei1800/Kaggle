library(data.table)
library(dplyr)
library(tidyr)


# Load Data ---------------------------------------------------------------
data = fread("/Users/xiaofeifei/I/Kaggle/Basket/all_features.csv")

# Train / Test datasets ---------------------------------------------------
train <- as.data.frame(data[data$eval_set == "train",])
train$eval_set <- NULL
train$product_id <- NULL
train$order_id <- NULL
train$product_name = NULL
train$reordered[is.na(train$reordered)] <- 0

test <- as.data.frame(data[data$eval_set == "test",])
test$eval_set <- NULL
test$user_id <- NULL
test$reordered <- NULL
test$product_name = NULL

rm(data)
gc()


# Model -------------------------------------------------------------------
library(xgboost)

params <- list(
  "objective"           = "reg:logistic",
  "eval_metric"         = "logloss",
  "eta"                 = 0.1,
  "max_depth"           = 6,
  "min_child_weight"    = 10,
  "gamma"               = 0.70,
  "subsample"           = 0.77,
  "colsample_bytree"    = 0.95,
  "alpha"               = 2e-05,
  "lambda"              = 10
)

set.seed(112)
user_id = unique(train$user_id)
valid_set = sample(length(user_id), 0.1*length(user_id))
valid_set = user_id[valid_set]
train_set = user_id[!user_id %in% valid_set]

train_set = train[train$user_id %in% train_set,]
valid_set = train[train$user_id %in% valid_set,]

train_set$user_id <- NULL
valid_set$user_id <- NULL

dtrain <- xgb.DMatrix(as.matrix(train_set %>% select(-reordered)), label = train_set$reordered)
dvalid <- xgb.DMatrix(as.matrix(valid_set %>% select(-reordered)), label = valid_set$reordered)

rm(train)

watchlist = list(train=dtrain, test=dvalid)

model <- xgb.train(data = dtrain, params = params, nrounds = 280, nthread = 6, watchlist = watchlist,
                   early_stopping_rounds = 10)

train_set = rbind(train_set, valid_set)
dtrain <- xgb.DMatrix(as.matrix(train_set %>% select(-reordered)), label = train_set$reordered)
model <- xgb.train(data = dtrain, params = params, nrounds = 280, nthread = 6)
importance <- xgb.importance(colnames(train_set), model = model)
xgb.ggplot.importance(importance)

rm(train_set, valid_set)
gc()


# Apply model -------------------------------------------------------------
X <- xgb.DMatrix(as.matrix(test %>% select(-order_id,-product_id)))
test$reordered <- predict(model, X)

fwrite(test[,c("order_id","product_id","reordered")], file = "/Users/xiaofeifei/I/Kaggle/Basket/result4.csv", row.names = F)

test$reordered <- (test$reordered > 0.21) * 1

submission <- test %>%
  filter(reordered == 1) %>%
  group_by(order_id) %>%
  summarise(
    products = paste(product_id, collapse = " ")
  )

missing <- data.frame(
  order_id = unique(test$order_id[!test$order_id %in% submission$order_id]),
  products = "None"
)

submission <- submission %>% bind_rows(missing) %>% arrange(order_id)
fwrite(submission, file = "test_cv.csv", row.names = F)


data = fread("/Users/xiaofeifei/I/Kaggle/Basket/F1_max_online.csv")
data = data[-1,]

colnames(data) = c("products", "order_id")

data = data[,c(2,1)]

fwrite(data, file = "/Users/xiaofeifei/I/Kaggle/Basket/F1_max_online.csv", row.names = F)
