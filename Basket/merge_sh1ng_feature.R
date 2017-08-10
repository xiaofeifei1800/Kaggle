library(data.table)
library(lightgbm)
library(dplyr)

train = fread("/Users/xiaofeifei/I/Kaggle/Basket/online_train.csv")
train = train[,-1]
colnames(train) = as.character(train[1,])
train = train[-1,]
cols <- colnames(train)[9:85]
train[,(cols):= lapply(.SD, as.numeric), .SDcols = cols]
train[,(cols) := round(.SD,3), .SDcols=cols]

cols <- colnames(train)[1:7]
train[,(cols):= lapply(.SD, as.numeric), .SDcols = cols]
train = train[,-c("reordered","product_name")]

fwrite(train, file = "/Users/xiaofeifei/I/Kaggle/Basket/online_train_c.csv", row.names = F)

test = fread("/Users/xiaofeifei/I/Kaggle/Basket/online_test.csv")
test = test[,-1]
colnames(test) = as.character(test[1,])
test = test[-1,]
cols <- colnames(test)[9:84]
test[,(cols):= lapply(.SD, as.numeric), .SDcols = cols]
test[,(cols) := round(.SD,3), .SDcols=cols]

cols <- colnames(test)[1:7]
test[,(cols):= lapply(.SD, as.numeric), .SDcols = cols]
test = test[,-c("reordered","product_name")]

fwrite(test, file = "/Users/xiaofeifei/I/Kaggle/Basket/online_test_c.csv", row.names = F)

lable = fread("/Users/xiaofeifei/I/Kaggle/Basket/online_label.csv")
lable = lable[-1]

train = fread("/Users/xiaofeifei/I/Kaggle/Basket/online_train_c.csv")
colnames(train) = as.character(train[1,])
train = train[-1,]
feature = c(
  'user_product_reordered_ratio', 'reordered_sum',
  'add_to_cart_order_inverted_mean', 'add_to_cart_order_relative_mean',
  'reorder_prob',
  'dep_reordered_ratio', 'aisle_reordered_ratio',
  'aisle_products',
  'aisle_reordered',
  'dep_products',
  'dep_reordered',
  'prod_users_unq', 'prod_users_unq_reordered',
  'order_number', 'prod_add_to_card_mean',
  'days_since_prior_order',
  'order_dow', 'order_hour_of_day',
  'reorder_ration',
  'user_orders', 'user_order_starts_at', 'user_mean_days_since_prior',
  'user_average_basket', 'user_distinct_products', 'user_reorder_ratio', 'user_total_products',
  'prod_orders', 'prod_reorders',
  'up_order_rate', 'up_orders_since_last_order', 'up_order_rate_since_first_order',
  'up_orders', 'up_first_order', 'up_last_order', 'up_mean_cart_position',
  'days_since_prior_order_mean',
  'order_dow_mean',
  "user_id",
  "product_id","aisle_id","department_id"
  )

train = train[,feature, with = FALSE]
train$reordered = lable$V2

cols <- colnames(train)
train[,(cols):= lapply(.SD, as.numeric), .SDcols = cols]

set.seed(112)
user_id = unique(train$user_id)
valid_set = sample(length(user_id), 0.1*length(user_id))
valid_set = user_id[valid_set]
train_set = user_id[!user_id %in% valid_set]

train_set = train[train$user_id %in% train_set,]
valid_set = train[train$user_id %in% valid_set,]

train_set$user_id <- NULL
valid_set$user_id <- NULL

train_set$eval_set <- NULL
valid_set$eval_set <- NULL

dtrain <- lgb.Dataset(as.matrix(train_set %>% select(-reordered)), label = train_set$reordered,
                      categorical_feature = c('product_id', 'aisle_id', 'department_id'))
dvalid <- lgb.Dataset(as.matrix(valid_set %>% select(-reordered)), label = valid_set$reordered,
                      categorical_feature = c('product_id', 'aisle_id', 'department_id'))

valids <- list(test=dvalid,train=dtrain)

params <- list(task = 'train',
               boosting_type = "gbdt",
               objective="binary", 
               metric="binary_logloss",
               num_leaves = 256,
               min_sum_hessian_in_leaf = 20,
               max_depth = 12,
               learning_rate = 0.05,
               feature_fraction = 0.6,
               verbose = 1,
               num_threads = 6
               )


model = lgb.train(params,dtrain, nrounds = 380, verbose = 1,early_stopping_rounds = 10,
                  valids=valids)
# [380]:	train's binary_logloss:0.239096	test's binary_logloss:0.243223 

#
dtest <- lgb.Dataset(as.matrix(test %>% select(-order_id)), categorical_feature = 
  c('product_id', 'aisle_id', 'department_id'))

pred = predict(model, as.matrix(test %>% select(-order_id)))
                      
test$reordered = pred

fwrite(test[,c("order_id","product_id","reordered")], file = "/Users/xiaofeifei/I/Kaggle/Basket/result2.csv", row.names = F)






