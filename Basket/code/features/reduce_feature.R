data = fread("/Users/xiaofeifei/I/Kaggle/Basket/feature.csv")
importance = fread("/Users/xiaofeifei/I/Kaggle/Basket/impotance.csv")

data = as.data.table(data)
data = data[,c("eval_set","order_id","product_id","reordered","user_id","aisle_id",importance$Feature[1:50]), with = FALSE]

cols = importance$Feature[1:50]
cols = cols[-36]
data = data[,(cols) := round(.SD,3), .SDcols=cols]

e_feature = fread("/Users/xiaofeifei/I/Kaggle/Basket/early_feature.csv")
rc_feature = fread("/Users/xiaofeifei/I/Kaggle/Basket/recent_comple.csv")

b = c("product_id", "user_id", "e_up_order_rate_since_se_first_order","e_user_total_products", "e_prod_reorder_times",
      "r_up_first_order","r_up_most_dow","r_up_orders","r_up_average_cart_position",
      "r_user_distinct_products")
e_feature = e_feature[,colnames(e_feature)[colnames(e_feature)%in%b], with=FALSE]
rc_feature = rc_feature[,colnames(rc_feature)[colnames(rc_feature)%in%b], with=FALSE]
data = data%>%
  left_join(e_feature)%>%
  left_join(rc_feature)

rm(e_feature,rc_feature)
#[80]	train-logloss:0.243746	test-logloss:0.244772 

Pattern = fread("/Users/xiaofeifei/I/Kaggle/Basket/pattern_feature.csv")
cols = colnames(Pattern)[4:10]
Pattern[,(cols) := round(.SD,3), .SDcols=cols]
data = data%>%
  left_join(Pattern)
rm(Pattern)

stack = fread("/Users/xiaofeifei/I/Kaggle/Basket/order_streaks.csv")
data = data%>%
  left_join(stack)
rm(stack)

best = fread("/Users/xiaofeifei/I/Kaggle/Basket/aisles_best_before.csv")
best = best[,c(1,3)]
data = data%>%
  left_join(best)
rm(best)

product_embeddings = fread("/Users/xiaofeifei/I/Kaggle/Basket/product_embeddings.csv")
colnames(product_embeddings) = as.character(product_embeddings[1,])
product_embeddings = product_embeddings[-1,]
product_embeddings = product_embeddings[,c(1,5:36)]
product_embeddings$product_id = as.numeric(product_embeddings$product_id)
data = data%>%
  left_join(product_embeddings)
rm(product_embeddings)


data = as.data.table(data)
data[is.na(data)] = 0

# fwrite(data, file ="/Users/xiaofeifei/I/Kaggle/Basket/all_features.csv" )
# feature = c(
#   'user_product_reordered_ratio', 'reordered_sum',
#   'add_to_cart_order_inverted_mean', 'add_to_cart_order_relative_mean',
#   'reorder_prob',
#   'dep_reordered_ratio', 'aisle_reordered_ratio',
#   'aisle_products',
#   'aisle_reordered',
#   'dep_products',
#   'dep_reordered',
#   'prod_users_unq', 'prod_users_unq_reordered',
#   'order_number', 'prod_add_to_card_mean',
#   'days_since_prior_order',
#   'order_dow', 'order_hour_of_day',
#   'reorder_ration',
#   'user_orders', 'user_order_starts_at', 'user_mean_days_since_prior',
#   'user_average_basket', 'user_distinct_products', 'user_reorder_ratio', 'user_total_products',
#   'prod_orders', 'prod_reorders',
#   'up_order_rate', 'up_orders_since_last_order', 'up_order_rate_since_first_order',
#   'up_orders', 'up_first_order', 'up_last_order', 'up_mean_cart_position',
#   'days_since_prior_order_mean',
#   'order_dow_mean',
#   "aisle_id","department_id"
# )
# 
# rm(Pattern)
# 
# cols = colnames(data)[!colnames(data) %in% feature]
# data = data[,cols, with = FALSE]
# 
# cols = colnames(data)[-c(1,2,16,17)]
# 
# data[,(cols) := round(.SD,3), .SDcols=cols]
# data[,reordered:=NULL]
# 
# train = fread("/Users/xiaofeifei/I/Kaggle/Basket/online_train_c.csv")
# colnames(train) = as.character(train[1,])
# train = train[-1,]
# feature = c(
#   'user_product_reordered_ratio', 'reordered_sum',
#   'add_to_cart_order_inverted_mean', 'add_to_cart_order_relative_mean',
#   'reorder_prob',
#   'dep_reordered_ratio', 'aisle_reordered_ratio',
#   'aisle_products',
#   'aisle_reordered',
#   'dep_products',
#   'dep_reordered',
#   'prod_users_unq', 'prod_users_unq_reordered',
#   'order_number', 'prod_add_to_card_mean',
#   'days_since_prior_order',
#   'order_dow', 'order_hour_of_day',
#   'reorder_ration',
#   'user_orders', 'user_order_starts_at', 'user_mean_days_since_prior',
#   'user_average_basket', 'user_distinct_products', 'user_reorder_ratio', 'user_total_products',
#   'prod_orders', 'prod_reorders',
#   'up_order_rate', 'up_orders_since_last_order', 'up_order_rate_since_first_order',
#   'up_orders', 'up_first_order', 'up_last_order', 'up_mean_cart_position',
#   'days_since_prior_order_mean',
#   'order_dow_mean',
#   "user_id","order_id",
#   "product_id","aisle_id","department_id"
# )
# 
# lable = fread("/Users/xiaofeifei/I/Kaggle/Basket/online_label.csv")
# lable = lable[-1]
# 
# train = train[,feature, with = FALSE]
# train$reordered = lable$V2
# 
# rm(lable)
# 
# cols <- colnames(train)
# train[,(cols):= lapply(.SD, as.numeric), .SDcols = cols]
# 
# setkey(train, user_id,order_id,product_id)
# setkey(data, user_id,order_id,product_id)
# 
# train = merge(train,data[data$eval_set == "train"],all.x=TRUE)
# train[,order_id:=NULL]
# 
# 
# fwrite(train, file = "/Users/xiaofeifei/I/Kaggle/Basket/final_train.csv")
# fwrite(data, file = "/Users/xiaofeifei/I/Kaggle/Basket/two_dig_feature.csv")
# 
# ######### test ##############
# data = fread("/Users/xiaofeifei/I/Kaggle/Basket/two_dig_feature.csv")
# 
# test = fread("/Users/xiaofeifei/I/Kaggle/Basket/online_test_c.csv")
# colnames(test) = as.character(test[1,])
# test = test[-1,]
# feature = c(
#   'user_product_reordered_ratio', 'reordered_sum',
#   'add_to_cart_order_inverted_mean', 'add_to_cart_order_relative_mean',
#   'reorder_prob',
#   'dep_reordered_ratio', 'aisle_reordered_ratio',
#   'aisle_products',
#   'aisle_reordered',
#   'dep_products',
#   'dep_reordered',
#   'prod_users_unq', 'prod_users_unq_reordered',
#   'order_number', 'prod_add_to_card_mean',
#   'days_since_prior_order',
#   'order_dow', 'order_hour_of_day',
#   'reorder_ration',
#   'user_orders', 'user_order_starts_at', 'user_mean_days_since_prior',
#   'user_average_basket', 'user_distinct_products', 'user_reorder_ratio', 'user_total_products',
#   'prod_orders', 'prod_reorders',
#   'up_order_rate', 'up_orders_since_last_order', 'up_order_rate_since_first_order',
#   'up_orders', 'up_first_order', 'up_last_order', 'up_mean_cart_position',
#   'days_since_prior_order_mean',
#   'order_dow_mean',
#   "user_id","order_id",
#   "product_id","aisle_id","department_id"
# )
# test = test[,feature, with = FALSE]
# 
# cols <- colnames(test)
# test[,(cols):= lapply(.SD, as.numeric), .SDcols = cols]
# 
# setkey(test, user_id,order_id,product_id)
# setkey(data, user_id,order_id,product_id)
# 
# test = merge(test,data[data$eval_set == "test"],all.x=TRUE)
# test$eval_set <- NULL
# 
# ###################
# data = fread("/Users/xiaofeifei/I/Kaggle/Basket/two_dig_feature.csv")
# 
# train = data[data$eval_set=='train']
# train[,eval_set:=NULL]
# 
# fwrite(train, file = "/Users/xiaofeifei/I/Kaggle/Basket/two_dig_train.csv")
# 
# rm(train)
# 
# test = data[data$eval_set=='test']
# test[,eval_set:=NULL]
# 
# fwrite(test, file = "/Users/xiaofeifei/I/Kaggle/Basket/two_dig_test.csv")
# 
# 
