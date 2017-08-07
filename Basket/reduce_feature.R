data = fread("/Users/xiaofeifei/I/Kaggle/Basket/feature.csv")
importance = fread("/Users/xiaofeifei/I/Kaggle/Basket/impotance.csv")

data = as.data.table(data)
data = data[,c("eval_set","order_id","product_id","reordered","user_id",importance$Feature[1:50]), with = FALSE]

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
data = data%>%
  left_join(Pattern)

data = as.data.table(data)
data[is.na(data)] = 0

rm(Pattern)





