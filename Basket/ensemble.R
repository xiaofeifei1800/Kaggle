my_data = fread("/Users/xiaofeifei/I/Kaggle/Basket/ensemble/prediction_lgbm_my_data.csv")

stack = fread("/Users/xiaofeifei/I/Kaggle/Basket/ensemble/prediction_lgbm_stack_aisle.csv")

online = fread("/Users/xiaofeifei/I/Kaggle/Basket/ensemble/prediction_lgbm.csv")

result = fread("/Users/xiaofeifei/I/Kaggle/Basket/ensemble/result2.csv")

data = my_data %>%
  inner_join(stack, by = c("order_id", "product_id")) %>%
  inner_join(online, by = c("order_id", "product_id")) %>%
  inner_join(result, by = c("order_id", "product_id"))

a = as.data.table(a)
a[,predic:= rowMeans(a[,c("prediction","prediction.x","prediction.y","reordered")])]
a = a[,-c("prediction","prediction.x","prediction.y","reordered")]

a = a[,c(1,3,2)]
colnames(a) = c('order_id', 'prediction', 'product_id')

fwrite(a, file = "/Users/xiaofeifei/I/Kaggle/Basket/ensemble/ensemble.csv", row.names = F)

a = fread("/Users/xiaofeifei/I/Kaggle/Basket/ensemble/ensemble.csv")


