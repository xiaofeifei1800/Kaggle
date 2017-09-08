library(data.table)
library(dplyr)
my_data = fread("/Users/xiaofeifei/I/Kaggle/Basket/ensemble/prediction_lgbm_my_data.csv")

stack = fread("/Users/xiaofeifei/I/Kaggle/Basket/ensemble/prediction_lgbm_stack_aisle.csv")

online = fread("/Users/xiaofeifei/I/Kaggle/Basket/ensemble/prediction_lgbm.csv")

result = fread("/Users/xiaofeifei/I/Kaggle/Basket/ensemble/result2.csv")

result3 = fread("/Users/xiaofeifei/I/Kaggle/Basket/ensemble/result3.csv")

arboretum1 = fread("/Users/xiaofeifei/I/Kaggle/Basket/prediction_arboretum1.csv")

my_data1 = fread('/Users/xiaofeifei/I/Kaggle/Basket/prediction_lgbm_my_data1.csv')

data = my_data %>%
  inner_join(stack, by = c("order_id", "product_id")) %>%
  inner_join(online, by = c("order_id", "product_id")) 
  # inner_join(my_data1, by = c("order_id", "product_id")) %>%
  # inner_join(arboretum1, by = c("order_id", "product_id")) 

  # inner_join(result3, by = c("order_id", "product_id"))
a = data
rm(data)
a = as.data.table(a)
# a[,predic:= rowMeans(a[,c("prediction","prediction.x","prediction.y","reordered.x","reordered.y")])]
# a = a[,-c("prediction","prediction.x","prediction.y","reordered.x","reordered.y")]
a[,predic:= rowMeans(a[,c("prediction","prediction.x","prediction.y")])]

b = apply(a[,c("prediction.x","prediction.y","prediction.x.x","prediction.y.y","prediction")],1,median)
a[,predic:= b]
a = a[,-c("prediction.x","prediction.y","prediction")]
a = a[,c(1,3,2)]
colnames(a) = c('order_id', 'prediction', 'product_id')

fwrite(a, file = "/Users/xiaofeifei/I/Kaggle/Basket/ensemble/ensemble6.csv", row.names = F)

b = fread("/Users/xiaofeifei/I/Kaggle/Basket/ensemble/ensemble3.csv")









