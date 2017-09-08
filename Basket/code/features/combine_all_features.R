library(data.table)
library(dplyr)

rc_feature =fread("/Users/xiaofeifei/I/Kaggle/Basket/feature_recent.csv")
pattern_feature =fread("/Users/xiaofeifei/I/Kaggle/Basket/pattern_feature.csv")
paper_feature =fread("/Users/xiaofeifei/I/Kaggle/Basket/paper_feature.csv")
markov_feature =fread("/Users/xiaofeifei/I/Kaggle/Basket/markov_feature.csv")
feature =fread("/Users/xiaofeifei/I/Kaggle/Basket/feature.csv")
e_feature =fread("/Users/xiaofeifei/I/Kaggle/Basket/early_feature.csv")
feature_compli =fread("/Users/xiaofeifei/I/Kaggle/Basket/feature_compli.csv")
recent_comple =fread("/Users/xiaofeifei/I/Kaggle/Basket/recent_comple.csv")
stack = fread("/Users/xiaofeifei/I/Kaggle/Basket/order_streaks.csv")
best = fread("/Users/xiaofeifei/I/Kaggle/Basket/aisles_best_before.csv")
word2vec = fread("/Users/xiaofeifei/I/Kaggle/Basket/word2vec.csv")


feature = feature %>%
  left_join(feature_compli,by = c("order_id","user_id","product_id")) %>%
  left_join(stack, by=c("user_id","product_id")) %>%
  left_join(best, by = c("aisle_id")) %>%
  left_join(rc_feature, by=c("order_id","user_id","product_id")) %>%
  left_join(recent_comple, by=c("order_id","user_id","product_id")) %>%
  left_join(pattern_feature, by = c("user_id", "product_id")) %>%
  left_join(paper_feature, by=c("product_id","user_id"))%>%
  left_join(markov_feature, by = c("order_id","user_id","product_id"))%>%
  left_join(e_feature,by =c("order_id","user_id","product_id"))%>%
  left_join(word2vec, by=c("product_id"))


fwrite(feature, file = "/Users/xiaofeifei/I/Kaggle/Basket/all_features.csv", row.names = F)