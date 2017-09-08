
imp = fread("/Users/xiaofeifei/I/Kaggle/Basket/importance_8.csv")

e_feature = fread("/Users/xiaofeifei/I/Kaggle/Basket/F1_Max_ensemble6_1.csv")
rc_feature = fread("/Users/xiaofeifei/I/Kaggle/Basket/recent_comple.csv",nrows = 10)

Pattern = fread("/Users/xiaofeifei/I/Kaggle/Basket/pattern_feature.csv",nrows = 10)

stack = fread("/Users/xiaofeifei/I/Kaggle/Basket/order_streaks.csv",nrows = 10)

best = fread("/Users/xiaofeifei/I/Kaggle/Basket/aisles_best_before.csv")

compliment = fread("/Users/xiaofeifei/I/Kaggle/Basket/recent.csv")


rc_feature= rc_feature[,-c('r_up_last_day', 'r_up_most_dow', 'r_up_last_day', 'r_up_most_dow')]

fwrite(rc_feature, file = "/Users/xiaofeifei/I/Kaggle/Basket/recent_comple.csv", row.names = F)


e_feature = fread("/Users/xiaofeifei/I/Kaggle/Basket/F1_Max_my_data1_new.csv")
e_feature1 = fread("/Users/xiaofeifei/I/Kaggle/Basket/F1_Max_ensemble6_2.csv")

temp = setdiff(union(e_feature$order_id,e_feature1$order_id), intersect(e_feature$order_id,e_feature1$order_id))


e_feature = rbind(e_feature,  e_feature1[e_feature1$order_id%in%temp])

e_feature = e_feature[order(order_id)]

fwrite(e_feature , file = "/Users/xiaofeifei/I/Kaggle/Basket/F1_Max_my_data1_new.csv", row.names = FALSE)










