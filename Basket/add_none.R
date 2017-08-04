test = fread("/Users/xiaofeifei/I/Kaggle/Basket/result1.csv")
none = fread("/Users/xiaofeifei/I/Kaggle/Basket/None1.csv")

none = none[-1,-3]
none = none[none$V2 == "None"]
colnames(none) = c("order_id","product_id")
none$reordered = 1

test$reordered <- (test$reordered > 0.21) * 1
test = rbind(test, none)

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

fwrite(submission, file = "none_test.csv", row.names = F)
