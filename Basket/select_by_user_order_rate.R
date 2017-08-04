orders <- fread(file.path(path, "orders.csv"))
orderp <- fread(file.path(path, "order_products__prior.csv"))
test = fread("/Users/xiaofeifei/I/Kaggle/Basket/result1.csv")

orders_products <- orders %>% inner_join(orderp, by = "order_id")
test = test %>% left_join(select(orders,"order_id", "user_id"), by = "order_id")

rm(orders, orderp)

us <- orders_products %>%
  group_by(user_id) %>%
  summarise(user_reorder_ratio = sum(reordered == 1) / sum(order_number > 1))

test = test %>% left_join(us) 
rm(orders_products,us)

test = as.data.table(test)
test[,reordered := (reordered >= user_reorder_ratio*0.7)* 1]
test = test[, c("order_id", "product_id", "reordered")]

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
fwrite(submission, file = "n_order_test.csv", row.names = F)



a = fread("/Users/xiaofeifei/I/Kaggle/Basket/F1_max1.csv")
a = a[-1,]
colnames(a) = c("products", "order_id")
setcolorder(a, c("order_id", "products"))

fwrite(a, file = "/Users/xiaofeifei/I/Kaggle/Basket/F1_max1.csv", row.names = F)

