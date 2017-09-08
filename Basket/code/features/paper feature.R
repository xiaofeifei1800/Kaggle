library(data.table)
library(dplyr)
library(tidyr)

# Feature from paper ---------------------------------------
# Repeat Buyer Prediction for E-Commerce
# http://www.kdd.org/kdd2016/papers/files/adf0160-liuA.pdf

# Load Data ---------------------------------------------------------------
path <- "/Users/xiaofeifei/I/Kaggle/Basket/"
orderp <- fread(file.path(path, "order_products__prior.csv"))
orders <- fread(file.path(path, "orders.csv"))
products <- fread(file.path(path, "products.csv"))

# Reshape data ------------------------------------------------------------
orders$eval_set <- as.factor(orders$eval_set)
ordert$user_id <- orders$user_id[match(ordert$order_id, orders$order_id)]

orders_products <- orders %>% inner_join(orderp, by = "order_id") %>% 
  left_join(select(products,-product_name))

rm(orderp,products)

paper = orders_products[,c("product_id","department_id","aisle_id","user_id")]
paper = as.data.table(paper)
paper = paper[!duplicated(paper)]
# Repeat buyer features
p <- orders_products %>%
  filter(reordered == 1)%>%
  group_by(product_id) %>% 
  summarise(
    repeat_p = n_distinct(user_id)
  )

p = orders_products %>%
  group_by(product_id) %>% 
  summarise(
    total_p = n_distinct(user_id)
  ) %>%
  right_join(p) %>%
  mutate(repet_p_ratio = repeat_p/total_p)%>%
  select(-total_p)
  
  
d <- orders_products %>%
  filter(reordered == 1)%>%
  group_by(department_id) %>% 
  summarise(
    repeat_d = n_distinct(user_id)
  )

d = orders_products %>%
  group_by(department_id) %>% 
  summarise(
    total_d = n_distinct(user_id)
  ) %>%
  right_join(d) %>%
  mutate(repet_d_ratio = repeat_d/total_d)%>%
  select(-total_d)

a <- orders_products %>%
  filter(reordered == 1)%>%
  group_by(aisle_id) %>% 
  summarise(
    repeat_a = n_distinct(user_id)
  )

a = orders_products %>%
  group_by(aisle_id) %>% 
  summarise(
    total_a = n_distinct(user_id)
  ) %>%
  right_join(a) %>%
  mutate(repet_a_ratio = repeat_a/total_a)%>%
  select(-total_a)

# Market share features
pd = orders_products %>%
  filter(reordered == 1)%>%
  group_by(department_id, product_id) %>%
  summarise(
   n_dp = n()
  )
  
pd = orders_products %>%
  filter(reordered == 1)%>%
  group_by(department_id) %>%
  summarise(
    n_d = n()
  ) %>%
  inner_join(pd)%>%
  ungroup() %>%
  mutate(n_dp_ratio = n_dp/n_d)
  
pa = orders_products %>%
  filter(reordered == 1)%>%
  group_by(aisle_id, product_id) %>%
  summarise(
    n_ap = n()
  )

pa = orders_products %>%
  filter(reordered == 1)%>%
  group_by(aisle_id) %>%
  summarise(
    n_a = n()
  ) %>%
  inner_join(pa)%>%
  ungroup() %>%
  mutate(n_ap_ratio = n_ap/n_a)

# user
pd1 = orders_products %>%
  filter(reordered == 1)%>%
  group_by(department_id, product_id) %>%
  summarise(
    u_dp = n_distinct(user_id)
  )

pd = orders_products %>%
  filter(reordered == 1)%>%
  group_by(department_id) %>%
  summarise(
    u_d = n_distinct(user_id)
  ) %>%
  inner_join(pd1)%>%
  ungroup() %>%
  mutate(u_dp_ratio = u_dp/u_d) %>%
  inner_join(pd)

pa1 = orders_products %>%
  filter(reordered == 1)%>%
  group_by(aisle_id, product_id) %>%
  summarise(
    u_ap = n_distinct(user_id)
  )

pa1 = orders_products %>%
  filter(reordered == 1)%>%
  group_by(aisle_id) %>%
  summarise(
    u_a = n_distinct(user_id)
  ) %>%
  inner_join(pa1)

pa1$u_ap_ratio = pa1$u_ap/pa1$u_a
pa = inner_join(pa1, pa)
  
rm(pd1, pa1)

# repeat buy day
up = orders_products %>%
  filter(reordered==1)%>%
  group_by(user_id, product_id) %>%
  summarise(
    repeat_buy_day_o = mean(days_since_prior_order)
    # repeat_buy_day = mean(days_since_prior_order)
  )

up = orders_products %>%
  group_by(user_id, product_id) %>%
  summarise(
    repeat_buy_day = mean(days_since_prior_order)
  )%>%
  left_join(up)

up = orders_products %>%
  filter(reordered==1)%>%
  group_by(product_id) %>%
  summarise(
    repeat_buy_day_p = mean(days_since_prior_order)
  )%>%
  right_join(up)

up$repeat_buy_day_ratio = up$repeat_buy_day_o/up$repeat_buy_day_p

up = as.data.table(up)
up[is.na(up)]=0

rm(orders, orders_products)

paper = paper %>%
  # left_join(a)%>%
  left_join(d)%>%
  left_join(p)%>%
  left_join(pa)%>%
  left_join(pd)%>%
  left_join(up)

fwrite(paper, file = "/Users/xiaofeifei/I/Kaggle/Basket/paper_feature.csv", row.names = F)

  



