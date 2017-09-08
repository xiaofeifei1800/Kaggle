library(data.table)
library(dplyr)
library(tidyr)


# Load Data ---------------------------------------------------------------
path <- "/Users/xiaofeifei/I/Kaggle/Basket/"

aisles <- fread(file.path(path, "aisles.csv"))
departments <- fread(file.path(path, "departments.csv"))
orderp <- fread(file.path(path, "order_products__prior.csv"))

orders <- fread(file.path(path, "orders.csv"))
products <- fread(file.path(path, "products.csv"))

# select last 3 orders
orders = orders %>%
  group_by(user_id) %>%
  slice(c(n()-2,n()-1, n())) %>%
  ungroup()

# combine data ------------------------------------------------------------
aisles$aisle <- as.factor(aisles$aisle)
departments$department <- as.factor(departments$department)
orders$eval_set <- as.factor(orders$eval_set)
products$product_name <- as.factor(products$product_name)

orders_products <- orders %>% inner_join(orderp, by = "order_id") %>% 
    left_join(select(products,-product_name))

recent = orders_products[,c("product_id","order_id","user_id")]
recent = as.data.table(recent)
recent = recent[!duplicated(recent)]

# Users -------------------------------------------------------------------
r_users <- orders %>%
  filter(eval_set == "prior") %>%
  group_by(user_id) %>%
  summarise(
    # 用户的总订单数
    r_user_orders = max(order_number),
    # total user time 
    r_user_period = sum(days_since_prior_order, na.rm = T),
    r_user_mean_days_since_prior = mean(days_since_prior_order, na.rm = T),
    r_user_median_days_since_prior = median(days_since_prior_order, na.rm = T),
    r_user_sd_days_since_prior = var(days_since_prior_order, na.rm = T),
    # user preferred time of day
    r_up_last_day = which.max(table(order_hour_of_day)),
    # user preferred day of week
    r_up_most_dow = which.max(table(order_dow))
  )

us <- orders_products %>%
  group_by(user_id) %>%
  summarise(
    #用户购买的总商品数
    r_user_total_products = n(),
    # reorder的总次数 / 第一单后买后的总次数
    r_user_reorder_ratio = sum(reordered == 1) / sum(order_number > 1),
    #用户购买的unique商品数
    r_user_distinct_products = n_distinct(product_id),
    #用户购买的unique department
    r_user_median_basket = n_distinct(department_id)
  )

r_users <- r_users %>% inner_join(us)
r_users$r_user_average_basket <- r_users$r_user_total_products / r_users$r_user_orders

us <- orders %>%
  filter(eval_set != "prior") %>%
  select(user_id, order_id, eval_set,
         time_since_last_order = days_since_prior_order)

r_users <- r_users %>% inner_join(us)

r_users <- orders_products %>%
  group_by(user_id, order_id) %>%
  summarise(r_order_size = n()) %>%
  ungroup() %>%
  group_by(user_id) %>%
  summarise(r_mean_order_size = mean(r_order_size)) %>%
  right_join(r_users)


rm(us)

# Products ----------------------------------------------------------------
prd <- orders_products %>%
  arrange(user_id, order_number, product_id) %>%
  group_by(user_id, product_id) %>%
  mutate(r_product_time = row_number()) %>%
  ungroup() %>%
  group_by(product_id) %>%
  summarise(
    # total order number of each product
    r_prod_orders = n(),
    # total reorder number of each product
    r_prod_reorders = sum(reordered),
    # for calculate product reorder_probability 
    # if product_time = 2 exit, then the product definatly be reordered
    r_prod_first_orders = sum(r_product_time == 1),
    r_prod_second_orders = sum(r_product_time == 2)
  )

prd$r_prod_reorder_probability <- prd$r_prod_second_orders / prd$r_prod_first_orders
prd$r_prod_reorder_times <- 1 + prd$r_prod_reorders / prd$r_prod_first_orders
prd$r_prod_reorder_ratio <- prd$r_prod_reorders / prd$r_prod_orders

prd <- prd %>% select(-r_prod_first_orders, -r_prod_second_orders)

rm(orderp,orders)
gc()

recent = recent %>% 
  left_join(prd, by = "product_id") %>%
  left_join(r_users, by = "user_id")
  
rm(aisles, departments,orders_products,prd,products,r_users)

recent$order_id.y=NULL
recent$eval_set = NULL
colnames(recent)[2] = 'order_id'
fwrite(recent, file = "/Users/xiaofeifei/I/Kaggle/Basket/feature_recent.csv",row.names = F)