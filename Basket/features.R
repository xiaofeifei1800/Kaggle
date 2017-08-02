###########################################################################################################
#
# Kaggle Instacart competition
# Fabien Vavrand, June 2017
# Simple xgboost starter, score 0.3791 on LB
# Products selection is based on product by product binary classification, with a global threshold (0.21)
#
###########################################################################################################

library(data.table)
library(dplyr)
library(tidyr)


# Load Data ---------------------------------------------------------------
path <- "/Users/xiaofeifei/I/Kaggle/Basket/"

aisles <- fread(file.path(path, "aisles.csv"))
departments <- fread(file.path(path, "departments.csv"))
orderp <- fread(file.path(path, "order_products__prior.csv"))
ordert <- fread(file.path(path, "order_products__train.csv"), nrows = 2000)
orders <- fread(file.path(path, "orders.csv"), nrows = 2000)
products <- fread(file.path(path, "products.csv"))

recency_order = orders %>%
    group_by(user_id) %>%
    slice(c(n()-2,n()-1, n())) %>%
    ungroup()

# Reshape data ------------------------------------------------------------
aisles$aisle <- as.factor(aisles$aisle)
departments$department <- as.factor(departments$department)
orders$eval_set <- as.factor(orders$eval_set)
products$product_name <- as.factor(products$product_name)

products <- products %>% 
  inner_join(aisles) %>% inner_join(departments) %>% 
  select(-aisle_id, -department_id)
rm(aisles, departments)

ordert$user_id <- orders$user_id[match(ordert$order_id, orders$order_id)]

orders_products <- orders %>% inner_join(orderp, by = "order_id")

rm(orderp)
gc()


# Products ----------------------------------------------------------------
prd <- orders_products %>%
  arrange(user_id, order_number, product_id) %>%
  group_by(user_id, product_id) %>%
  mutate(product_time = row_number()) %>%
  ungroup() %>%
  group_by(product_id) %>%
  summarise(
    # total order number of each product
    prod_orders = n(),
    # user number
    prod_user = n_distinct(user_id),
    # total reorder number of each product
    prod_reorders = sum(reordered),
    # for calculate product reorder_probability 
    # if product_time = 2 exit, then the product definatly be reordered
    prod_first_orders = sum(product_time == 1),
    prod_second_orders = sum(product_time == 2)
  )

prd$prod_reorder_probability <- prd$prod_second_orders / prd$prod_first_orders
prd$prod_reorder_times <- 1 + prd$prod_reorders / prd$prod_first_orders
prd$prod_reorder_ratio <- prd$prod_reorders / prd$prod_orders

prd <- prd %>% select(-prod_first_orders, -prod_second_orders)

# _user_buy_product_times: 用户是第几次购买该商品 too early
# user_buy_product <- orders_products %>%
#   group_by(user_id, product_id) %>%
#   mutate(product_time = row_number())
# 
# user_buy_product = user_buy_product[,c("user_id", "product_id","product_time")]


rm(products)
gc()

# Users -------------------------------------------------------------------
users <- orders %>%
  filter(eval_set == "prior") %>%
  group_by(user_id) %>%
  summarise(
    # 用户的总订单数
    user_orders = max(order_number),
    # total user time 
    user_period = sum(days_since_prior_order, na.rm = T),
    user_mean_days_since_prior = mean(days_since_prior_order, na.rm = T),
    user_median_days_since_prior = median(days_since_prior_order, na.rm = T),
    user_sd_days_since_prior = var(days_since_prior_order, na.rm = T)
  )

us <- orders_products %>%
  group_by(user_id) %>%
  summarise(
    #用户购买的总商品数
    user_total_products = n(),
    # reorder的总次数 / 第一单后买后的总次数
    user_reorder_ratio = sum(reordered == 1) / sum(order_number > 1),
    #用户购买的unique商品数
    user_distinct_products = n_distinct(product_id),
    #用户购买的unique department
    user_median_basket = median(user_order_product_num)
  )

users <- users %>% inner_join(us)
users$user_average_basket <- users$user_total_products / users$user_orders

us <- orders %>%
  filter(eval_set != "prior") %>%
  select(user_id, order_id, eval_set,
         time_since_last_order = days_since_prior_order)

users <- users %>% inner_join(us)

rm(us)
gc()

# User time interaction----------------------------------------------------------------
# similar features for products and aisles
#user preferred day of week, user preferred time of day,
#users, #orders, order frequency, reorder rate, recency, mean add_to_cart_order, etc.
#Products purchased, #Orders made, frequency and recency of orders, #Aisle purchased from, 
#Department purchased from, frequency and recency of reorders, tenure, mean order size, etc.

# User aisle and department interaction ----------------------------------------------------------------
# similar to product features
#users, #orders, order frequency, reorder rate, recency, mean add_to_cart_order, etc.

# User aisle and department interaction: 
#similar to product features
#purchases, #reorders, #day since last purchase, #order since last purchase etc.

# Database ----------------------------------------------------------------
# product last order time 

# product user

data <- orders_products %>%
  group_by(user_id, product_id) %>% 
  summarise(
    # 用户购买该商品的总次数
    up_orders = n(),
    # 用户购买该商品第一次在哪个order里
    up_first_order = min(order_number),
    # # 用户购买该商品最后2次在哪个order里
    up_se_last_order = max( c(0,order_number)[c(0,order_number)!=max(c(0,order_number))], na.rm = T),
    # 用户购买该商品最后一次在哪个order里
    up_last_order = max(order_number),
    #该商品被添加到购物篮中的平均位置
    up_average_cart_position = mean(add_to_cart_order),
    # mean order_hour_of_day
    up_last_day = mean(order_hour_of_day),
    # mpde order_hour_of_day
    up_last_day = which.max(table(order_hour_of_day)),
    # mod order_dow
    up_most_dow = which.max(table(order_dow))
    )

data <- orders_products %>%
  filter(reordered == 1) %>%
  group_by(user_id, product_id) %>%
  summarise(
    up_reorder_rate = n()
  ) %>%
  right_join(data)

data$up_reorder_rate = data$up_reorder_rate/data$up_orders
data[is.na(data)] = 0

rm(orders_products, orders)

data <- data %>% 
  inner_join(prd, by = "product_id") %>%
  inner_join(users, by = "user_id") 
  # inner_join(user_buy_product, by = c("user_id", "product_id"))

#该商品购买次数 / 总的订单数
data$up_order_rate <- data$up_orders / data$user_orders
# 最近一次购买商品 - 最后一次购买该商品
data$up_orders_since_last_order <- data$user_orders - data$up_last_order
# 该商品购买次数 / 第一次购买该商品到最后一次购买商品的的订单数
data$up_order_rate_since_first_order <- data$up_orders / (data$user_orders - data$up_first_order + 1)

# 最近一次购买商品 - 最后2次购买该商品
data$up_orders_since_se_last_order <- data$up_last_order - data$up_se_last_order
# 该商品购买次数 / 第一次购买该商品到最后2次购买商品的的订单数
data$up_order_rate_since_se_first_order <- data$up_orders / (data$user_orders - data$up_se_last_order + 1)

data <- data %>% 
  left_join(ordert %>% select(user_id, product_id, reordered), 
            by = c("user_id", "product_id"))

rm(ordert, prd, users)
gc()

# add word2vec
products <- fread(file.path(path, "products.csv"))
word2vec = fread(file.path(path, "word2vec.csv"))
prod_name = fread(file.path(path, "pca_name.csv"))

products = left_join(products,word2vec)
products = left_join(products,prod_name)
products[is.na(products)] = 0

rm(word2vec)

data = as.data.table(data)
products = as.data.table(products)
setkey(data, product_id)
setkey(products, product_id)

data = merge(data, products, all.x = T)