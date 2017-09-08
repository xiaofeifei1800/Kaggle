library(data.table)
library(dplyr)
library(tidyr)

# function
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Load Data ---------------------------------------------------------------
path <- "/Users/xiaofeifei/I/Kaggle/Basket/"
orderp <- fread(file.path(path, "order_products__prior.csv"))
ordert <- fread(file.path(path, "order_products__train.csv"))
orders <- fread(file.path(path, "orders.csv"))
products <- fread(file.path(path, "products.csv"))

# Reshape data ------------------------------------------------------------
orders$eval_set <- as.factor(orders$eval_set)
ordert$user_id <- orders$user_id[match(ordert$order_id, orders$order_id)]

orders_products <- orders %>% inner_join(orderp, by = "order_id") %>% 
  left_join(select(products,-product_name))

rm(orderp,products)
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

user_mode = orders[orders$eval_set == "prior", c("user_id", "order_hour_of_day", "order_dow")]
# user preferred time of day
user_mode[,up_last_day:=Mode(order_hour_of_day), by = user_id]
# user preferred day of week
user_mode[,up_most_dow:=Mode(order_dow), by = user_id]
user_mode = unique(user_mode[, c("user_id", "up_last_day", "up_most_dow")])

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
    user_median_basket = n_distinct(department_id)
  )

users <- users %>% inner_join(us) %>% inner_join(user_mode)
users$user_average_basket <- users$user_total_products / users$user_orders

us <- orders %>%
  filter(eval_set != "prior") %>%
  select(user_id, order_id, eval_set,
         time_since_last_order = days_since_prior_order)

users <- users %>% inner_join(us)

users <- orders_products %>%
  group_by(user_id, order_id) %>%
  summarise(order_size = n()) %>%
  ungroup() %>%
  group_by(user_id) %>%
  summarise(mean_order_size = mean(order_size)) %>%
  right_join(users)


rm(us, user_mode)
gc()

# User time interaction----------------------------------------------------------------

user_dow = orders_products %>%
  group_by(user_id, order_dow) %>%
  summarise(
    # 用户购买该商品的总次数
    ut_product = n(),
    ut_reoreder = sum(reordered == 1),
    ut_user = n_distinct(user_id),
    ut_orders = n_distinct(order_id),
    ut_department = n_distinct(department_id),
    ut_aisle = n_distinct(aisle_id)
  )
user_dow$ua_order_frequency <- user_dow$ut_orders / user_dow$ut_user
user_dow$ut_reorder_ratio <- user_dow$ut_reoreder / user_dow$ut_orders

user_hour = orders_products %>%
  group_by(user_id, order_hour_of_day) %>%
  summarise(
    # 用户购买该商品的总次数
    uh_product = n(),
    uh_reoreder = sum(reordered == 1),
    uh_user = n_distinct(user_id),
    uh_orders = n_distinct(order_id),
    uh_department = n_distinct(department_id),
    uh_aisle = n_distinct(aisle_id)
  )
user_hour$uh_order_frequency <- user_hour$uh_orders / user_hour$uh_user
user_hour$uh_reorder_ratio <- user_hour$uh_reoreder / user_hour$uh_orders

# User aisle and department interaction ----------------------------------------------------------------
# similar to product features
#users, #orders, order frequency, reorder rate, recency, mean add_to_cart_order, etc.
user_aisle = orders_products %>%
  group_by(user_id, aisle_id) %>%
  summarise(
    # 用户购买该商品的总次数
    ua_product = n(),
    ua_reoreder = sum(reordered == 1),
    ua_user = n_distinct(user_id),
    ua_orders = n_distinct(order_id),
    ua_add_to_cart = mean(add_to_cart_order)
  )
user_aisle$ua_reorder_ratio <- user_aisle$ua_reoreder / user_aisle$ua_orders
user_aisle$ua_order_frequency <- user_aisle$ua_orders / user_aisle$ua_user

user_depart = orders_products %>%
  group_by(user_id, department_id) %>%
  summarise(
    # 用户购买该商品的总次数
    ud_product = n(),
    ud_reoreder = sum(reordered == 1),
    ud_user = n_distinct(user_id),
    ud_orders = n_distinct(order_id),
    ud_add_to_cart = mean(add_to_cart_order)
  )
user_depart$ud_reorder_ratio <- user_depart$ud_reoreder / user_depart$ud_orders
user_depart$ud_order_frequency <- user_depart$ud_orders / user_depart$ud_user

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
    # up_se_last_order = max( c(0,order_number)[c(0,order_number)!=max(c(0,order_number))], na.rm = T),
    # 用户购买该商品最后一次在哪个order里
    up_last_order = max(order_number),
    #该商品被添加到购物篮中的平均位置
    up_average_cart_position = mean(add_to_cart_order),
    # mean order_hour_of_day
    up_last_day = mean(order_hour_of_day)
    # mode order_hour_of_day
    # up_last_day = which.max(table(order_hour_of_day)),
    # mode order_dow
    # up_most_dow = which.max(table(order_dow))
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
  inner_join(users, by = "user_id") %>%
  inner_join(select(products,-product_name), by = "product_id") %>%
  inner_join(user_aisle, by = c("user_id", "aisle_id")) %>%
  inner_join(user_depart, by = c("user_id", "department_id"))
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

fwrite(data, file = "/Users/xiaofeifei/I/Kaggle/Basket/feature.csv", row.names = F)


