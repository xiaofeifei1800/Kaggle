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

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
# Load Data ---------------------------------------------------------------
path <- "/Users/xiaofeifei/I/Kaggle/Basket/"

orderp <- fread(file.path(path, "order_products__prior.csv"))

orders <- fread(file.path(path, "orders.csv"))
products <- fread(file.path(path, "products.csv"))

orders = orders %>%
  group_by(user_id) %>%
  slice(c(2,3,4)) %>%
  ungroup()


# Reshape data ------------------------------------------------------------
orders_products <- orders %>% inner_join(orderp, by = "order_id") %>% 
  left_join(select(products,-product_name))

rm(orderp, products)
# Users -------------------------------------------------------------------
r_users <- orders_products %>%
  group_by(user_id) %>%
  summarise(
    # 用户的总订单数
    e_user_orders = max(order_number),
    # total user time 
    e_user_period = sum(days_since_prior_order, na.rm = T),
    e_user_mean_days_since_prior = mean(days_since_prior_order, na.rm = T),
    e_user_median_days_since_prior = median(days_since_prior_order, na.rm = T),
    e_user_sd_days_since_prior = var(days_since_prior_order, na.rm = T),
    # user preferred time of day
    e_up_last_day = which.max(table(order_hour_of_day)),
    # user preferred day of week
    e_up_most_dow = which.max(table(order_dow))
  )

us <- orders_products %>%
  group_by(user_id) %>%
  summarise(
    #用户购买的总商品数
    e_user_total_products = n(),
    # reorder的总次数 / 第一单后买后的总次数
    e_user_reorder_ratio = sum(reordered == 1) / sum(order_number > 1),
    #用户购买的unique商品数
    e_user_distinct_products = n_distinct(product_id),
    #用户购买的unique department
    e_user_median_basket = n_distinct(department_id)
  )

r_users <- r_users %>% inner_join(us)
r_users$e_user_average_basket <- r_users$e_user_total_products / r_users$e_user_orders

r_users <- orders_products %>%
  group_by(user_id, order_id) %>%
  summarise(e_order_size = n()) %>%
  ungroup() %>%
  group_by(user_id) %>%
  summarise(e_mean_order_size = mean(e_order_size)) %>%
  right_join(r_users)


rm(us)

# Products ----------------------------------------------------------------
prd <- orders_products %>%
  arrange(user_id, order_number, product_id) %>%
  group_by(user_id, product_id) %>%
  mutate(e_product_time = row_number()) %>%
  ungroup() %>%
  group_by(product_id) %>%
  summarise(
    # total order number of each product
    e_prod_orders = n(),
    # total reorder number of each product
    e_prod_reorders = sum(reordered),
    # for calculate product reorder_probability 
    # if product_time = 2 exit, then the product definatly be reordered
    e_prod_first_orders = sum(e_product_time == 1),
    e_prod_second_orders = sum(e_product_time == 2)
  )

prd$e_prod_reorder_probability <- prd$e_prod_second_orders / prd$e_prod_first_orders
prd$e_prod_reorder_times <- 1 + prd$e_prod_reorders / prd$e_prod_first_orders
prd$e_prod_reorder_ratio <- prd$e_prod_reorders / prd$e_prod_orders

prd <- prd %>% select(-e_prod_first_orders, -e_prod_second_orders)

rm(orderp,orders)
gc()


#### data -----------
data <- orders_products %>%
  group_by(user_id, product_id) %>% 
  summarise(
    # 用户购买该商品的总次数
    e_up_orders = n(),
    # 用户购买该商品第一次在哪个order里
    e_up_first_order = min(order_number),
    # # 用户购买该商品最后2次在哪个order里
    # up_se_last_order = max( c(0,order_number)[c(0,order_number)!=max(c(0,order_number))], na.rm = T),
    # 用户购买该商品最后一次在哪个order里
    e_up_last_order = max(order_number),
    #该商品被添加到购物篮中的平均位置
    e_up_average_cart_position = mean(add_to_cart_order),
    # mean order_hour_of_day
    e_up_last_day = mean(order_hour_of_day)
    # mode order_hour_of_day
    # up_last_day = which.max(table(order_hour_of_day)),
    # mode order_dow
    # up_most_dow = which.max(table(order_dow))
  )

dow_mode = orders_products[, c("user_id", "product_id", "order_hour_of_day", "order_dow", "order_number")]
dow_mode = as.data.table(dow_mode)
# user preferred time of day
dow_mode[,e_up_most_day:=Mode(order_hour_of_day), by = c("user_id","product_id")]
# user preferred day of week
dow_mode[,e_up_most_dow:=Mode(order_dow), by = c("user_id","product_id")]
#
dow_mode[,e_up_se_last_order:= max(c(0,order_number)[c(0,order_number)!=max(c(0,order_number))],
                                 na.rm = T), by = c("user_id","product_id")]
dow_mode = unique(dow_mode[, c("user_id", "e_up_most_day", "e_up_most_dow", "product_id","e_up_se_last_order")])

data = data %>% inner_join(dow_mode)

data = data %>% 
  left_join(prd, by = "product_id") %>%
  left_join(r_users, by = "user_id")

#该商品购买次数 / 总的订单数
data$e_up_order_rate <- data$e_up_orders / data$e_user_orders
# 最近一次购买商品 - 最后一次购买该商品
data$e_up_orders_since_last_order <- data$e_user_orders - data$e_up_last_order
# 该商品购买次数 / 第一次购买该商品到最后一次购买商品的的订单数
data$e_up_order_rate_since_first_order <- data$e_up_orders / (data$e_user_orders - data$e_up_first_order + 1)


# 最近一次购买商品 - 最后2次购买该商品
data$e_up_orders_since_se_last_order <- data$e_up_last_order - data$e_up_se_last_order
# 该商品购买次数 / 第一次购买该商品到最后2次购买商品的的订单数
data$e_up_order_rate_since_se_first_order <- data$e_up_orders / (data$e_user_orders - data$e_up_se_last_order + 1)

data$e_up_last_day.y = NULL
data$e_up_most_dow.y = NULL

colnames(data)[c(7,9)] = c("e_up_last_day", "e_up_most_dow")

fwrite(data, file = "/Users/xiaofeifei/I/Kaggle/Basket/early_feature.csv",row.names = F)

