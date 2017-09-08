library(data.table)
library(dplyr)
library(tidyr)


# Load Data ---------------------------------------------------------------
path <- "/Users/xiaofeifei/I/Kaggle/Basket/"

orderp <- fread(file.path(path, "order_products__prior.csv"))

orders <- fread(file.path(path, "orders.csv"))
products <- fread(file.path(path, "products.csv"))
orders = orders %>%
  group_by(user_id) %>%
  slice(c(n()-2,n()-1, n())) %>%
  ungroup()

# Reshape data ------------------------------------------------------------

orders_products <- orders %>% inner_join(orderp, by = "order_id") 

# Users -------------------------------------------------------------------
r_users <- orders %>%
  filter(eval_set == "prior") %>%
  group_by(user_id) %>%
  summarise(
    # 用户的总订单数
    r_user_orders = max(order_number),
  )

#### data -----------
data <- orders_products %>%
  group_by(user_id, product_id) %>% 
  summarise(
    # 用户购买该商品的总次数
    r_up_orders = n(),
    # 用户购买该商品第一次在哪个order里
    r_up_first_order = min(order_number),
    # # 用户购买该商品最后2次在哪个order里
    # up_se_last_order = max( c(0,order_number)[c(0,order_number)!=max(c(0,order_number))], na.rm = T),
    # 用户购买该商品最后一次在哪个order里
    r_up_last_order = max(order_number),
    #该商品被添加到购物篮中的平均位置
    r_up_average_cart_position = mean(add_to_cart_order),
    # mean order_hour_of_day
    r_up_last_day = mean(order_hour_of_day)
    # mode order_hour_of_day
    # up_last_day = which.max(table(order_hour_of_day)),
    # mode order_dow
    # up_most_dow = which.max(table(order_dow))
  )

dow_mode = orders_products[, c("user_id", "product_id", "order_hour_of_day", "order_dow", "order_number")]
dow_mode = as.data.table(dow_mode)
# user preferred time of day
dow_mode[,r_up_most_day:=Mode(order_hour_of_day), by = c("user_id","product_id")]
# user preferred day of week
dow_mode[,r_up_most_dow:=Mode(order_dow), by = c("user_id","product_id")]
#
dow_mode[,r_up_se_last_order:= max(c(0,order_number)[c(0,order_number)!=max(c(0,order_number))],
                                   na.rm = T), by = c("user_id","product_id")]
dow_mode = unique(dow_mode[, c("user_id", "r_up_most_day", "r_up_most_dow", "product_id","r_up_se_last_order")])

data = data %>% inner_join(dow_mode)

data = data %>% 
  left_join(r_users, by = "user_id")

#该商品购买次数 / 总的订单数
data$r_up_order_rate <- data$r_up_orders / data$r_user_orders
# 最近一次购买商品 - 最后一次购买该商品
data$r_up_orders_since_last_order <- data$r_user_orders - data$r_up_last_order
# 该商品购买次数 / 第一次购买该商品到最后一次购买商品的的订单数
data$r_up_order_rate_since_first_order <- data$r_up_orders / (data$r_user_orders - data$r_up_first_order + 1)

# 最近一次购买商品 - 最后2次购买该商品
data$r_up_orders_since_se_last_order <- data$r_up_last_order - data$r_up_se_last_order
# 该商品购买次数 / 第一次购买该商品到最后2次购买商品的的订单数
data$r_up_order_rate_since_se_first_order <- data$r_up_orders / (data$r_user_orders - data$r_up_se_last_order + 1)

data$r_user_orders = NULL

fwrite(data, file = "/Users/xiaofeifei/I/Kaggle/Basket/recent_comple.csv", row.names = FALSE)
