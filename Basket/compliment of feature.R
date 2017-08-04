library(data.table)
library(dplyr)
library(tidyr)

#
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

orders_products <- orders %>% left_join(orderp, by = "order_id") %>% 
  left_join(select(products,-product_name))


# User ------------------------------------------------------------
user_mode = orders[orders$eval_set == "prior", c("user_id", "order_hour_of_day", "order_dow",
                                                 "order_number")]
# user preferred time of day
user_mode[,u_last_day:=Mode(order_hour_of_day), by = user_id]
# user preferred day of week
user_mode[,u_most_dow:=Mode(order_dow), by = user_id]
user_mode[,user_orders := max(order_number), by = user_id]
user_mode[,order_number:=NULL]
user_mode = unique(user_mode[, c("user_id", "u_last_day", "u_most_dow", "user_orders")])

us <- orders %>%
  filter(eval_set != "prior") %>%
  select(user_id, order_id)

user_mode <- user_mode %>% inner_join(us)
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
    ut_aisle = n_distinct(aisle_id),
  )

dow_mode = orders_products[, c("user_id", "order_dow", "department_id", "aisle_id")]
dow_mode = as.data.table(dow_mode)
# user preferred time of department_id
dow_mode[,ut_mode_department:=Mode(department_id), by = c("user_id","order_dow")]
# user preferred day of aisle_id
dow_mode[,ut_mode_aisle:=Mode(aisle_id), by = c("user_id","order_dow")]
dow_mode = unique(dow_mode[, c("user_id", "ut_mode_department", "ut_mode_aisle", "order_dow")])
user_dow = user_dow%>% inner_join(dow_mode)

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

dow_mode = orders_products[, c("user_id", "order_hour_of_day", "department_id", "aisle_id")]
dow_mode = as.data.table(dow_mode)
# user preferred time of day
dow_mode[,uh_mode_department:=Mode(department_id), by = c("user_id","order_hour_of_day")]
# user preferred day of week
dow_mode[,uh_mode_aisle:=Mode(aisle_id), by = c("user_id","order_hour_of_day")]
dow_mode = unique(dow_mode[, c("user_id", "uh_mode_department", "uh_mode_aisle", "order_hour_of_day")])
user_hour = user_hour%>% inner_join(dow_mode)

user_hour$uh_order_frequency <- user_hour$uh_orders / user_hour$uh_user
user_hour$uh_reorder_ratio <- user_hour$uh_reoreder / user_hour$uh_orders

rm(orderp,products,dow_mode)

# data----------------------------------------data <- orders_products %>%
data <- orders_products %>%
  group_by(user_id, product_id) %>% 
  summarise(
    up_orders = n(),
    # 用户购买该商品最后一次在哪个order里
    up_last_order = max(order_number),
    up_last_day = mean(order_hour_of_day)

  )

dow_mode = orders_products[, c("user_id", "product_id", "order_hour_of_day", "order_dow", "order_number")]
dow_mode = as.data.table(dow_mode)
# user preferred time of day
dow_mode[,up_most_day:=Mode(order_hour_of_day), by = c("user_id","product_id")]
# user preferred day of week
dow_mode[,up_most_dow:=Mode(order_dow), by = c("user_id","product_id")]
#
dow_mode[,up_se_last_order:= max(c(0,order_number)[c(0,order_number)!=max(c(0,order_number))],
                                 na.rm = T), by = c("user_id","product_id")]
dow_mode = unique(dow_mode[, c("user_id", "up_most_day", "up_most_dow", "product_id","up_se_last_order")])

data = data %>% inner_join(dow_mode)


data = data %>% 
  left_join(user_mode)


# 最近一次购买商品 - 最后2次购买该商品
data$up_orders_since_se_last_order <- data$up_last_order - data$up_se_last_order
# 该商品购买次数 / 第一次购买该商品到最后2次购买商品的的订单数
data$up_order_rate_since_se_first_order <- data$up_orders / (data$user_orders - data$up_se_last_order + 1)
data$up_orders = NULL
data$up_last_order = NULL
data$up_last_day = NULL

order_time = unique(orders[,c("order_id","order_hour_of_day","order_dow")])
data = data %>% 
  inner_join(order_time) %>%
  left_join(user_hour) %>%
  left_join(user_dow)

data = fread("/Users/xiaofeifei/I/Kaggle/Basket/feature_clean.csv")

data = data %>%
  left_join(data1, by = c("user_id", "product_id","order_id"))

rm(data1)
data$ua_order_frequency.y = NULL
data$user_orders.y = NULL
data$up_most_dow.y = NULL
data$product_name = NULL
colnames(data)[c(16,21,38)] = c("user_orders","up_most_dow","ua_order_frequency")

data[is.na(data)] = 0
fwrite(data, file = "/Users/xiaofeifei/I/Kaggle/Basket/feature.csv", row.names = F)







