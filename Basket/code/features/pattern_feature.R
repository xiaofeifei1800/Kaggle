library(data.table)
library(dplyr)
library(tidyr)

# Pattern --------------------------------
# 1010101
# 000100010010101
# 000000011111

# Load Data ---------------------------------------------------------------
path <- "/Users/xiaofeifei/I/Kaggle/Basket/"

orderp <- fread(file.path(path, "order_products__prior.csv"))
orders <- fread(file.path(path, "orders.csv"))
products <- fread(file.path(path, "products.csv"))

# Reshape data ------------------------------------------------------------
orders_products <- orders %>% inner_join(orderp, by = "order_id") %>% 
  left_join(select(products,-product_name))

rm(orderp, products)

# Pattern feature ----------------------------------
Pattern = orders_products %>%
  filter(reordered == 1) %>%
  group_by(user_id, product_id) %>%
  mutate(cumsum_order_day = cumsum(days_since_prior_order))

Pattern = Pattern %>%
  group_by(user_id, product_id) %>%
  mutate(diff_order_day = c(0,diff(cumsum_order_day)))

Pattern = as.data.table(Pattern)
Pattern[,diff_order_day:=c(0,diff(cumsum_order_day)), by = c("user_id", "product_id")]
Pattern = Pattern[,c("order_id","product_id","user_id","cumsum_order_day","diff_order_day"),
                  with = FALSE]

Pattern1 = orders_products %>%
  filter(reordered == 1) %>%
  group_by(user_id, product_id) %>%
  summarise(
    order_number_mean = sum(order_number)/n(),
    order_number_skew = order_number_mean/max(order_number)
  )

Pattern2 = Pattern %>%
  group_by(user_id, product_id) %>%
  summarise(
    mean_diff_order_day = mean(diff_order_day)
  )

Pattern1 = Pattern1 %>% inner_join(Pattern2)
rm(Pattern2)
Pattern = Pattern %>% left_join(Pattern1)
rm(Pattern1)

range01 <- function(x){(x-min(x))/(max(x)-min(x))}

Pattern$order_number_mean = range01(Pattern$order_number_mean)
Pattern$order_number_skew = range01(Pattern$order_number_skew)
  
Pattern_earily = Pattern %>%
  group_by(user_id, product_id) %>%
  slice(c(2, 3, 4)) %>%
  summarise(
    e_mean_diff_order_day = mean(diff_order_day)
  )

Pattern_last = Pattern %>%
  group_by(user_id, product_id) %>%
  slice(c(n()-1,n())) %>%
  summarise(
    r_mean_diff_order_day = mean(diff_order_day)
  )


Pattern = as.data.table(Pattern)

Pattern = Pattern %>% 
  left_join(Pattern_earily) %>%
  left_join(Pattern_last)

Pattern = as.data.table(Pattern)
Pattern[is.na(Pattern)] = 0
fwrite(Pattern, file = "/Users/xiaofeifei/I/Kaggle/Basket/pattern_feature.csv", row.names = F)

# normalized to [0, 1] interval, order_number_mean, order_number_skew


















