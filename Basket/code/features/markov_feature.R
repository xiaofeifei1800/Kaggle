library(data.table)
library(dplyr)
library(tidyr)
library(reshape2)
library(data.table)

# markov feature -------------------
# apply markov chain, 10,11,01,00 probability


orders <- fread("/Users/xiaofeifei/I/Kaggle/Basket/orders.csv", select=c("order_id", "user_id", "order_number", "eval_set"))
orderp <- fread("/Users/xiaofeifei/I/Kaggle/Basket/order_products__prior.csv", drop="add_to_cart_order")

# prior
product_list <- orderp[, .(current_order = list(product_id)), by=order_id]
order_list <- merge(orders, product_list, by="order_id")

#
intersect <- function(x, y) y[match(x, y, 0L)]
setdiff <- function(x, y) x[match(x, y, 0L) == 0L]

# add lag
setorderv(order_list, c("user_id", "order_number"))
order_list[, previous_order:= shift(list(current_order)), by=user_id]
order_list[1:5]

#
order_list[order_number>1, T11 := mapply(intersect, previous_order, current_order)]
order_list[order_number>1, T01 := mapply(setdiff,   current_order,  previous_order)]
order_list[order_number>1, T10 := mapply(setdiff,   previous_order, current_order)]

order_list[1:3, -c("order_id", "user_id", "eval_set")]

#
N_PRODUCTS <- 49688L
transitionCount <- function(L) tabulate(unlist(L), nbins=N_PRODUCTS)

order_list[, n_orders := max(order_number), user_id] 
N <- order_list[, sum(n_orders-1L)]

N1  <- order_list[order_number>1, transitionCount(previous_order)]
N11 <- order_list[order_number>1, transitionCount(T11)]
N10 <- order_list[order_number>1, transitionCount(T10)]

N0  <- N - N1
N01 <- order_list[order_number>1, transitionCount(T01)]
N00 <- N0 - N01

P <- data.table(
  product_id = 1:N_PRODUCTS,
  # transitions probabilities out of state 0
  P0 = (N0+1) / (N+2), 
  P00 = (N00+1) / (N0+2),
  P01 = (N01+1) / (N0+2),
  # transitions probabilities out of state 1
  P1 = (N1+1) / (N+2), 
  P10 = (N10+1) / (N1+2),
  P11 = (N11+1) / (N1+2)
)



# Load Data ---------------------------------------------------------------
path <- "/Users/xiaofeifei/I/Kaggle/Basket/"

aisles <- fread(file.path(path, "aisles.csv"))
departments <- fread(file.path(path, "departments.csv"))
orderp <- fread(file.path(path, "order_products__prior.csv"))

orders <- fread(file.path(path, "orders.csv"))
products <- fread(file.path(path, "products.csv"))
orders_pre = orders %>%
  group_by(user_id) %>%
  slice(c(n()-1)) %>%
  ungroup()
  
orders_pre = left_join(orders_pre, orderp, by = "order_id")

orders_pre = orders_pre[,c("order_id", "user_id","product_id","reordered")]

new_P = P

new_P <- melt(new_P, id=c("product_id"))

levels(new_P$variable) = c(0,0,0,1,1,1)

P_0 = new_P[new_P$variable == 0]
P_1 = new_P[new_P$variable == 1]
P_0 = P_0[order(product_id, decreasing = F)]
P_1 = P_1[order(product_id, decreasing = F)]

P_0[,prob :=1:3]
P_1[,prob :=1:3]

P_0[,variable:=NULL]
P_1[,variable:=NULL]
P_0 = dcast(data = P_0,formula = product_id~prob,value.var = "value")
P_1 = dcast(data = P_1,formula = product_id~prob,value.var = "value")

P_0$reordered = 0
P_1$reordered = 1

colnames(P_0)[c(2,3,4)] = c("P0","P1","P2")
colnames(P_1)[c(2,3,4)] = c("P0","P1","P2")

p = rbind(P_0,P_1)
orders_pre = orders_pre[,c(1,2,3,4)]
orders_pre = left_join(orders_pre, p, by = c("product_id", "reordered"))

colnames(orders_pre)[4] = "pre_reorder"

orders_pre = as.data.table(orders_pre)

fwrite(orders_pre, file = "/Users/xiaofeifei/I/Kaggle/Basket/markov_feature.csv")



