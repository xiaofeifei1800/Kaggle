library(data.table)
library(dplyr)
library(ggplot2)
library(knitr)
library(stringr)
library(DT)
library(treemap)

orders <- fread('/Users/xiaofeifei/I/Kaggle/Basket/orders.csv')
products <- fread('/Users/xiaofeifei/I/Kaggle/Basket/products.csv')
order_products <- fread('/Users/xiaofeifei/I/Kaggle/Basket/order_products__train.csv')
order_products_prior <- fread('/Users/xiaofeifei/I/Kaggle/Basket/order_products__prior.csv')
aisles <- fread('/Users/xiaofeifei/I/Kaggle/Basket/aisles.csv')
departments <- fread('/Users/xiaofeifei/I/Kaggle/Basket/departments.csv')

######################## join orders
total_order = rbind(order_products, order_products_prior)
setkey(orders, order_id)
setkey(total_order, order_id)
orders_pro = merge(orders, total_order, all.x = T)

summary(orders_pro)

rm(orders, order_products, order_products_prior, total_order)

########################## join products
aisles$aisle <- as.factor(aisles$aisle)
departments$department <- as.factor(departments$department)
products$product_name <- as.factor(products$product_name)

products <- products %>% 
  inner_join(aisles) %>% inner_join(departments) %>% 
  select(-aisle_id, -department_id)

rm(aisles, departments)

########################## orders and products
products = as.data.table(products)
setkey(orders_pro, product_id)
setkey(products, product_id)

orders_pro = merge(orders_pro, products, all.x = T)

rm(products)

train = orders_pro[orders_pro$eval_set == "train"]
prior = orders_pro[orders_pro$eval_set == "prior"]
test = orders_pro[orders_pro$eval_set == "test"]
###################
ggplot(aes(x=order_hour_of_day), data = prior) +
  geom_histogram(stat="count",fill="red")

ggplot(aes(x=order_dow), data = prior) + 
  geom_histogram(stat="count",fill="red")

ggplot(aes(x=days_since_prior_order), data = prior) + 
  geom_histogram(stat="count",fill="red")

# people's order
orders_pro[,total_order := max(order_number), by = user_id]

a = as.data.frame(table(orders_pro$order_number))
a$Var1 = as.numeric(a$Var1)
ggplot(data = a, aes(x = Var1, y = Freq))+
  geom_line(color="red", size=1) +
  geom_point(size=2, color="red") 

# order's product
train %>% 
  group_by(order_id) %>% 
  summarize(n_items = last(add_to_cart_order)) %>%
  ggplot(aes(x=n_items))+
  geom_histogram(stat="count",fill="red") + 
  geom_rug()+
  coord_cartesian(xlim=c(0,80))

# order's reproduct
train %>% 
  group_by(order_id) %>% 
  summarize(n_items = sum(reordered)) %>%
  ggplot(aes(x=n_items))+
  geom_histogram(stat="count",fill="red") + 
  geom_rug()+
  coord_cartesian(xlim=c(0,80))

#########
tmp <- train %>%  
  group_by(product_id) %>% 
  summarize(count = n()) %>% 
  top_n(10, wt = count) %>%
  left_join(select(train,product_id,product_name),by="product_id") %>%
  arrange(desc(count)) %>%
  unique()
kable(tmp)

ggplot(data = tmp, aes(x = reorder(product_name, - count), y = count)) +
  geom_bar(stat="identity",fill="red")+
  theme(axis.text.x=element_text(angle=90, hjust=1))

tmp <- train %>%  
  group_by(product_id) %>% 
  filter(reordered == 1) %>%
  summarize(count = n()) %>% 
  top_n(10, wt = count) %>%
  left_join(select(train,product_id,product_name),by="product_id") %>%
  arrange(desc(count)) %>%
  unique()
kable(tmp)

tmp <-train %>% 
  group_by(product_id) %>% 
  summarize(proportion_reordered = mean(reordered), n=n()) %>% 
  filter(n>40) %>% 
  arrange(desc(proportion_reordered)) %>% 
  left_join(select(train,product_id,product_name),by="product_id") %>%
  unique()

kable(tmp)

ggplot(data = tmp, aes(x = reorder(product_name, - proportion_reordered), y = proportion_reordered)) +
  geom_bar(stat="identity",fill="red")+
  theme(axis.text.x=element_text(angle=90, hjust=1))+
  coord_cartesian(ylim=c(0.85,0.95))

#
tmp2 <-train %>% 
  filter(reordered == 1) %>%
  group_by(product_id, add_to_cart_order) %>% 
  summarize(count1 = n()) 

tmp = as.data.table(tmp)
tmp2 = as.data.table(tmp2)

setkey(tmp, product_id, add_to_cart_order)
setkey(tmp2, product_id, add_to_cart_order)

a = merge(tmp2, tmp, all.x = T)
a = a[,reorder_rate := count1/count]

b = a %>%
    group_by(add_to_cart_order) %>%
    summarize(reorder_rate = mean(reorder_rate), total_count = n())

ggplot(data = b, aes(x = add_to_cart_order, y = reorder_rate)) +
  geom_line(color = "red")

train %>%
  group_by(days_since_prior_order) %>%
  summarize(mean_reorder = mean(reordered)) %>%
  ggplot(aes(x=days_since_prior_order,y=mean_reorder))+
  geom_bar(stat="identity",fill="red")

#############
tmp <- products %>% group_by(department_id, aisle_id) %>% summarize(n=n())
tmp <- tmp %>% left_join(departments,by="department_id")
tmp <- tmp %>% left_join(aisles,by="aisle_id")


tmp2<-train %>% 
  group_by(product_id) %>% 
  summarize(count=n()) %>% 
  left_join(products,by="product_id") %>% 
  ungroup() %>% 
  group_by(department_id,aisle_id) %>% 
  summarize(sumcount = sum(count)) %>% 
  left_join(tmp, by = c("department_id", "aisle_id")) %>% 
  mutate(onesize = 1)


treemap(tmp2,index=c("department","aisle"),vSize="n",vColor="department",
        palette="Set3",title="",sortID="-sumcount", border.col="#FFFFFF",type="categorical", 
        fontsize.legend = 0,bg.labels = "#FFFFFF")

treemap(tmp2,index=c("department","aisle"),vSize="sumcount",vColor="department",
        palette="Set3",title="", border.col="#FFFFFF",type="categorical", 
        fontsize.legend = 0,bg.labels = "#FFFFFF")
######
tmp = train %>%
  group_by(order_id) %>%
  summarise(mean = mean(reordered), n = n()) %>%
  right_join(filter(train, order_number >2), by = "order_id")

tmp = tmp[,c("user_id", "mean")]
tmp2 <- tmp %>% 
  group_by(user_id) %>% 
  summarize(n_equal = sum(mean==1,na.rm=T), percent_equal = n_equal/n()) %>% 
  filter(percent_equal == 1) %>% 
  arrange(desc(n_equal))

train %>%
  group_by(aisle) %>%
  summarise(n = n()) %>%
  top_n(15, wt = n) %>%
  arrange(desc(n)) %>%
  ggplot(aes(x = reorder(aisle, -n), y = n)) +
  geom_bar(stat="identity",fill="red") +
    theme(axis.text.x=element_text(angle=90, hjust=1))
    
  
tmp = train[, c("department","reordered")]

tmp = as.data.table(table(tmp))

tmp = dcast(tmp, department ~ reordered, value.var = "N")
tmp[, total := `0`+ `1`]
tmp[, reorder_rate := `1` / total]

ggplot(tmp, aes(x = department, y = reorder_rate, group = 1)) +
  geom_line(color="red", size=1) +
  geom_point(size=2, color="red") 
















