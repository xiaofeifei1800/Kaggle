library(data.table)
library(ggplot2)
library(dplyr)

train = fread("/Users/xiaofeifei/I/Kaggle/Mercari/train.tsv")
#print(object.size(train), units = 'Mb')

ggplot(data = train, aes(x = log(price+1))) + 
  geom_histogram(fill = 'orangered2') +
  labs(title = 'Histogram of log item price + 1')


train[, .N, by = item_condition_id] %>%
  ggplot(aes(x = as.factor(item_condition_id), y = N/1000)) +
  geom_bar(stat = 'identity', fill = 'cyan2') + 
  labs(x = 'Item condition', y = 'Number of items (000s)', title = 'Number of items by condition category')

train[, .(.N, median_price = median(price)), by = item_condition_id][order(item_condition_id)]
train[, .(.N, mean_price = mean(price)), by = item_condition_id][order(item_condition_id)]

ggplot(data = train, aes(x = as.factor(item_condition_id), y = log(price + 1))) + 
  geom_boxplot(fill = 'cyan2', color = 'darkgrey')

train %>%
  ggplot(aes(x = log(price+1), fill = factor(shipping))) + 
  geom_density(adjust = 2, alpha = 0.6) + 
  labs(x = 'Log price', y = '', title = 'Distribution of price by shipping')

train[, .(median_price = min(price)), by = brand_name] %>%
  head(25) %>%
  ggplot(aes(x = reorder(brand_name, median_price), y = median_price)) + 
  geom_point(color = 'cyan2') + 
  scale_y_continuous(labels = scales::dollar) + 
  coord_flip() +
  labs(x = '', y = 'Median price', title = 'Top 25 most expensive brands') 

train[train$price==0][2:3]

length(unique(train$category_name))

train[, .(median = median(price)), by = category_name][order(median, decreasing = TRUE)][1:30] %>%
  ggplot(aes(x = reorder(category_name, median), y = median)) + 
  geom_point(color = 'orangered2') + 
  coord_flip() + 
  labs(x = '', y = 'Median price', title = 'Median price by item category (Top 30)') + 
  scale_y_continuous(labels = scales::dollar)

train[, c("level_1_cat", "level_2_cat", "level_3_cat") := tstrsplit(train$category_name, split = "/", keep = c(1,2,3))]

length(unique(train$level_1_cat))
length(unique(train$level_2_cat))
length(unique(train$level_3_cat))

train %>%
  ggplot(aes(x = level_1_cat, y = log(price+1))) + 
  geom_boxplot(fill = 'cyan2', color = 'darkgrey') + 
  coord_flip() + 
  labs(x = '', y = 'Log price + 1', title = 'Boxplot of price by top-level category')
