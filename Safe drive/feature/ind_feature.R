library(data.table)
library(lightgbm)
library(ggplot2)


# ind feature
train = fread('/Users/xiaofeifei/I/Kaggle/Safe drive/train.csv')
ind = train[,1:23]

# two way interact
ggplot(train, aes(ind$ps_ind_07_bin)) +
  geom_bar(aes(fill = as.factor(ind$target)))

a = as.matrix(table(ind$ps_ind_06_bin, by = ind$target))
a[,2]/a[,1]
#         0          1 
#0.04338925 0.02937559 
two_inter = matrix(NA, nrow = 1, ncol = 5)
for (i in 3:14)
{
  for (j in (i+1):15)
  {
    a = as.matrix(table(ind[,i], ind[,j],by = ind$target))
    var = a[5:8]/a[1:4]
    var = append(var, (i-2)*100+(j-2))
    two_inter = rbind(two_inter,var)
    
  }
  
  print(i)
}

two_inter = two_inter[-1,]
two_inter = as.data.table(two_inter)
two_inter$index = 1:78
# use two way to select best interaction
# 1.var, 2.max value
temp = (apply(two_inter[,1:4], 1, var, na.rm = T))
two_inter$var = temp
#25,45,4 12,46,23,9 11


# covert onehot encoding back
# 8:11
sum(apply(ind[,8:11],1,sum) == 1)
for (i in 8:11)
{
  ind[ind[,i] == 1,i] = i-7
}
ind = as.data.table(ind)

# add feature
ind[,ps_ind_cat := ps_ind_06_bin+ps_ind_07_bin+ps_ind_08_bin+ps_ind_09_bin]
ind[,ps_two_25 := paste0(ps_ind_02_cat, ps_ind_05_cat)]
ind[,ps_two_45 := paste0(ps_ind_04_cat, ps_ind_05_cat)]
ind[,ps_two_412 := paste0(ps_ind_04_cat, ps_ind_12_bin)]
ind[,ps_two_46 := paste0(ps_ind_04_cat, ps_ind_06_bin)]
ind[,ps_two_23 := paste0(ps_ind_02_cat, ps_ind_03)]
ind[,ps_two_911 := paste0(ps_ind_09_bin, ps_ind_11_bin)]

ind = temp

changeCols = c("ps_two_25", "ps_two_45", "ps_two_412", "ps_two_46",
               "ps_two_23", "ps_two_911")
ind[,(changeCols):= lapply(.SD, as.factor), .SDcols = changeCols]
ind[,(changeCols):= lapply(.SD, as.numeric), .SDcols = changeCols]

ind_feature = ind[,c(1,24,25,26,27,28,29,30)]


# model choose
hold = sample(1:dim(ind)[1],dim(ind)[1]*0.1)
dtrain = as.matrix(ind[-hold, 3:30])
dtest = as.matrix(ind[hold, 3:30])
train_y = ind$target[-hold]
test_y = ind$target[hold]

dtrain <- lgb.Dataset(data = dtrain,
                      label = train_y)
dtest <- lgb.Dataset(data = dtest,
                     label = test_y)

normalizedGini <- function(preds, dtrain) {
  Gini <- function(preds, dtrain) {
    a <- getinfo(dtrain, "label")
    p = preds
    temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
    temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
    population.delta <- 1 / length(a)
    total.losses <- sum(a)
    null.losses <- rep(population.delta, length(a)) # Hopefully is similar to accumulatedPopulationPercentageSum
    accum.losses <- temp.df$actual / total.losses # Hopefully is similar to accumulatedLossPercentageSum
    gini.sum <- cumsum(accum.losses - null.losses) # Not sure if this is having the same effect or not
    sum(gini.sum) / length(a)
  }
  Gini_train <- function(dtrain) {
    a <- getinfo(dtrain, "label")
    p = getinfo(dtrain, "label")
    temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
    temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
    population.delta <- 1 / length(a)
    total.losses <- sum(a)
    null.losses <- rep(population.delta, length(a)) # Hopefully is similar to accumulatedPopulationPercentageSum
    accum.losses <- temp.df$actual / total.losses # Hopefully is similar to accumulatedLossPercentageSum
    gini.sum <- cumsum(accum.losses - null.losses) # Not sure if this is having the same effect or not
    sum(gini.sum) / length(a)
  }
  
  out = Gini(preds,dtrain) / Gini_train(dtrain)
  
  return(list(name = "NormalizedGini", value = out, higher_better = TRUE))
}

# original
#[73]:	train's NormalizedGini:0.30443	valid's NormalizedGini:0.251338 
#[56]:	train's NormalizedGini:0.294638	valid's NormalizedGini:0.26192 
model <- lgb.train(list(objective = "binary",
                        learning_rate = 0.1,
                        max_depth = 6,
                        is_unbalance = TRUE,
                        num_threads = 6
                  
                        
),
dtrain, eval = normalizedGini,nrounds = 100, 
valids = list(train = dtrain, valid = dtest), early_stopping_round = 10)


tree_imp <- lgb.importance(model, percentage = TRUE)
lgb.plot.importance(tree_imp, top_n = 10, measure = "Gain")

