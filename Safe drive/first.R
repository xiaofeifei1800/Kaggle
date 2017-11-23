library(data.table)
library(ggplot2)
library(lightgbm)
library(caret)
train = fread('/Users/xiaofeifei/I/Kaggle/Safe drive/train.csv')

## look at the distribution of target
hist(train$target)
table(train$target)[2]/dim(train)[1]
# unbalance data, 0.035% positive data

# looks like most are cate data, let's do some plot
ggplot(train, aes(train$ps_calc_15_bin)) +
  geom_bar(aes(fill = as.factor(train$target)))

# three major feature, user, user_reg, car, calc
set.seed(1)
hold = sample(1:dim(train)[1],dim(train)[1]*0.1)
dtrain = as.matrix(train[-hold, 3:59, with = FALSE])
dtest = as.matrix(train[hold, 3:59, with = FALSE])
train_y = train$target[-hold]
test_y = train$target[hold]

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


model <- lgb.train(list(objective = "binary",
                        learning_rate = 0.1,
                        max_depth = 6,
                        is_unbalance = TRUE,
                        num_threads = 6,
                        categorical_feature = c(4, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34)
                        ),
                   dtrain, eval = normalizedGini,nrounds = 100, 
                   valids = list(train = dtrain, valid = dtest), early_stopping_round = 10)
 
test = fread('/Users/xiaofeifei/I/Kaggle/Safe drive/test.csv')

dtest = as.matrix(test[, 2:74, with = FALSE])
pred = predict(model, dtest)
submit = fread('/Users/xiaofeifei/I/Kaggle/Safe drive/sample_submission.csv')
submit$target = pred

fwrite(submit, file = '/Users/xiaofeifei/I/Kaggle/Safe drive/all_feature.csv', row.names = F)





