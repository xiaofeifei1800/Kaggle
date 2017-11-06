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