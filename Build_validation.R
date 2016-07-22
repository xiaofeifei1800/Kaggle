# PB score 0.48964
setwd("/Users/xiaofeifei/Downloads/data/grupo")
library(data.table)
# use for kable
library(knitr)
library(dplyr)
library(Metrics)
print("Reading data")
train <- fread("train.csv", 
               select = c('Semana','Cliente_ID', 'Producto_ID', 'Agencia_ID', 'Ruta_SAK', 'Demanda_uni_equil'))
train$id = 10000000 + 1:nrow(train)
train$log_demand = log1p(train$Demanda_uni_equil) 
# 10% of the training
validation_size = dim(train)[1]*0.1
result = numeric()
for (i in 0:9) 
{  
  print("Computing means")
  #transform target variable to log(1 + demand) - this makes sense since we're 
  #trying to minimize rmsle and the mean minimizes rmse:
  
  # split training and validation set
  validate = train[train$Semana == 4]
  # validate = rbind(validate, train[Semana == i+1])
  # setkey(validate, id)
  # setkey(train, id)
  test_train = train[train$Semana == 3]
  # train = train[Semana != 1+1]
  
  # calculate mean
  mean_total <- mean(test_train$log_demand) #overall mean
  #mean by product
  mean_P <-  test_train[, .(MP = mean(log_demand)), by = .(Producto_ID)]
  #mean by product and ruta
  mean_PR <- test_train[, .(MPR = mean(log_demand)), by = .(Producto_ID, Ruta_SAK)] 
  #mean by product, client, agencia
  mean_PCA <- test_train[, .(MPCA = mean(log_demand)), by = .(Producto_ID, Cliente_ID, Agencia_ID)]
  
  print("Merging means with training set")
  submit <- merge(validate, mean_PCA, all.x = TRUE, by = c("Producto_ID", "Cliente_ID", "Agencia_ID"))
  submit <- merge(submit, mean_PR, all.x = TRUE, by = c("Producto_ID", "Ruta_SAK"))
  submit <- merge(submit, mean_P, all.x = TRUE, by = "Producto_ID")
  
  submit$Pred <- expm1(submit$MPCA)*0.826+0.45
  submit[is.na(Pred)]$Pred <- expm1(submit[is.na(Pred)]$MPR)*0.788+0.09
  submit[is.na(Pred)]$Pred <- expm1(submit[is.na(Pred)]$MP)*1.05+0.6
  submit[is.na(Pred)]$Pred <- expm1(mean_total) + 1.08
  
  # print("fit the linear function")
  # fit1 = lm(log_demand~MPCA,data = submit)
  # fit2 = lm(log_demand~MPR,data = submit)
  # fit3 = lm(log_demand~MP,data = submit)
  # fit4 = lm(log_demand~rep(mean_total, dim(train)[1]),data = train)
  # 
  # print("Merging means with test set")
  # rm(submit)
  # submit <- merge(validate, mean_PCA, all.x = TRUE, by = c("Producto_ID", "Cliente_ID", "Agencia_ID"))
  # submit <- merge(submit, mean_PR, all.x = TRUE, by = c("Producto_ID", "Ruta_SAK"))
  # submit <- merge(submit, mean_P, all.x = TRUE, by = "Producto_ID")
  # 
  # # Now create Predictions column;
  # submit$Pred <- expm1(submit$MPCA)*fit1$coefficients[2]+fit1$coefficients[1]
  # submit[is.na(Pred)]$Pred <- expm1(submit[is.na(Pred)]$MPR)*fit2$coefficients[2]+fit2$coefficients[1]
  # submit[is.na(Pred)]$Pred <- expm1(submit[is.na(Pred)]$MP)*fit3$coefficients[2]+fit3$coefficients[1]
  # submit[is.na(Pred)]$Pred <- expm1(mean_total)+fit4$coefficients[1]
  i = i + 1
  result[i] = rmsle(validate[,Demanda_uni_equil], submit[,Pred])
  i = i - 1
}

val = train[train$Semana == 4]
train = train[train$Semana == 3]
new = train[, .(mean = mean(Demanda_uni_equil),
          median = median(as.double(Demanda_uni_equil)), #data.table doesn't like ints for medians
          log1p_mean = expm1(mean(log1p(Demanda_uni_equil)))), by = .(Producto_ID)]

result = merge(val, new, all.x = TRUE, by = "Producto_ID") -> merged_val #merge with validation set by product
kable(head(merged_val))
mask = !(is.na(merged_val$mean))
y_val = merged_val$Demanda_uni_equil[mask]
results = c(rmsle(y_val, merged_val$mean[mask]), 
            rmsle(y_val, merged_val$median[mask]),
            rmsle(y_val, merged_val$log1p_mean[mask]))