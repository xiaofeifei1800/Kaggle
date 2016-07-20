# PB score 0.48964
setwd("/Users/xiaofeifei/Downloads/data/grupo")
library(data.table)
# use for kable
library(knitr)
library(Metrics)
print("Reading data")
train <- fread("train.csv", 
               select = c('Semana','Cliente_ID', 'Producto_ID', 'Agencia_ID', 'Ruta_SAK', 'Demanda_uni_equil'))

print("Computing means")
#transform target variable to log(1 + demand) - this makes sense since we're 
#trying to minimize rmsle and the mean minimizes rmse:
train$log_demand = log1p(train$Demanda_uni_equil) 

# split training and validation set
validate = train[Semana == 9]
train = train[Semana != 9]

# calculate mean
mean_total <- mean(train$log_demand) #overall mean
#mean by product
mean_P <-  train[, .(MP = mean(log_demand)), by = .(Producto_ID)]
#mean by product and ruta
mean_PR <- train[, .(MPR = mean(log_demand)), by = .(Producto_ID, Ruta_SAK)] 
#mean by product, client, agencia
mean_PCA <- train[, .(MPCA = mean(log_demand)), by = .(Producto_ID, Cliente_ID, Agencia_ID)]

print("Merging means with training set")
submit <- merge(train, mean_PCA, all.x = TRUE, by = c("Producto_ID", "Cliente_ID", "Agencia_ID"))
submit <- merge(submit, mean_PR, all.x = TRUE, by = c("Producto_ID", "Ruta_SAK"))
submit <- merge(submit, mean_P, all.x = TRUE, by = "Producto_ID")

print("fit the linear function")
fit1 = lm(log_demand~MPCA,data = submit)
fit2 = lm(log_demand~MPR,data = submit)
fit3 = lm(log_demand~MP,data = submit)
fit4 = lm(log_demand~rep(mean_total, dim(train)[1]),data = train)

print("Merging means with test set")
rm(submit)
submit <- merge(validate, mean_PCA, all.x = TRUE, by = c("Producto_ID", "Cliente_ID", "Agencia_ID"))
submit <- merge(submit, mean_PR, all.x = TRUE, by = c("Producto_ID", "Ruta_SAK"))
submit <- merge(submit, mean_P, all.x = TRUE, by = "Producto_ID")

# Now create Predictions column;
submit$Pred <- expm1(submit$MPCA)*fit1$coefficients[2]+fit1$coefficients[1]
submit[is.na(Pred)]$Pred <- expm1(submit[is.na(Pred)]$MPR)*fit2$coefficients[2]+fit2$coefficients[1]
submit[is.na(Pred)]$Pred <- expm1(submit[is.na(Pred)]$MP)*fit3$coefficients[2]+fit3$coefficients[1]
submit[is.na(Pred)]$Pred <- expm1(mean_total)+fit4$coefficients[1]

rmsle(validate[,Demanda_uni_equil], submit[,Pred])
