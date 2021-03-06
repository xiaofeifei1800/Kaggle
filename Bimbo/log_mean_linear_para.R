setwd("/Users/xiaofeifei/Downloads/data/grupo")
library(data.table)
# use for kable
library(knitr)
library(Metrics)
print("Reading data")
train <- fread("train.csv", 
               select = c('Semana','Cliente_ID', 'Producto_ID', 'Agencia_ID', 'Ruta_SAK', 'Demanda_uni_equil'))

test <- fread("test.csv", 
              select = c('id', 'Cliente_ID', 'Producto_ID', 'Agencia_ID', 'Ruta_SAK'))

print("Computing means")
#transform target variable to log(1 + demand) - this makes sense since we're 
#trying to minimize rmsle and the mean minimizes rmse:
train$log_demand = log1p(train$Demanda_uni_equil) 

# split training and validation set
validate = train[Semana == 8]
validate = rbind(validate, train[Semana == 9])
train_test = train
# week7 = train[Semana == 7]
# train = rbind(week6,week7)
# calculate mean
mean_total <- mean(train_test$log_demand) #overall mean
#mean by product
mean_P <-  train_test[, .(MP = mean(log_demand)), by = .(Producto_ID)]
#mean by product and ruta
mean_PR <- train_test[, .(MPR = mean(log_demand)), by = .(Producto_ID, Ruta_SAK)] 
#mean by product, client, agencia
mean_PCA <- train_test[, .(MPCA = mean(log_demand)), by = .(Producto_ID, Cliente_ID, Agencia_ID)]

print("Merging means with training set")
submit <- merge(validate, mean_PCA, all.x = TRUE, by = c("Producto_ID", "Cliente_ID", "Agencia_ID"))
submit <- merge(submit, mean_PR, all.x = TRUE, by = c("Producto_ID", "Ruta_SAK"))
submit <- merge(submit, mean_P, all.x = TRUE, by = "Producto_ID")

print("fit the linear function")
fit1 = lm(log_demand~MPCA,data = submit)
fit2 = lm(log_demand~MPR,data = submit)
fit3 = lm(log_demand~MP,data = submit)
fit4 = lm(log_demand~rep(mean_total, dim(validate)[1]),data = submit)

print("Merging means with test set")
rm(submit)
submit <- merge(test, mean_PCA, all.x = TRUE, by = c("Producto_ID", "Cliente_ID", "Agencia_ID"))
submit <- merge(submit, mean_PR, all.x = TRUE, by = c("Producto_ID", "Ruta_SAK"))
submit <- merge(submit, mean_P, all.x = TRUE, by = "Producto_ID")

# Now create Predictions column;
submit$Pred <- expm1(submit$MPCA)*fit1$coefficients[2]+fit1$coefficients[1]
submit[is.na(Pred)]$Pred <- expm1(submit[is.na(Pred)]$MPR)*fit2$coefficients[2]+fit2$coefficients[1]
submit[is.na(Pred)]$Pred <- expm1(submit[is.na(Pred)]$MP)*fit3$coefficients[2]+fit3$coefficients[1]
submit[is.na(Pred)]$Pred <- expm1(mean_total)+fit4$coefficients[1]

print("Write out submission file")
# now relabel columns ready for creating submission
setnames(submit,"Pred","Demanda_uni_equil")
setwd("/Users/xiaofeifei/GitHub/Kaggle_Bimbo/submit")
# Any results you write to the current directory are saved as output.
write.csv(submit[,.(id,Demanda_uni_equil)],"submit_mean_by_week89.csv", row.names = FALSE)
print("Done!")