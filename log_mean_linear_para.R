setwd("/Users/xiaofeifei/Downloads/data/grupo")
library(data.table)

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
mean_total <- median(train$log_demand) #overall mean
#mean by product
mean_P <-  train[, .(MP = median(log_demand)), by = .(Producto_ID)]
#mean by product and ruta
mean_PR <- train[, .(MPR = median(log_demand)), by = .(Producto_ID, Ruta_SAK)] 
#mean by product, client, agencia
mean_PCA <- train[, .(MPCA = median(log_demand)), by = .(Producto_ID, Cliente_ID, Agencia_ID)]

print("Merging means with training set")
submit <- merge(train, mean_PCA, all.x = TRUE, by = c("Producto_ID", "Cliente_ID", "Agencia_ID"))
submit <- merge(submit, mean_PR, all.x = TRUE, by = c("Producto_ID", "Ruta_SAK"))
submit <- merge(submit, mean_P, all.x = TRUE, by = "Producto_ID")

print("fit the linear function")
fit1 = lm(log_demand~MPCA,data = submit)
fit2 = lm(log_demand~MPR,data = submit)
fit3 = lm(log_demand~MP,data = submit)
fit4 = lm(log_demand~rep(mean_total, dim(train)[1]),data = submit)

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
# Any results you write to the current directory are saved as output.
write.csv(submit[,.(id,Demanda_uni_equil)],"submit_mean_by_Agency_Ruta_Client.csv", row.names = FALSE)
print("Done!")