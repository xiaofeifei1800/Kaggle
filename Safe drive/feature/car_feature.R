library(data.table)
library(infotheo)

train = fread('/Users/xiaofeifei/I/Kaggle/Safe drive/train.csv')
test = fread('/Users/xiaofeifei/I/Kaggle/Safe drive/test.csv')
target = train$target

train$target = NULL
train$data_index = 'train'
test$data_index = 'test'
data = rbind(train, test)
rm(train,test)

a = c(1,23:38,59)
ind = data[,a,with = FALSE]

#ps_car_11_cat, car maker

# car_12 is cc, car_13 mileage in a fixed period, car_14 is weight car 15 is age
ind$cc = ind$ps_car_12^2
ind$mileage = (ind$ps_car_13*220)^2
ind$weight = ind$ps_car_14^2
ind$age = ind$ps_car_15^2

ind$mile_age = ind$mileage/(ind$age+1)
ind$mile_wei = ind$mileage*ind$weight
ind$mile_cc = ind$mileage*ind$cc

ind$target = target
ind = as.data.frame(ind)
two_inter = matrix(NA, nrow = 1, ncol = 2)
for (i in 2:11)
{
  for (j in (i+1):12)
  {
    a = as.matrix(table(ind[,i], ind[,j],by = ind$target))
    len = dim(a)[1]/2
    var1 = a[(len+1):dim(a)[1]]/a[1:len]
    var1 = var(var1, na.rm = T)
    var1 = append(var1, (i-1)*100+(j-1))
    two_inter = rbind(two_inter,var1)
  }
  
  print(i)
}

#16, 47, 12
ind = as.data.table(ind)
ind[,car_two_16 := paste0(ps_car_01_cat, ps_car_06_cat)]
ind[,car_two_47 := paste0(ps_car_04_cat, ps_car_07_cat)]
ind[,car_two_12 := paste0(ps_car_01_cat, ps_car_02_cat)]
ind$target = NULL

ind$ps_car_12 = NULL
ind$ps_car_13 = NULL
ind$ps_car_14 = NULL
ind$ps_car_15 = NULL

changeCols = c("car_two_16", "car_two_47", "car_two_12")
ind[,(changeCols):= lapply(.SD, as.factor), .SDcols = changeCols]
ind[,(changeCols):= lapply(.SD, as.numeric), .SDcols = changeCols]


data = cbind(data, ind[,19:28])
train = data[data$data_index =='train']
test = data[data$data_index =='test']

train$mile_age = NULL
test$mile_age = NULL

train$target = target

fwrite(train, file = '/Users/xiaofeifei/I/Kaggle/Safe drive/new_train.csv')
fwrite(test, file = '/Users/xiaofeifei/I/Kaggle/Safe drive/new_test.csv')


