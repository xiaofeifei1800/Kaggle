library(data.table)
library(ggplot2)
library(MASS)

# ind feature
train = fread('/Users/xiaofeifei/I/Kaggle/Safe drive/train.csv')
test = fread('/Users/xiaofeifei/I/Kaggle/Safe drive/test.csv')
target = train$target
train$target = NULL
train$data_index = 'train'
test$data_index = 'test'
data = rbind(train, test)
rm(train,test)

ind = data[,1:22]
ind$data_index = data$data_index

ind_train = train[,1:23]

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

data = cbind(data, ind[,24:30])
# add lda left to python



