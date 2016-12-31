library(data.table)

a = events[,1:3,with = FALSE]
fwrite(a, file.path = "new_train.csv")
