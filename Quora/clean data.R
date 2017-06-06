library(data.table)
train = fread("/Users/xiaofeifei/I/Kaggle/Quora/x_train_0.15.csv")
train1 = fread("/Users/xiaofeifei/I/Kaggle/Quora/x_test_git.csv")

drop = train1$feature[51:70]
train = train[, -drop, with =F]
data = train
train1 = cbind(train, train1)
train1 = train1[,102:207, with = FALSE]
train = data[test_id==-1]

names = colnames(test)
names = paste0(names, "_c")
colnames(test) = names

train_name = colnames(train)
train1_name = colnames(train1)
add_columns = train_name[!train_name %in%train1_name]
add_columns = add_columns[-c(1:7)]
add_columns = add_columns[-1]

train1 = cbind(train1, train[,add_columns, with=FALSE])
train1[,euclidean_distance:=NULL]
train1[, abs_diff := abs(diff_len)]
train1[, ratio := len_q1/len_q2]

test[, c("question1","question2","len_q1", "len_q2", "diff_len"):= NULL]
test = fread("/Users/xiaofeifei/I/Kaggle/Quora/x_test_0.15.csv")
test1 = fread("/Users/xiaofeifei/I/Kaggle/Quora/x_test_git.csv")

test1 = test1[,102:207, with = FALSE]

test[,V1:=NULL]
test1 = cbind(test, test1)
test_name = colnames(test)
test1_name = colnames(test1)
add_columns = test_name[!test_name %in%test1_name]
add_columns = add_columns[-c(1:3)]
add_columns = add_columns[-2]
test1 = cbind(test1, test[,add_columns,with = FALSE])
test1[,jaccard_distance:=NULL]
test1[, abs_diff := abs(diff_len)]
test1[, ratio := len_q1/len_q2]

train1 = cbind(train[,c("qid1", "qid2","is_duplicate","id"), with = FALSE], train1)
test1 = cbind(test[,c("qid1", "qid2", "test_id"), with = FALSE], test1)

kcore = fread("/Users/xiaofeifei/I/Kaggle/Quora/question_max_kcores.csv")
train1 = train1[,-19]
train1$qid1 = as.numeric(train1$qid1)
train1$qid2 = as.numeric(train1$qid2)

colnames(kcore) = c("qid1", "max_kcore")
setkey(kcore, qid1)
setkey(train1, qid1)
train1 = merge(train1, kcore, all.x = T)
train1$id = as.numeric(train1$id)
train1 = train1[order(id)]

test1$qid1 = as.numeric(test1$qid1)
test1$qid2 = as.numeric(test1$qid2)

colnames(kcore) = c("qid1", "max_kcore")
setkey(kcore, qid1)
setkey(test1, qid1)
test1 = merge(test1, kcore, all.x = T)

test1$test_id = as.numeric(test1$test_id)
test1 = test1[order(test_id)]


train1[,qid1 := NULL]
train1[,qid2 := NULL]
train1[,id := NULL]

write.csv(y2, file = "/Users/xiaofeifei/I/Kaggle/Quora/y_train.csv", row.names = F)

train1[,is_duplicate := NULL]
fwrite(train1, file = "/Users/xiaofeifei/I/Kaggle/Quora/x_train_git_r.csv", row.names = F)

test1[,qid1 := NULL]
test1[,qid2 := NULL]
test1[,test_id := NULL]
fwrite(test1, file = "/Users/xiaofeifei/I/Kaggle/Quora/x_test_git.csv", row.names = F)

train1 = train1[,V1:=NULL]
colnames(train1) = c('test_id', 'question1', 'question2', 'qid1', 'qid2')

y = fread("/Users/xiaofeifei/I/Kaggle/Quora/y_train.csv")
y[,V1 := NULL]
y1 = fread("/Users/xiaofeifei/I/Kaggle/Quora/y_train copy.csv", header = T)

gamma_0 = 1.30905513329
gamma_1 = 0.472008228977

x = train$is_duplicate
pred = gamma_1*x/(gamma_1*x + gamma_0*(1 - x))

fwrite(train, file = "/Users/xiaofeifei/I/Kaggle/Quora/predictions.csv", row.names = F)

# train1[,tfidf_wm:=NULL]
# train1[,q2_which:=NULL]
# train1[,what_both:=NULL]
# train1[,who_both:=NULL]
# train1[,which_both:=NULL]
# train1[,where_both:=NULL]
# train1[,q1_when:=NULL]
# train1[,trigram_in_title:=NULL]
