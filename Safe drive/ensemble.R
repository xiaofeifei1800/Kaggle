library(data.table)

en1 = fread("/Users/xiaofeifei/I/Kaggle/Safe drive/ensemble/FattyKimJungUn.csv")
en2 = fread("/Users/xiaofeifei/I/Kaggle/Safe drive/ensemble/Froza_and_Pascal.csv")
en3 = fread("/Users/xiaofeifei/I/Kaggle/Safe drive/ensemble/lgb_submission.csv")
en4 = fread("/Users/xiaofeifei/I/Kaggle/Safe drive/ensemble/stacked_1.csv")
en5 = fread("/Users/xiaofeifei/I/Kaggle/Safe drive/ensemble/xgb_submission.csv")
en6 = fread("/Users/xiaofeifei/I/Kaggle/Safe drive/ensemble/sub.csv")

en6$target = log(en6$target) * 5

en1$target = log(en1$target)
en2$target = log(en2$target)
en3$target = log(en3$target)
en4$target = log(en4$target)
en5$target = log(en5$target)

final = (en1$target+en2$target+en3$target+en4$target+en5$target+en6$target)/10
final = exp(final)

submit = fread('/Users/xiaofeifei/I/Kaggle/Safe drive/sample_submission.csv')
submit$target = final

fwrite(submit, file = '/Users/xiaofeifei/I/Kaggle/Safe drive/ensemble_online.csv', row.names = F)





