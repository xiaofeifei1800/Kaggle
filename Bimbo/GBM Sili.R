setwd("C:\\Users\\Sili Fan\\Desktop\\KAGGLE\\Bimbo Competition with Guoxin\\")
library(data.table)
library(sqldf)

#2 way count
my.f2cnt<-function(th2, vn1, vn2, filter=TRUE) {
  df<-data.table(f1=th2[,which(colnames(th2)%in%vn1),with = FALSE],f2=th2[,which(colnames(th2)%in%vn2),with = FALSE],filter=filter)
  colnames(df) = c("f1","f2","filter")
  return(df[,':='(cnt=.N),by=.(f1,f2)][,cnt])
}

#3 way count
my.f3cnt<-function(th2, vn1, vn2, vn3, filter=TRUE) {
  df<-data.table(f1=th2[,which(colnames(th2)%in%vn1),with = FALSE],
                 f2=th2[,which(colnames(th2)%in%vn2),with = FALSE],
                 f2=th2[,which(colnames(th2)%in%vn3),with = FALSE],
                 filter=filter)
  colnames(df) = c("f1","f2","f3","filter")
  return(df[,':='(cnt=.N),by=.(f1,f2,f3)][,cnt])
}




############################################
# import data.

T1 <- fread('train.csv')
H1 <- fread('test.csv')
T1$id = 10000000 + 1:nrow(T1)
H1$Demanda_uni_equil <- -1

TH1 <- rbindlist(list(set(T1, j=which(!colnames(T1)%in%colnames(H1)), value=NULL ), H1))
TH1[,ws:=0]
TH1$ws[TH1$id>10000000] = 1

#######################################################################################################################
#convert variables into sequential IDs
TH1 <- TH1[,smn_f := as.factor(Semana)]
TH1 <- TH1[,smn_id := as.integer(smn_f)]
TH1 <- TH1[,ag_f := as.factor(Agencia_ID)]
TH1 <- TH1[,ag_id := as.integer(ag_f)]
TH1 <- TH1[,cn_f := as.factor(Canal_ID)]
TH1 <- TH1[,cn_id := as.integer(cn_f)]
TH1 <- TH1[,rt_f := as.factor(Ruta_SAK)]
TH1 <- TH1[,rt_id := as.integer(rt_f)]
TH1 <- TH1[,clt_f := as.factor(Cliente_ID)]
TH1 <- TH1[,clt_id := as.integer(clt_f)]
TH1 <- TH1[,prd_f := as.factor(Producto_ID)]
TH1 <- TH1[,prd_id := as.integer(prd_f)]


TH1 <- TH1[,clt_prd_cnt:=my.f2cnt(TH1, "Cliente_ID", "Producto_ID")]

TH1 <- TH1[,clt_prd_f := as.factor(clt_id*10000000000 + prd_id)] #saved.
TH1 <- TH1[,clt_prd_id := as.integer(clt_prd_f)]

#######################################################################################################################
#one way count
mean_t<-TH1[,mean(Demanda_uni_equil)] 
for(ii in 1:6) {
  print(names(TH1)[ii])
  TH1[,x:=TH1[,ii, with = FALSE]]
  # tmp<-sqldf("select cnt from TH1 a left join sum1 b on a.x=b.x")
  TH1[, paste(names(TH1)[ii], "_cnt", sep=""):=merge(TH1,TH1[,.N,by=x],by="x")[,N]]
}




#######################################################################################################################
#selected 2-way count
TH1[,smn_ag_cnt:=my.f2cnt(TH1, "Semana", "Agencia_ID")]
TH1[,smn_cn_cnt:=my.f2cnt(TH1, "Semana", "Canal_ID")]
TH1[,smn_rt_cnt:=my.f2cnt(TH1, "Semana", "Ruta_SAK")]
TH1[,smn_clt_cnt:=my.f2cnt(TH1, "Semana", "Cliente_ID")]
TH1[,smn_prd_cnt:=my.f2cnt(TH1, "Semana", "Producto_ID")]

TH1[,ag_cn_cnt:=my.f2cnt(TH1, "Agencia_ID", "Canal_ID")]
TH1[,ag_rt_cnt:=my.f2cnt(TH1, "Agencia_ID", "Ruta_SAK")]
TH1[,ag_clt_cnt:=my.f2cnt(TH1, "Agencia_ID", "Cliente_ID")]
TH1[,ag_prd_cnt:=my.f2cnt(TH1, "Agencia_ID", "Producto_ID")]

TH1[,cn_rt_cnt:=my.f2cnt(TH1, "Canal_ID", "Ruta_SAK")]
TH1[,cn_clt_cnt:=my.f2cnt(TH1, "Canal_ID", "Cliente_ID")]
TH1[,cn_prd_cnt:=my.f2cnt(TH1, "Canal_ID", "Producto_ID")]

TH1[,rt_clt_cnt:=my.f2cnt(TH1, "Ruta_SAK", "Cliente_ID")]
TH1[,rt_prd_cnt:=my.f2cnt(TH1, "Ruta_SAK", "Producto_ID")]

TH1[,clt_prd_cnt:=my.f2cnt(TH1, "Cliente_ID", "Producto_ID")]




#######################################################################################################################
#selected 3 way count
TH1$smn_ag_cn_cnt<-my.f3cnt(TH1, "Semana", "Agencia_ID", "Canal_ID")
TH1$smn_ag_rt_cnt<-my.f3cnt(TH1, "Semana", "Agencia_ID", "Ruta_SAK")
TH1$smn_ag_clt_cnt<-my.f3cnt(TH1, "Semana", "Agencia_ID", "Cliente_ID")
TH1$smn_ag_prd_cnt<-my.f3cnt(TH1, "Semana", "Agencia_ID", "Producto_ID")

TH1$ag_cn_rt_cnt<-my.f3cnt(TH1, "Agencia_ID", "Canal_ID", "Ruta_SAK")
TH1$ag_cn_clt_cnt<-my.f3cnt(TH1, "Agencia_ID", "Canal_ID", "Cliente_ID")
TH1$ag_cn_prd_cnt<-my.f3cnt(TH1, "Agencia_ID", "Canal_ID", "Producto_ID")


TH1$cn_ct_clt_cnt<-my.f3cnt(TH1, "Canal_ID", "Ruta_SAK", "Cliente_ID")
TH1$cn_ct_prd_cnt<-my.f3cnt(TH1, "Canal_ID", "Ruta_SAK", "Producto_ID")



