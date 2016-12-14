library(data.table)
library(xgboost)
library(Matrix)

cols_in = c('fecha_dato', 'ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1','ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1','ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1','ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1','ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1','ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1','ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1','ind_nomina_ult1',   'ind_nom_pens_ult1', 'ind_recibo_ult1')
train = fread('/Users/xiaofeifei/I/Kaggle/Santander/train_ver2.csv', header = T)
train = train[fecha_dato %in% '2015-06-28']

train[, fecha_dato := as.Date(fecha_dato)]
train[, month_num  := as.integer(format(fecha_dato, format = '%m'))]
train[, fecha_dato := NULL]

# unpivot products (put products in rows from columns)
train = melt(train, measure = 2:25, variable.name = 'product_id')
train [, product_id := as.integer(product_id)]

# pivot month (put month from columns to rows)
train = dcast(train, ncodpers + product_id ~ month_num, value.var = c('value'))
cols_in = c('customer_id','product_id','month_05','month_06')
setnames(train, cols_in)
train[is.na(train)] = 0L

# added = present in current month and absent in prev
june_added = train[(month_06 == 1 & month_05 == 0)]
june_added_customers = unique(june_added$customer_id)

# join add product back to june data
cols_in = c('ncodpers','product_id','month_05','month_06')
setnames(june_added, cols_in)
setkey(june_added,ncodpers)
setkey(train,ncodpers)

# perform the join using the merge function
Result <- merge(june_added,train, all.x=TRUE)
Result = Result[,nomprov:=NULL]
col_names = c("ind_empleado","pais_residencia", "sexo", "ult_fec_cli_1t",
              "indrel_1mes", "tiprel_1mes","indresi", "indext", "conyuemp",
              "canal_entrada", "indfall", "segmento")
col_names = c(ind_empleado,pais_residencia, sexo, ult_fec_cli_1t,
              indrel_1mes, tiprel_1mes,indresi, indext, conyuemp,
              canal_entrada, indfall, segmento)
col_int = c("age", "fecha_alta", "ind_nuevo", "antiguedad", "indrel",
            "tipodom","cod_prov","ind_actividad_cliente","renta")
dtnew <- Result[,(col_int):=lapply(.SD, as.numeric),.SDcols=col_int]
dtnew = dtnew[,(col_names):=lapply(.SD, as.factor),.SDcols=col_names]
dtnew <- dtnew[,("fecha_alta"):=lapply(.SD, as.Date),.SDcols="fecha_alta"]

elapsed_months <- function(end_date, start_date) {
  ed <- as.POSIXlt(end_date)
  sd <- as.POSIXlt(start_date)
  12 * (ed$year - sd$year) + (ed$mon - sd$mon)
}

elapsed_months(Sys.time(),dtnew$fecha_alta[1])
dtnew$fecha_alta = elapsed_months(Sys.time(),dtnew$fecha_alta)

dtnew$label = label
dtnew[which(is.na(dtnew),arr.ind=T)] = ""
dtnew <- dtnew[, -cols_in, with =F]
options(na.action='na.pass')
sparese = sparse.model.matrix(label~.-1, data = dtnew)


bstSparse <- xgboost(data = sparese, label = label-1, 
                     objective = "multi:softprob", num_class = 24,
                     eta = 0.05, max.depth = 8, nthread = 2,
                     subsample = 0.7, colsample_bytree = 0.7,
                     eval_metric = "mlogloss", nrounds = 50)


col = names(dtnew)[1:21]
test1 = test[, col, with = F]
col_names = c("ind_empleado","pais_residencia", "sexo", "ult_fec_cli_1t",
              "indrel_1mes", "tiprel_1mes","indresi", "indext", "conyuemp",
              "canal_entrada", "indfall", "segmento")


dttest <- test1[,(col_names):=lapply(.SD, as.factor),.SDcols=col_names]

dttest <- dttest[,("fecha_alta"):=lapply(.SD, as.Date),.SDcols="fecha_alta"]

elapsed_months <- function(end_date, start_date) {
  ed <- as.POSIXlt(end_date)
  sd <- as.POSIXlt(start_date)
  12 * (ed$year - sd$year) + (ed$mon - sd$mon)
}


dttest$fecha_alta = elapsed_months(Sys.time(),dttest$fecha_alta)


dttest[which(is.na(dttest),arr.ind=T)] = ""
dttest <- test1[,c("cod_prov","renta"):=lapply(.SD, as.numeric),.SDcols=c("cod_prov","renta")]
dttest$label = 1
sparesetest = sparse.model.matrix(label~.-1, data = dttest)

pred <- predict(bstSparse, sparesetest)
head(pred)
product = rep(cols_in, 929615)
pred_data = cbind(pred, product)
pred <- matrix(pred, ncol = 24, byrow = TRUE)
recom <- t(apply(pred, 1, order, decreasing = TRUE)) - 1
top_8 = recom[,1:8]
# for (i in nrow(top_7))
# {
#   top_8[i,8] = paste(cols_in[top_7[i,]], collapse = " ")
# }

func = function(dt)
{
  st = paste(cols_in[as.numeric(dt[,1:7,with = FALSE])], collapse = " ")
  return(st)
}
top_7 = as.data.table(top_7)
top_7[, ..I := .I]

b = top_7[, func(.SD), by = ..I]

recom <- cbind(data.table(b$V1), test[,.(ncodpers)])
colnames(recom) = c("added_products","ncodpers")
write.csv(recom, file = "first_sub.csv", row.names = F, quote=F)
head(recom)
