library(dplyr)
library(tidyr)
library(data.table)
train = fread("/.../Santander/data/train_ver2.csv", header = T)

train$fecha_dato <- as.POSIXct(strptime(train$fecha_dato,format="%Y-%m-%d"))
train$fecha_alta <- as.POSIXct(strptime(train$fecha_alta,format="%Y-%m-%d"))
train$month <- month(train$fecha_dato)

#dealing with age, here are a large number of university aged students, and then another peak around
#middle-age. Let’s separate the distribution and move the outliers to the mean of the closest one.

train$age[(train$age < 18)] <- mean(train$age[(train$age >= 18) & (train$age <=30)],na.rm=TRUE)
train$age[(train$age > 100)] <- mean(train$age[(train$age >= 30) & (train$age <=100)],na.rm=TRUE)
train$age[is.na(train$age)] <- median(train$age,na.rm=TRUE)
train$age <- round(train$age)

# ind_nuevo is 	New customer Index. 1 if the customer registered in the last 6 months.
# based on the data exploration on kaggle, all of those NA(7326) have history shorter
# than 6, so give 1 to them.
train$ind_nuevo[is.na(train$ind_nuevo)] = 1 

# antiguedad, customer seniority (in months). for the same people, give them minimum seniority
train$antiguedad[is.na(train$antiguedad)] <- min(train$antiguedad,na.rm=TRUE)
train$antiguedad[train$antiguedad<0] = 0

# fecha_alta, the date in which the customer became as the first holder of a contract in the bank
# Some entries don’t have the date they joined the company. Just give them something 
# in the middle of the pack
train$fecha_alta[is.na(train$fecha_alta)] = median(train$fecha_alta,na.rm=TRUE)

#indrel 1 (First/Primary), 99 (Primary customer during the month but not at the end of the month)
# most are 1, gives 1 to the miss value
train$indrel[is.na(train$indrel)] = 1

# cod_prov, Province code (customer's address)
# nomprov, Province name.  Duplicate information, keep on is enough
train[,-cod_prov]

# tipodom, Addres type. 1, primary address. Not useful
train[,-tipodom]

# ind_actividad_cliente, Activity index (1, active customer; 0, inactive customer)
# the NA are coming from those 7326 customors, assume they are new, so they are active, gives
# 1 to them
train$ind_actividad_cliente[is.na(train$ind_actividad_cliente)] = 1

#nomprov, Province name
train$nomprov[train$nomprov==""] <- "UNKNOWN"

#renta, Gross income of the household
# sign the income based on the location
new.incomes <-train %>%
  select(nomprov) %>%
  merge(train %>%
          group_by(nomprov) %>%
          summarise(med.income=median(renta,na.rm=TRUE)),by="nomprov") %>%
  select(nomprov,med.income) %>%
  arrange(nomprov)
train <- arrange(train,nomprov)
train$renta[is.na(train$renta)] <- new.incomes$med.income[is.na(train$renta)]
rm(new.incomes)

train$renta[is.na(train$renta)] <- median(train$renta,na.rm=TRUE)

#
train$intrainall[train$intrainall==""]               <- "N"
train$tiprel_1mes[train$tiprel_1mes==""]       <- "A"
train$indrel_1mes[train$indrel_1mes==""] <- "1"
train$indrel_1mes[train$indrel_1mes=="P"] <- "5"
train$indrel_1mes <- as.factor(as.integer(train$indrel_1mes))

train$pais_residencia[train$pais_residencia==""] <- "UNKNOWN"
train$sexo[train$sexo==""]                       <- "UNKNOWN"
train$ult_fec_cli_1t[train$ult_fec_cli_1t==""]   <- "UNKNOWN"
train$ind_empleado[train$ind_empleado==""]       <- "UNKNOWN"
train$indext[train$indext==""]                   <- "UNKNOWN"
train$indresi[train$indresi==""]                 <- "UNKNOWN"
train$conyuemp[train$conyuemp==""]               <- "UNKNOWN"
train$segmento[train$segmento==""]               <- "UNKNOWN"


fwrite(train, "/.../Santander/data/train_clean.csv")


