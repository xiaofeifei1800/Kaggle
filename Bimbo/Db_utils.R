library(rmongodb)
library(rjson)
library(RSQLite)
mongo.is.connected(mongo)

insert_dataframe = function(data_path, collection_name)
{
  mongo = mongo.create(host = "localhost")
  if (mongo.is.connected(mongo) == TRUE) 
  {
    db_name = "Kaggle"
    data <- fread(data_path)
    df_list <- lapply(split(data, 1:nrow(data)), function(x) mongo.bson.from.JSON(toJSON(x)))
    dbname <- paste(db_name, collection_name, sep=".Bimbo_")
    mongo.insert.batch(mongo, dbname, df_list)
  }
}

insert_column = function(data, column_name)
{
  mongo = mongo.create(host = "localhost")
  if (mongo.is.connected(mongo) == TRUE) 
  {
    db_name = "Kaggle"
    df_list <- lapply(split(data, 1:nrow(data)), function(x) mongo.bson.from.JSON(toJSON(x)))
    dbname <- paste(db_name, column_name, sep=".Bimbo_")
    mongo.insert.batch(mongo, dbname, df_list)
  }
}

get_column = function(column_name)
{
  mongo = mongo.create(host = "localhost")
  if (mongo.is.connected(mongo) == TRUE) 
  {
    cursor <- mongo.find(mongo, column_name)
    res <- mongo.cursor.to.data.frame(cursor)
  }
  return(res)
}

insert_column(test_db[1:100000], "smnf")
dbname = "Kaggle.Bimbo_smnf"
data_test = get_column(dbname)
fwrite 
# # updata_column_by_id = 
# mongo.update(mongo,db_name)
# 
# mongo <- mongo.create()
# if (mongo.is.connected(mongo)) {
#   ns <- "test.people"
#   
#   buf <- mongo.bson.buffer.create()
#   mongo.bson.buffer.append(buf, "name", "Joe")
#   criteria <- mongo.bson.from.buffer(buf)
#   
#   buf <- mongo.bson.buffer.create()
#   mongo.bson.buffer.start.object(buf, "$inc")
#   mongo.bson.buffer.append(buf, "age", 1L)
#   mongo.bson.buffer.finish.object(buf)
#   objNew <- mongo.bson.from.buffer(buf)
#   
#   # increment the age field of the first record matching name "Joe"
#   mongo.update(mongo, ns, criteria, objNew)



# sql
#https://www.simple-talk.com/dotnet/software-tools/sql-and-r-/