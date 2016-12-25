library(data.table)

events <- fread("I:/Kaggle/Data/Outbrain/events.csv")
events$day = round(events$timestamp/(3600 * 24 * 1000),1)
