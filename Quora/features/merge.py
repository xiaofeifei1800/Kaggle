import pandas as pd
input_folder = '/Users/xiaofeifei/I/Kaggle/Quora/'

data = pd.read_csv(input_folder+'base_feature.csv', index=False)
temp = pd.read_csv(input_folder+'base_feature1.csv', index=False)
data = data.apped(temp)
del temp

temp = pd.read_csv(input_folder+'count_feature.csv', index=False)
data = data.apped(temp)
del temp

temp = pd.read_csv(input_folder+'distance_feature.csv', index=False)
data = data.apped(temp)
del temp

temp = pd.read_csv(input_folder+'magic_feature.csv', index=False)
data = data.apped(temp)
del temp

temp = pd.read_csv(input_folder+'russian feature.csv', index=False)
data = data.apped(temp)
del temp

train = data.loc[data["index"]=="train"]
test = data.loc[data["index"]=="test"]
del data

train.to_csv(input_folder+"all_train.csv")
test.to_csv(input_folder+"all_test.csv")