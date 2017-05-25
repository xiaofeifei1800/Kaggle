from wmil import SGDWeights
import scipy
import numpy as np
import pandas as pd
data = pd.read_csv("/Users/xiaofeifei/I/Kaggle/a/Modis.csv")

x = []
for i in range(1,1365):
    a = data[data["ID"] == i]
    a.drop(["ID","label"], 1, inplace=True )
    sparse_matrix = scipy.sparse.csr_matrix(a.values)
    x.append(sparse_matrix)

print x[1]
#
# y = []
# for i in range(1,1365):
#     a = data[data["ID"] == i]
#     y.append(a['label'][i*100-1])
#
model = SGDWeights(alpha=0.4, momentum=0.0, minib=50)
# model.fit(x, y)