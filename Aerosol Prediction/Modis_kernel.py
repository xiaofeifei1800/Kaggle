from __future__ import division
import time

import csv
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
import pandas as pd

X_train = []
y_train = []
X_test = []

with open("/Users/xiaofeifei/I/Kaggle/a/train_new_data.csv", 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    spamreader.next()
    for row in spamreader:
        X_train.append(row[0:12])
        y_train.append(row[13])

with open("/Users/xiaofeifei/I/Kaggle/a/test_new_data.csv", 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    spamreader.next()
    for row in spamreader:
        X_test.append(row[0:12])



X_train = np.asarray(X_train)[1:600]
print X_train.shape
y_train = np.asarray(y_train)[1:600]
print len(y_train)
# y_train=LabelEncoder().fit_transform(y_train)
X_test = np.asarray(X_test)
print X_test.shape

clf = KernelRidge(alpha=1e-11, gamma=1e-6)
scores = cross_val_score(clf, X_train, y_train, cv=5)
print scores
# clf.fit(X_train, y_train)
# pred = clf.predict(X_test)
# df = pd.DataFrame(pred)
#
#
# df.to_csv("/Users/xiaofeifei/I/Kaggle/a/predictions_test.csv",index=False)
