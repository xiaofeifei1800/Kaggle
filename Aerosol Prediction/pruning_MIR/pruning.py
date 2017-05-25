#!/usr/bin/env python
"""
A simple example of using MIClusterRegress with clustering
and regression classes from scikits-learn
"""
import csv
import numpy as np
import pandas as pd
from pmir import PruningMIR
from sklearn.neural_network import MLPRegressor

def mse(observed, actual):
    return np.sqrt(np.average(np.square(observed - actual)))

# if __name__ == '__main__':
#     exset = loadmat('/Users/xiaofeifei/Downloads/thrombin.mat', struct_as_record=False)['thrombin']
#
#     # Construct bags
#     all_labels = exset[:, 0]
#     X = exset[:, 1:-1]
#     bags = []
#     values = []
#     for label in np.unique(all_labels.flat):
#         indices = np.nonzero(all_labels == label)
#         bags.append(X[indices])
#         values.append(float(exset[indices, -1][0, 0]))
#     y = np.array(values)
#     print type(y[1])

#
#
#     # Fit bags, predict labels, and compute simple MSE
#     regress = NuSVR(kernel='rbf', gamma=0.1, nu=0.2, C=1.0)
#     pmir = PruningMIR(regress)
#     pmir.fit(bags, y)
#     y_hat = pmir.predict(bags)
#     print 'R^2: %f' % r2_score(y, y_hat)
#
X_train = []
y_train = []
X_test = []

with open("/Users/xiaofeifei/I/Kaggle/a/X.csv", 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    spamreader.next()
    i = 1
    for row in spamreader:
        if i <=980*100:
            a = np.zeros(shape=(1,12))
            a[0] = row[0:12]
            X_train.append(a)
            # y_train.append(row[13])
        else:
            a = np.zeros(shape=(1,12))
            a[0] = row[0:12]
            X_test.append(a)
        i += 1

with open("/Users/xiaofeifei/I/Kaggle/a/ytr.csv", 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    i = 1
    spamreader.next()
    for row in spamreader:
        # for i in range(0,100,1):
        y_train.append(float(row[1]))

# reformat bags
bags = np.asarray(X_train)
new_bags = np.zeros(shape=(980,100,12))

for i in range(0,980,1):
    for j in range(0,100,1):
        new_bags[i][j] = bags[((100*i)+j)]


#reformat test
X_test = np.asarray(X_test)


# reformat y
y = np.array(y_train)


# # print type(y[1])
# # print bags.shape
# # print y.shape
# # print y_train[99:102]
regress = MLPRegressor(hidden_layer_sizes=(10,), activation='logistic', max_iter=2000)
# # regress = KernelRidge(alpha=1e-11, gamma=1e-6)
pmir = PruningMIR(regress, r=0.05)
pmir.fit(new_bags, y)

# y_hat = pmir.predict(X_test)


#
#
#
#
#     # df.to_csv("/Users/xiaofeifei/I/Kaggle/a/predictions_pruning_git.csv",index=False)
