from numpy import *
import pandas as pd
import time


train = pd.read_csv("/Users/xiaofeifei/I/Kaggle/a/X.csv")
y = pd.read_csv("/Users/xiaofeifei/I/Kaggle/a/ytr.csv")

# print train.head()
# calculate the sigmoid function
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

numSamples, numFeatures = shape(train)
weights = ones((numFeatures, 1))
output = sigmoid(train * weights)

# def trainLogRegres(train_x, train_y, opts):
#     """
#
#     :param train_x:
#     :param train_y:
#     :param opts:  {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'smoothStocGradDescent'}
#     :return:
#     """
#     # calculate training time
#     startTime = time.time()
#
#     numSamples, numFeatures = shape(train_x)
#     alpha = opts['alpha']; maxIter = opts['maxIter']
#     weights = ones((numFeatures, 1))
#
#     # optimize through gradient descent algorilthm
#     for k in range(maxIter):
#         if opts['optimizeType'] == 'gradDescent': # gradient descent algorilthm
#             output = sigmoid(train_x * weights)
#
#         else:
#             raise NameError('Not support optimize method type!')
#
#
#     print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
#     return weights
