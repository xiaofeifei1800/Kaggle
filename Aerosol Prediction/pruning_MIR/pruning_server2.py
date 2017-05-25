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

"""BalancedPruning-MIR"""
import numpy as np
from random import randrange
from itertools import count

def mse(observed, actual):
    # new_actual =  a = np.zeros(shape=(980,1))
    # for i in range(0,980,1):
    #     new_actual = actual[100*i+1]
    return np.average(np.square(observed - actual))

class PruningMIR(object):
    """Implements the BalancedPruning-MIR algorithm described in:
    Zhuang Wang, Vladan Radosavljevic, Bo Han, and Zoran Obradovic.  Aerosol
    Optical Depth Prediction from Satellite Observations by Multiple Instance
    Regression. In Proceedings of the SIAM International Conference on Data
    Mining, 2008.
    """

    def __init__(self, regress, verbose=True,
                 md=lambda x: float(np.median(x)), r=0.05,
                 max_prune=None, sample_size=None):
        """
        BalancedPruning-MIR
        @param regress : returns an instance of a regression object
        @param verbose : print output during training
        @param md : median or mean used to select bag label
        @param r : fraction to prune at each iteration
        @param max_prune : maximum pruning iterations
        @param sample_size : sample size used to make consistent bags
        """
        self.regress = regress
        self.verbose = verbose
        self.md = md
        self.r = r
        self.max_prune = max_prune
        self.sample_size = sample_size

        self.f = None
        self._iters = None
        self._N = None

    def fit(self, bags, y):
        """
        Fit the model according to the given training data
        @param bags : sequence of array-like bags, each with
                      shape [n_instances, n_features]
        @param y : bag labels, array-like, shape [n_bags]
        @return : self
        """
        bags = map(np.asarray, bags)
        y = np.asarray(y)

        if self.verbose: print ('Creating sampled dataset...')
        N, bags = _sampled_dataset(bags, self.sample_size)

        if self.verbose: print ('Fitting initial regression model...')
        self.f = _instance_mir(bags, y, self.regress)

        new_mse = mse(self.predict(bags), y)
        if self.verbose: print ('MSE: %f...\n' % new_mse)

        if self.max_prune is None:
            iterator = count(1)
        else:
            iterator = range(1, self.max_prune + 1)

        iteration = 0
        for iteration in iterator:
            if self.verbose: print ('Iteration %d:' % iteration)
            old_mse = new_mse
            old_f = self.f

            print ("first N", N)
            N = int((1-self.r)*N)
            print ("delete N", N)
            if N < 1: break

            if self.verbose: print ('Pruning bags...')
            new_bags = []
            for bag in bags:
                bp = self._predict_bag(bag)
                def score(instance):

                    return (self.f.predict(instance.reshape(1, -1)) - bp)**2
                scored_instances = sorted([i for i in bag], key=score)
                new_bags.append(np.vstack(scored_instances[:N]))
            bags = new_bags

            if self.verbose: print ('Fitting regression model...')
            self.f = _instance_mir(bags, y, self.regress)
            new_mse = mse(self.predict(bags), y)
            if self.verbose: print ('Old MSE: %f, New MSE: %f\n' % (old_mse, new_mse))
            if new_mse >= old_mse:
                self.f = old_f
                break

        self._N = max(1, N)
        self._iters = iteration
        if self.verbose: print ('Done.')

    def _predict_bag(self, bag):
        return self.md(self.f.predict(bag))

    def predict(self, bags):
        """
        Apply fit regression function to each bag
        @param bags : sequence of array-like bags, each with
                      shape [n_instances, n_features]
        @return y : array, shape [n_bags]
        """
        return np.array([self._predict_bag(bag) for bag in bags])

def _sampled_dataset(bags, N=None):
    if N is None:
        # Use the largest bag size as N
        N = max(bag.shape[0] for bag in bags)
    return N, [_sample_replace(bag, N) for bag in bags]

def _sample_replace(bag, N):
    return np.vstack([bag[randrange(bag.shape[0])]
                      for _ in range(N)])

def _instance_mir(bags, y, regress):
    """
    Returns a Predictor trained on the given bags
    and class labels by applying a bag label to
    each instance in the bag.
    Arguments:
    @param bags : a sequence of NumPy arrays, with
    an instance in each row of each array
    @param y : a NumPy array holding bag label values
    @param regress : a function that takes an array of
    data and corresponding labels, and returns
    a RegressionModel object.
    """
    N = bags[0].shape[0]

    X = np.vstack(bags)

    de = X.shape[0]/980

    Y = np.hstack(y_i*np.ones(de) for y_i in y.flat)
    # print Y[98:105]
    return regress.fit(X, Y)

X_train = []
y_train = []
X_test = []

with open("/home/slfan/GuoxinLi/a/X.csv", 'rb') as csvfile:
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

with open("/home/slfan/GuoxinLi/a/ytr.csv", 'rb') as csvfile:
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

new_test = np.zeros(shape=(980,100,12))

for i in range(0,384,1):
    for j in range(0,100,1):
        new_test[i][j] = X_test[((100*i)+j)]

# reformat y
y = np.array(y_train)

results = pd.DataFrame(np.zeros(38400))

for k in range(0,50,1):
    bootstrap = np.random.choice(980, 980, replace=True)
    y_bot = np.zeros(shape=(980,1))
    bot_bags = np.zeros(shape=(980,100,12))
    j = 0
    for i in bootstrap:
        bot_bags[j] = new_bags[i]
        y_bot[j] = y[i]
        j += 1

    # # print type(y[1])
    # # print bags.shape
    # # print y.shape
    # # print y_train[99:102]
    regress = MLPRegressor(hidden_layer_sizes=(10, ), activation = 'logistic', max_iter=2000)
    # # regress = KernelRidge(alpha=1e-11, gamma=1e-6)
    pmir = PruningMIR(regress, r=0.01)
    pmir.fit(bot_bags, y_bot)

    y_hat = pmir.predict(X_test)
    results[k]= y_hat


    results.to_csv("/home/slfan/GuoxinLi/a/predictions_pruning_git2.csv",index=False)
