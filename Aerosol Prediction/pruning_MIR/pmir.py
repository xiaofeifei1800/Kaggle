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

        if self.verbose: print 'Creating sampled dataset...'
        N, bags = _sampled_dataset(bags, self.sample_size)

        if self.verbose: print 'Fitting initial regression model...'
        self.f = _instance_mir(bags, y, self.regress)

        new_mse = mse(self.predict(bags), y)
        if self.verbose: print 'MSE: %f...\n' % new_mse

        if self.max_prune is None:
            iterator = count(1)
        else:
            iterator = range(1, self.max_prune + 1)

        iteration = 0
        for iteration in iterator:
            if self.verbose: print 'Iteration %d:' % iteration
            old_mse = new_mse
            old_f = self.f
            print "rate", self.r
            print "first N", N
            N = int((1-self.r)*N)
            print "delete N", N
            if N < 1: break

            if self.verbose: print 'Pruning bags...'
            new_bags = []
            for bag in bags:
                bp = self._predict_bag(bag)
                def score(instance):

                    return (self.f.predict(instance.reshape(1, -1)) - bp)**2
                scored_instances = sorted([i for i in bag], key=score)
                new_bags.append(np.vstack(scored_instances[:N]))
            bags = new_bags

            if self.verbose: print 'Fitting regression model...'
            self.f = _instance_mir(bags, y, self.regress)
            new_mse = mse(self.predict(bags), y)
            if self.verbose: print 'Old MSE: %f, New MSE: %f\n' % (old_mse, new_mse)
            if new_mse >= old_mse:
                self.f = old_f
                break

        self._N = max(1, N)
        self._iters = iteration
        if self.verbose: print 'Done.'

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