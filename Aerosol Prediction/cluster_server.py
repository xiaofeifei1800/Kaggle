from wmil import SGDWeights
import scipy
import numpy as np
import pandas as pd
import datetime
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.svm import NuSVR
from sklearn.metrics import r2_score
from sklearn.grid_search import GridSearchCV
from math import log10

class MIClusterRegress(object):
    """Implements the MI-ClusterRegress algorithm described in:
    Kiri L. Wagstaff, Terran Lane, and Alex Roper. Multiple-Instance Regression
    with Structured Data. Proceedings of the 4th International Workshop on
    Mining Complex Data, December 2008.
    """

    def __init__(self, cluster, regress, verbose=True):
        """
        MI-ClusterRegress
        @param cluster : an implemented Clusterer object
        @param regress : returns an instance of a RegressionModel object
        @param verbose : print output during training
        """
        self.cluster = cluster
        self.regress = regress
        self.regression_models = None
        self.selected_index = None
        self.verbose = verbose

    def _status_msg(self, message):
        """
        Print a status message if verbose output is enabled.
        @param message : message to print
        """
        if self.verbose: print (message)

    def _exemplars(self, bags):
        """
        Compute an array of bag exemplars w.r.t. each cluster.
        @param bags : sequence of array-like bags, each with
                      shape [n_instances, n_features]
        @return : sequence of array exemplar sets, one for each
                  cluster with shape [n_bags, n_features]
        """

        # Sequence of matrices holding exemplars for each cluster
        cluster_examplars = [np.dot(self.cluster.relevance(bag), bag)
                             for bag in bags]

        # Convert to sequence of matrices holding bag exemplars per cluster
        bag_exemplars = map(np.vstack, zip(*cluster_examplars))
        k = len(list(map(np.vstack, zip(*cluster_examplars))))
        return bag_exemplars,k

    def _select(self, bags, y):
        """
        Select the cluster/model index to use for predicting new bags that
        minimizes training error.
        (Override this method to implement different selection criteria.)
        @param bags : sequence of array-like bags, each with
                      shape [n_instances, n_features]
        @param y : bag labels, array-like, shape [n_bags]
        @return : index of cluster/model to use for prediction
        """
        exemplars_models = zip(self._exemplars(bags), self.regression_models)
        predictions = [model.predict(ex_set)
                       for ex_set, model in exemplars_models]
        mses = [np.average(np.square(p - y)) for p in predictions]

        return mses.index(min(mses))

    def fit(self, bags, y):
        """
        Fit the model according to the given training data
        @param bags : sequence of array-like bags, each with
                      shape [n_instances, n_features]
        @param y : bag labels, array-like, shape [n_bags]
        @return : self
        """
        bags = map(np.asarray, bags)
        X = np.vstack(bags)

        self._status_msg('Clustering instances...')
        self.cluster.fit(X)

        self._status_msg('Computing exemplars...')
        exemplar_sets,k = self._exemplars(bags)

        self._status_msg('Computing regression models...')


        fstr = '    %%0%dd of %d...' % (int(log10(k)) + 1, k)
        self.regression_models = list()
        for i, ex_set in enumerate(exemplar_sets, 1):
            self._status_msg(fstr % i)
            model = self.regress()
            model.fit(ex_set, y)
            self.regression_models.append(model)

        self._status_msg('Selecting predictor...')
        self.selected_index = self._select(bags, y)

        return self

    def predict(self, bags):
        """
        Apply fit regression function to each bag
        @param bags : sequence of array-like bags, each with
                      shape [n_instances, n_features]
        @return y : array, shape [n_bags]
        """
        bags = map(np.asarray, bags)
        exemplars, model = zip(
            self._exemplars(bags), self.regression_models)[self.selected_index]
        return model.predict(exemplars)

class Clusterer(object):
    """Interface for an object that can cluster data"""

    def fit(self, X):
        """
        Computes clusters of data X.
        @param X : array-like, shape [n_instances, n_features]
        """
        pass

    def relevance(self, bag):
        """
        Returns the relevance matrix, i.e. the relevance of the bag instances to
        each cluster. Relevance should be normalized so that the relevances of
        instances within each cluster (each row) sum to one.
        @param bag : array-like, shape [n_instances, n_features]
                     bag for which relevance should be computed
        @return : array, shape [n_clusters, n_instances]
        """
        pass

    @staticmethod
    def normalize(relevance):
        """
        A convenience method that ensures the relevance matrix is properly
        normalized.
        @param relevance : array, shape [n_clusters, n_instances]
        @return : normalized copy of relevance matrix
        """
        relevance = np.array(relevance)
        sums = np.sum(relevance, axis=1)
        for row, s in enumerate(sums):
            if s == 1:
                continue
            elif s == 0:
                n = relevance.shape[1]
                relevance[row, :] = 1.0 / n
            else:
                relevance[row, :] /= s

        return relevance

class RegressionModel(object):
    """Interface for an instance-based regression technique"""

    def fit(self, X, y):
        """
        Fit the model according to the given training data
        @param X : array-like, shape [n_instances, n_features]
        @param y : array-like, shape [n_instances]
        """
        pass

    def predict(self, X):
        """
        Apply fit regression function to each instance in X
        @param X : array-like, shape [n_instances, n_features]
        @return y : array, shape [n_instances]
        """
        pass

class KMeansClusterer(KMeans, Clusterer):
    """
    Implements a k-means clusterer for MIClusterRegress
    """
    def relevance(self, bag):
        """
        Relevance equal to one for closest bag and zero for others
        """
        dist = self.transform(bag).T
        closest = np.argmin(dist, axis=1)
        indicators = np.zeros(dist.shape)
        for row, c in enumerate(closest):
            indicators[row, c] = 1.0
        return Clusterer.normalize(indicators)

def regress():
    """
    Implements the function to return a new RegressionModel with
    appropriate parameters. Note that the regression classes in
    scikits-learn already conform to the approprate interface.
    """
    param_values = {
        'C': [10.0**i for i in (1, 2, 3, 4)],
        'nu': [0.2, 0.4, 0.4, 0.6],
        'kernel': ['linear'],
    }
    nu_svr = NuSVR()
    grid_nu_svr = GridSearchCV(nu_svr, param_values, cv=5)
    return grid_nu_svr


start_time = datetime.datetime.now()
data = pd.read_csv("/home/slfan/GuoxinLi/a/Modis.csv")

x = []
for i in range(1,1365):
    a = data[data["ID"] == i]
    a.drop(["ID","label"], 1, inplace=True )

    x.append(a.values)



y = []
for i in range(1,1365):
    a = data[data["ID"] == i]
    y.append(a['label'][i*100-1])

y = np.array(y)


mcr = MIClusterRegress(KMeansClusterer(n_clusters=3), regress)
mcr.fit(x, y)
y_hat = mcr.predict(x)
print ('R^2: %f' % r2_score(y, y_hat))
print (datetime.datetime.now()-start_time)