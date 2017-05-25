from wmil import SGDWeights
import scipy
import numpy as np
import pandas as pd

from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.svm import NuSVR
from sklearn.metrics import r2_score
from sklearn.grid_search import GridSearchCV

from mcr import MIClusterRegress, Clusterer, RegressionModel

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



data = pd.read_csv("/Users/xiaofeifei/I/Kaggle/a/Modis.csv")

x = []
for i in range(1,10):
    a = data[data["ID"] == i]
    a.drop(["ID","label"], 1, inplace=True )

    x.append(a.values)



y = []
for i in range(1,10):
    a = data[data["ID"] == i]
    y.append(a['label'][i*100-1])

y = np.array(y)

if __name__ == '__main__':
    # exset = loadmat('/Users/xiaofeifei/I/Kaggle/a/thrombin.mat', struct_as_record=False)['thrombin']
    #
    # # Construct bags
    # all_labels = exset[:, 0]
    # X = exset[:, 1:-1]
    #
    # bags = []
    # values = []
    # for label in np.unique(all_labels.flat):
    #     indices = np.nonzero(all_labels == label)
    #     bags.append(X[indices])
    #     values.append(float(exset[indices, -1][0, 0]))
    # y = np.array(values)


    # # Fit bags, predict labels, and compute simple MSE
    mcr = MIClusterRegress(KMeansClusterer(n_clusters=3), regress)
    mcr.fit(x, y)
    y_hat = mcr.predict(x)
    print 'R^2: %f' % r2_score(y, y_hat)