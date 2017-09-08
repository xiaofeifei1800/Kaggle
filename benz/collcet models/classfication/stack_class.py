import pandas as pd
import numpy as np
from sklearn.svm import SVC

from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline, _name_estimators
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

path = "/Users/xiaofeifei/I/Kaggle/Benz/"

train = pd.read_csv(path+'train_start.csv')
test = pd.read_csv(path+'test_start.csv')

y_train = train['class']
# y_train = y_train-1
id_test = test['ID']

num_train = len(train)
df_all = pd.concat([train, test])
df_all.drop(['ID', 'y',"class"], axis=1, inplace=True)

# One-hot encoding of categorical/strings
df_all = pd.get_dummies(df_all, drop_first=True)

train = df_all[:num_train]
test = df_all[num_train:]


class AddColumns(BaseEstimator, TransformerMixin):
    def __init__(self, transform_=None):
        self.transform_ = transform_

    def fit(self, X, y=None):
        self.transform_.fit(X, y)
        return self

    def transform(self, X, y=None):
        xform_data = self.transform_.transform(X, y)
        return np.append(X, xform_data, axis=1)


class LogExpPipeline(Pipeline):
    def fit(self, X, y):
        super(LogExpPipeline, self).fit(X, y)

    def predict(self, X):
        return super(LogExpPipeline, self).predict(X)

#
# Model/pipeline with scaling,pca,svm
# knn
knn_pipe = LogExpPipeline(_name_estimators([RobustScaler(),
                                            KNeighborsClassifier(n_neighbors = 15, metric = 'cityblock')]))
#
svm_pipe = LogExpPipeline(_name_estimators([RobustScaler(),
                                            SVC(kernel='rbf', C=14)]))

# results = cross_val_score(svm_pipe, train, y_train, cv=5, scoring='r2')
# print("SVM score: %.4f (%.4f)" % (results.mean(), results.std()))
# exit()


#
# XGBoost model
#
xgb_model = xgb.XGBClassifier(max_depth=4, learning_rate=0.0045, subsample=0.921,nthread=6,
                                     objective='multi:softmax', n_estimators=500)


# results = cross_val_score(xgb_model, train, y_train, cv=5, scoring='r2')
# print("XGB score: %.4f (%.4f)" % (results.mean(), results.std()))


#
# Random Forest
#
rf_model = RandomForestClassifier(n_estimators=500, n_jobs=6, min_samples_split=20,
                                 min_samples_leaf=25, max_depth=5)

# results = cross_val_score(rf_model, train, y_train, cv=5, scoring='r2')
# print("RF score: %.4f (%.4f)" % (results.mean(), results.std()))

# NN
NN_pipe = LogExpPipeline(_name_estimators([RobustScaler(),
                                           MLPClassifier(hidden_layer_sizes=(25, 50), activation='relu',
                                                         alpha=0.0001,early_stopping=True, max_iter=1000)]))

# lightgbm
lightgbm = LGBMClassifier(boosting_type='gbdt', num_leaves=10, max_depth=4, learning_rate=0.005, n_estimators=500,
                       objective="multiclass",max_bin=25, subsample=0.995, silent=True, nthread=6)


#
# Now the training and stacking part.  In previous version i just tried to train each model and
# find the best combination, that lead to a horrible score (Overfit?).  Code below does out-of-fold
# training/predictions and then we combine the final results.
#
# Read here for more explanation (This code was borrowed/adapted) :
#

class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]

                print ("Model %d fold %d score %f" % (i, j, accuracy_score(y_holdout, y_pred)))

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(axis=1)
            oof_score = accuracy_score(y, S_train[:, i])
            print 'Final Out-of-Fold Score %f'%oof_score
        return S_train, S_test


# knn_pipe,svm_pipe,NN_pipe,xgb_model,rf_model,lightgbm
stack = Ensemble(n_splits=5,
                 #stacker=ElasticNetCV(l1_ratio=[x/10.0 for x in range(1,10)]),
                 stacker= xgb.XGBRegressor(max_depth=4, learning_rate=0.0045, subsample=0.93,
                                     objective='reg:linear', n_estimators=1300),
                 base_models=(knn_pipe,svm_pipe,NN_pipe,xgb_model,rf_model,lightgbm))

S_train, S_test = stack.fit_predict(train, y_train, test)

S_train = pd.DataFrame(S_train)
S_test = pd.DataFrame(S_test)
S_train.columns = ["knn_pipe","svm_pipe","NN_pipe","xgb_model","rf_model","lightgbm"] #"knn_pipe","svm_pipe","NN_pipe","xgb_model","rf_model","lightgbm"
S_test.columns = ["knn_pipe","svm_pipe","NN_pipe","xgb_model","rf_model","lightgbm"]

S_train.to_csv('stacking_cla_train.csv', index=False)
S_test.to_csv('stacking_cla_test.csv', index=False)
