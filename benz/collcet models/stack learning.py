import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline, _name_estimators
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor

path = "/Users/xiaofeifei/I/Kaggle/Benz/"

train = pd.read_csv(path+'train_final.csv')
test = pd.read_csv(path+'test_final.csv')

y_train = train['y'].values
y_mean = np.mean(y_train)
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
        super(LogExpPipeline, self).fit(X, np.log1p(y))

    def predict(self, X):
        return np.expm1(super(LogExpPipeline, self).predict(X))

#
# Model/pipeline with scaling,pca,svm
# knn
knn_pipe = LogExpPipeline(_name_estimators([RobustScaler(),
                                            KNeighborsRegressor(n_neighbors = 15, metric = 'cityblock')]))
#
svm_pipe = LogExpPipeline(_name_estimators([RobustScaler(),
                                            SVR(kernel='rbf', C=30, epsilon=0.05)]))

# results = cross_val_score(svm_pipe, train, y_train, cv=5, scoring='r2')
# print("SVM score: %.4f (%.4f)" % (results.mean(), results.std()))
# exit()

#
# Model/pipeline with scaling,pca,ElasticNet
#
en = ElasticNet(alpha=0.01, l1_ratio=0.9)

#
# XGBoost model
#
xgb_model = xgb.XGBRegressor(max_depth=4, learning_rate=0.0045, subsample=0.921,
                                     objective='reg:linear', n_estimators=1300, base_score=y_mean)


# results = cross_val_score(xgb_model, train, y_train, cv=5, scoring='r2')
# print("XGB score: %.4f (%.4f)" % (results.mean(), results.std()))


#
# Random Forest
#
rf_model = RandomForestRegressor(n_estimators=500, n_jobs=4, min_samples_split=10,
                                 min_samples_leaf=30, max_depth=3)

# results = cross_val_score(rf_model, train, y_train, cv=5, scoring='r2')
# print("RF score: %.4f (%.4f)" % (results.mean(), results.std()))

# ridge
Ridge = Ridge(alpha=37)

# lasso
lasso = LassoLarsCV(normalize=True)

#GBR
gbm = GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18,
min_samples_split=14, subsample=0.7)


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

                print ("Model %d fold %d score %f" % (i, j, r2_score(y_holdout, y_pred)))

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(axis=1)
            oof_score = r2_score(y, S_train[:, i])
            print 'Final Out-of-Fold Score %f'%oof_score

        # results = cross_val_score(self.stacker, S_train, y, cv=5, scoring='r2')
        # print("Stacker score: %.4f (%.4f)" % (results.mean(), results.std()))
        # exit()

        f_train = np.zeros((X.shape[0], 1))
        f_test = np.zeros((T.shape[0], 1))
        f_test_i = np.zeros((T.shape[0], self.n_splits))
        i = 0
        total_train = np.hstack((X, S_train))
        total_test = np.hstack((T, S_test))

        for j, (train_idx, test_idx) in enumerate(folds):
                X_train = total_train[train_idx]
                y_train = y[train_idx]
                X_holdout = total_train[test_idx]
                y_holdout = y[test_idx]

                self.stacker.fit(X_train, y_train)
                y_pred = self.stacker.predict(X_holdout)[:]

                print ("Model %d fold %d score %f" % (i, j, r2_score(y_holdout, y_pred)))
                f_train[test_idx, i] = y_pred
                f_test_i[:, j] = self.stacker.predict(total_test)[:]
        f_test[:, i] = f_test_i.mean(axis=1)
        oof_score = r2_score(y, f_train[:, i])
        print 'Final Out-of-Fold Score %f'%oof_score

        return f_test

#knn_pipe,svm_pipe, en,xgb_model, rf_model, Ridge, lasso, gbm
stack = Ensemble(n_splits=5,
                 #stacker=ElasticNetCV(l1_ratio=[x/10.0 for x in range(1,10)]),
                 stacker= xgb.XGBRegressor(max_depth=4, learning_rate=0.0045, subsample=0.93,
                                     objective='reg:linear', n_estimators=1300, base_score=y_mean),
                 base_models=())

# try lightgbm
# stack = Ensemble(n_splits=5,
#                  #stacker=ElasticNetCV(l1_ratio=[x/10.0 for x in range(1,10)]),
#                  stacker= LGBMRegressor(boosting_type='gbdt', max_depth=4, learning_rate=0.0045, n_estimators=1300,
#                        max_bin=25, subsample=0.995, silent=True, nthread=6),
#                  base_models=())
y_test = stack.fit_predict(train, y_train, test)

df_sub = pd.DataFrame({'ID': id_test, 'y': y_test.ravel()})

df_sub.to_csv('submission_new_feature.csv', index=False)

# Final Out-of-Fold Score 0.556746 R2 score on train data: 0.641067123739