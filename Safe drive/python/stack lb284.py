import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

# Regularized Greedy Forest
from rgf.sklearn import RGFClassifier     # https://github.com/fukatani/rgf_python

#####################
# just drop column and one hot encoding
####################
data_path = '/Users/xiaofeifei/I/Kaggle/Safe drive/'

train = pd.read_csv(data_path+'train_all.csv')
test = pd.read_csv(data_path+'test_all.csv')



train_stack = pd.read_csv(data_path + 'train_stack.csv')
test_stack = pd.read_csv(data_path + 'test_stack.csv')

### merge
train = train.merge(train_stack, on = 'id', how = 'left')
test = test.merge(test_stack, on = 'id', how = 'left')

# Preprocessing
id_test = test['id'].values
id_train = train['id'].values
target_train = train['target'].values

train = train.drop(['target','id'], axis = 1)
test = test.drop(['id'], axis = 1)


# col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
# train = train.drop(col_to_drop, axis=1)
# test = test.drop(col_to_drop, axis=1)


train = train.replace(-1, np.nan)
test = test.replace(-1, np.nan)


cat_features = [a for a in train.columns if a.endswith('cat')]

for column in cat_features:
	temp = pd.get_dummies(pd.Series(train[column]))
	train = pd.concat([train,temp],axis=1)
	train = train.drop([column],axis=1)

for column in cat_features:
	temp = pd.get_dummies(pd.Series(test[column]))
	test = pd.concat([test,temp],axis=1)
	test = test.drop([column],axis=1)


print(train.values.shape, test.values.shape)



class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
#                y_holdout = y[test_idx]

                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
#                cross_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
#                print("    cross_score: %.5f" % (cross_score.mean()))
                y_pred = clf.predict_proba(X_holdout)[:,1]

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        ##################################
        S_train = pd.DataFrame(S_train)
        n_splits = 5
        folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2016).split(S_train, y))

        SS_train = np.zeros((S_train.shape[0], 1))

        SS_test_i = np.zeros((S_test.shape[0], n_splits))

        for j, (train_idx, test_idx) in enumerate(folds):

            X_train = S_train.iloc[train_idx,]
            y_train = y[train_idx,]
            X_holdout = S_train.iloc[test_idx,]
            y_holdout = y[test_idx,]

            self.stacker.fit(X_train, y_train)

            y_pred = self.stacker.predict_proba(X_holdout)[:,1]
            SS_train[test_idx, 0] = y_pred


            SS_test_i[:, j] = self.stacker.predict_proba(S_test)[:,1]

        SS_test = S_test_i.mean(axis=1)
        ###################################
        return SS_train, SS_test



# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = 0.02
lgb_params['n_estimators'] = 650
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 10
lgb_params['colsample_bytree'] = 0.8
lgb_params['min_child_samples'] = 500
lgb_params['seed'] = 99
lgb_params['nthread'] = 6


lgb_params2 = {}
lgb_params2['n_estimators'] = 1090
lgb_params2['learning_rate'] = 0.02
lgb_params2['colsample_bytree'] = 0.3
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 16
lgb_params2['seed'] = 99
lgb_params2['nthread'] = 6

lgb_params3 = {}
lgb_params3['n_estimators'] = 1100
lgb_params3['max_depth'] = 4
lgb_params3['learning_rate'] = 0.02
lgb_params3['seed'] = 99
lgb_params3['nthread'] = 6


lgb_model = LGBMClassifier(**lgb_params)

lgb_model2 = LGBMClassifier(**lgb_params2)

lgb_model3 = LGBMClassifier(**lgb_params3)


log_model = LogisticRegression()



stack = Ensemble(n_splits=3,
        stacker = log_model,
        base_models = (lgb_model, lgb_model2, lgb_model3))

y_pred, y_test_pred = stack.fit_predict(train, target_train, test)


data_out = '/Users/xiaofeifei/I/Kaggle/Safe drive/stack/'

sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_test_pred
sub.to_csv(data_out+ 'stacked_1_test_s1.csv', index=False)

val = pd.DataFrame()
val['id'] = id_train
val['target'] = y_pred

val.to_csv(data_out+'stacked_1_train_s1.csv', float_format='%.6f', index=False)