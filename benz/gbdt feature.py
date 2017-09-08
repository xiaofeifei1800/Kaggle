import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import xgboost as xgb
import datetime
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# overfit!!!!!!!!!!!!!!!!!!!
path = "/Users/xiaofeifei/I/Kaggle/Benz/"
train = pd.read_csv(path+'train_start.csv')
test = pd.read_csv(path+'test_start.csv')

y = train["y"]
train = train.drop('y', axis=1)

y_mean = np.mean(y)

TREES = 50
NODES = 7
clr = GradientBoostingRegressor(n_estimators = TREES, max_depth = NODES, verbose = 1)
start_time = datetime.datetime.now()

clr.fit(train, y)
print(datetime.datetime.now()-start_time)
# y_pred = clr.predict(test)
train = train.values
train = train.astype(np.float32)
trees = clr.estimators_
n_trees = trees.shape[0]
indices = []

for i in range(n_trees):
    tree = trees[i][0].tree_
    indices.append(tree.apply(train))

indices = np.column_stack(indices)
indices = pd.DataFrame(indices)

train = pd.read_csv(path+'train_start.csv')
train = train.drop('y', axis=1)
train = pd.concat([train, indices], axis=1)


# dtrain = xgb.DMatrix(train, y_train)
#
#
params = {
    'n_trees': 1000,
    'eta': 0.005,
    'max_depth': 4,
    'subsample': 0.6,
    'objective': 'reg:linear',
    # 'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1,
    "n_jobs":6
}
# # xgboost, cross-validation
def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'R^2', (r2_score(labels, preds))

# cv_result = xgb.cv(xgb_params,
#                    dtrain,
#                    num_boost_round=1000, # increase to have better results (~700)
#                    early_stopping_rounds=50,
#                    maximize=True,
#                    verbose_eval=50,
#                    show_stdv=False,
#                    feval=evalerror
#                   )
#
# num_boost_rounds = len(cv_result)
#
# model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
#
# # now fixed, correct calculation
# print(r2_score(dtrain.get_label(), model.predict(dtrain)))


test = test.values
test = test.astype(np.float32)
trees = clr.estimators_
n_trees = trees.shape[0]
indices = []

for i in range(n_trees):
    tree = trees[i][0].tree_
    indices.append(tree.apply(test))

indices = np.column_stack(indices)
indices = pd.DataFrame(indices)
test = pd.read_csv(path+'test_start.csv')

test = pd.concat([test, indices], axis=1)
dtest = xgb.DMatrix(test)

# y_pred = model.predict(dtest)
#
# test = pd.read_csv(path+'test_start.csv')
# output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
# output.to_csv(path+'xgboost-depth{}-pca-ica.csv'.format(xgb_params['max_depth']), index=False)

n_splits = 5
kf = KFold(n_splits=n_splits)
kf.get_n_splits(train)
# dtest = xgb.DMatrix(test)
predictions = np.zeros((test.shape[0], n_splits))
score = 0

oof_predictions = np.zeros(train.shape[0])
for fold, (train_index, test_index) in enumerate(kf.split(train)):
    X_train, X_valid = train.iloc[train_index, :], train.iloc[test_index, :]
    y_train, y_valid = y[train_index], y[test_index]

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_valid, label=y_valid)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    model = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=50, feval=xgb_r2_score, maximize=True, verbose_eval=False)
    pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    oof_predictions[test_index] = model.predict(d_valid, ntree_limit=model.best_ntree_limit)
    predictions[:, fold] = pred
    score += model.best_score
    print 'Fold %d: Score %f'%(fold, model.best_score)

prediction = predictions.mean(axis=1)
score /= n_splits
oof_score = r2_score(y, oof_predictions)
oof_predictions_cate = oof_predictions
print '====================='
print 'Final Score %f'%score
print 'Final Out-of-Fold Score %f'%oof_score
print '====================='

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': prediction})
output.to_csv(path+'xgboost-gbdt-pca-ica.csv', index=False)