import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

path = "/Users/xiaofeifei/I/Kaggle/Benz/"
train = pd.read_csv(path+'train_test.csv')
test = pd.read_csv(path+'test_start.csv')

train_cate = train.iloc[:,2:10]
train_numeric = train.iloc[:,10:378]
train_project = train.iloc[:,378:]

test_cate = test.iloc[:,1:9]
test_numeric = test.iloc[:,9:377]
test_project = test.iloc[:,377:]

y = train["y"]
y_mean = np.mean(y)

train = train.drop('y', axis=1)

def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    r_squared = r2_score(labels, preds)

    return 'R^2', r_squared

params = {
    'n_trees': 500,
    'eta': 0.005,
    'max_depth': 4,
    'subsample': 0.921,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}

n_splits = 5
kf = KFold(n_splits=n_splits)
kf.get_n_splits(train)
dtest = xgb.DMatrix(test)
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
output.to_csv(path+'xgboost-test-pca-ica.csv', index=False)
# #####
# n_splits = 5
# kf = KFold(n_splits=n_splits)
# kf.get_n_splits(train_cate)
# dtest = xgb.DMatrix(test_cate)
# predictions = np.zeros((test.shape[0], n_splits))
# score = 0
#
# oof_predictions = np.zeros(train_cate.shape[0])
# for fold, (train_index, test_index) in enumerate(kf.split(train_cate)):
#     X_train, X_valid = train_cate.iloc[train_index, :], train_cate.iloc[test_index, :]
#     y_train, y_valid = y[train_index], y[test_index]
#
#     d_train = xgb.DMatrix(X_train, label=y_train)
#     d_valid = xgb.DMatrix(X_valid, label=y_valid)
#
#     watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#
#     model = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=50, feval=xgb_r2_score, maximize=True, verbose_eval=False)
#     pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
#     oof_predictions[test_index] = model.predict(d_valid, ntree_limit=model.best_ntree_limit)
#     predictions[:, fold] = pred
#     score += model.best_score
#     print 'Fold %d: Score %f'%(fold, model.best_score)
#
# prediction = predictions.mean(axis=1)
# test["cate"] = prediction
# score /= n_splits
# oof_score = r2_score(y, oof_predictions)
# oof_predictions_cate = oof_predictions
# train["cate"] = oof_predictions_cate
# print '====================='
# print 'Final Score %f'%score
# print 'Final Out-of-Fold Score %f'%oof_score
# print '====================='
#
#
# ###################################################################
# # numeric
# kf.get_n_splits(train_numeric)
# dtest = xgb.DMatrix(test_numeric)
# predictions = np.zeros((test.shape[0], n_splits))
# score = 0
#
# oof_predictions = np.zeros(train_numeric.shape[0])
# for fold, (train_index, test_index) in enumerate(kf.split(train_numeric)):
#     X_train, X_valid = train_numeric.iloc[train_index, :], train_numeric.iloc[test_index, :]
#     y_train, y_valid = y[train_index], y[test_index]
#
#     d_train = xgb.DMatrix(X_train, label=y_train)
#     d_valid = xgb.DMatrix(X_valid, label=y_valid)
#
#     watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#
#     model = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=50, feval=xgb_r2_score, maximize=True, verbose_eval=False)
#     pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
#     oof_predictions[test_index] = model.predict(d_valid, ntree_limit=model.best_ntree_limit)
#     predictions[:, fold] = pred
#     score += model.best_score
#     print 'Fold %d: Score %f'%(fold, model.best_score)
#
# prediction = predictions.mean(axis=1)
# test["num"] = prediction
# score /= n_splits
# oof_score = r2_score(y, oof_predictions)
# oof_predictions_num = oof_predictions
# train["num"] = oof_predictions_num
# print '====================='
# print 'Final Score %f'%score
# print 'Final Out-of-Fold Score %f'%oof_score
# print '====================='
# # output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': prediction})
# # output.to_csv(path+'xgboost-num-pca-ica.csv', index=False)
# ###################################################################
# # projection
# kf.get_n_splits(train_project)
# dtest = xgb.DMatrix(test_project)
# predictions = np.zeros((test.shape[0], n_splits))
# score = 0
#
# oof_predictions = np.zeros(train_project.shape[0])
# for fold, (train_index, test_index) in enumerate(kf.split(train_project)):
#     X_train, X_valid = train_project.iloc[train_index, :], train_project.iloc[test_index, :]
#     y_train, y_valid = y[train_index], y[test_index]
#
#     d_train = xgb.DMatrix(X_train, label=y_train)
#     d_valid = xgb.DMatrix(X_valid, label=y_valid)
#
#     watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#
#     model = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=50, feval=xgb_r2_score, maximize=True, verbose_eval=False)
#     pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
#     oof_predictions[test_index] = model.predict(d_valid, ntree_limit=model.best_ntree_limit)
#     predictions[:, fold] = pred
#     score += model.best_score
#     print 'Fold %d: Score %f'%(fold, model.best_score)
#
# prediction = predictions.mean(axis=1)
# test["proj"] = prediction
# score /= n_splits
# oof_score = r2_score(y, oof_predictions)
# oof_predictions_proj = oof_predictions
# train["proj"] = oof_predictions_proj
# print '====================='
# print 'Final Score %f'%score
# print 'Final Out-of-Fold Score %f'%oof_score
# print '====================='
#
# ###################################################################
# # stacking
#
# kf.get_n_splits(train)
# dtest = xgb.DMatrix(test)
# predictions = np.zeros((test.shape[0], n_splits))
# score = 0
#
# oof_predictions = np.zeros(train.shape[0])
# for fold, (train_index, test_index) in enumerate(kf.split(train)):
#     X_train, X_valid = train.iloc[train_index, :], train.iloc[test_index, :]
#     y_train, y_valid = y[train_index], y[test_index]
#
#     d_train = xgb.DMatrix(X_train, label=y_train)
#     d_valid = xgb.DMatrix(X_valid, label=y_valid)
#
#     watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#
#     model = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=50, feval=xgb_r2_score, maximize=True, verbose_eval=False)
#     pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
#     oof_predictions[test_index] = model.predict(d_valid, ntree_limit=model.best_ntree_limit)
#     predictions[:, fold] = pred
#     score += model.best_score
#     print 'Fold %d: Score %f'%(fold, model.best_score)
#
# prediction = predictions.mean(axis=1)
# score /= n_splits
# oof_score = r2_score(y, oof_predictions)
# oof_predictions_proj = oof_predictions
# print '====================='
# print 'Final Score %f'%score
# print 'Final Out-of-Fold Score %f'%oof_score
# print '====================='

# output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': prediction})
# output.to_csv(path+'xgboost-stacking-pca-ica.csv', index=False)

