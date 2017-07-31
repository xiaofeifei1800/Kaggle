import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

path = "/Users/xiaofeifei/I/Kaggle/Benz/"
train = pd.read_csv(path+'train_start.csv')
# test = pd.read_csv(path+'test_start.csv')

y = train["y"]

train = train.drop(["y"], axis = 1)

# # poly
kr = GridSearchCV(KernelRidge(kernel='poly'), cv=5, n_jobs = 6,verbose=1,scoring='r2',
                  param_grid={"alpha": [9000,10000,20000,30000,40000,50000,80000],
                              "degree": [2]})

kr.fit(train, y)
print kr.best_params_
print kr.best_score_
print kr.best_estimator_
# {'alpha': 20000, 'degree': 2}
# 0.511117661279
#
# clf = Ridge()
# ridge_params = {'alpha': [33,34,35,36,37,38,39,40]}
# ridge_grid = GridSearchCV(clf,ridge_params,cv=5,verbose=10,scoring='r2',n_jobs = 6,)
# model = ridge_grid.fit(train, y)
#
# print ridge_grid.best_params_
# print ridge_grid.best_score_
# print ridge_grid.best_estimator_
# {'alpha': 37}
# 0.540483130061