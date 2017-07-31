import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
path = "/Users/xiaofeifei/I/Kaggle/Benz/"
train = pd.read_csv(path+'train_start.csv')
# test = pd.read_csv(path+'test_start.csv')

y = train["y"]

train = train.drop(["y"], axis = 1)

# # poly
# rf_model = RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25,
#                                  min_samples_leaf=25, max_depth=3)

kr = GridSearchCV(RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=10,
                                 min_samples_leaf=30, max_depth=3), cv=5, n_jobs = 6,verbose=1,scoring='r2',
                  param_grid={
                              })

kr.fit(train, y)
print kr.best_params_
print kr.best_score_
print kr.best_estimator_

# {'max_depth': 3, 'min_samples_leaf': 30,min_samples_split=10}
# 0.566926722357