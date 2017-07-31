import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import RobustScaler

path = "/Users/xiaofeifei/I/Kaggle/Benz/"
train = pd.read_csv(path+'train_start.csv')
# test = pd.read_csv(path+'test_start.csv')

y = train["y"]

train = train.drop(["y"], axis = 1)
a= RobustScaler()
train = a.fit_transform(train,y)
# # poly
kr = GridSearchCV(ElasticNet(alpha=0.001, l1_ratio=0.1, max_iter=5000), cv=5, n_jobs = 6,verbose=1,scoring='r2',
                  param_grid={"alpha": [0.01,0.1,1],
                              "l1_ratio": [0.3,0.7,0.9]})

kr.fit(train, y)
print kr.best_params_
print kr.best_score_
print kr.best_estimator_

# {'alpha': 0.01, 'l1_ratio': 0.9}
# 0.543871287359