from sklearn.svm import SVR
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler


path = "/Users/xiaofeifei/I/Kaggle/Benz/"
train = pd.read_csv(path+'train_start.csv')
# test = pd.read_csv(path+'test_start.csv')

y = train["y"]

train = train.drop(["y"], axis = 1)

# # poly
svm = SVR(kernel='rbf', C=1.0, epsilon=0.05)

a= RobustScaler()
train = a.fit_transform(train,y)
kr = GridSearchCV(SVR(kernel='rbf', C=1.0, epsilon=0.05), cv=5, n_jobs = 6,verbose=1,scoring='r2',
                  param_grid={"C": [20,30],
                              "epsilon": [0.02,0.03,0.05,0.07]})

kr.fit(train, y)
print kr.best_params_
print kr.best_score_
print kr.best_estimator_

# {'epsilon': 0.01, 'C': 30}
# 0.536811148843