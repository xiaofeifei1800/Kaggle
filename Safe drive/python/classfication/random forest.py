import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler


path = "/Users/xiaofeifei/I/Kaggle/Benz/"
train = pd.read_csv(path+'train_start.csv')
# test = pd.read_csv(path+'test_start.csv')
train.drop(["y"], axis=1, inplace=True)

y = train["class"]

train = train.drop(["class"], axis = 1)
a= RobustScaler()
train = a.fit_transform(train,y)
# # poly
kr = GridSearchCV(RandomForestClassifier(n_estimators=500, max_depth = 5, min_samples_split = 20, min_samples_leaf= 20),
                  cv=5, n_jobs = 6,verbose=1,
                  param_grid={"min_samples_split": [15,20,25,30],
                              "min_samples_leaf": [15,20,25,30]})

kr.fit(train, y)
print kr.best_params_
print kr.best_score_
print kr.best_estimator_

# {'min_samples_split': 20, 'min_samples_leaf': 25}
# 0.732478023283