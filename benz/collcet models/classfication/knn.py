import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
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
# kr = GridSearchCV(KNeighborsClassifier(n_neighbors=5, metric='minkowski'), cv=5, n_jobs = 6,verbose=1,scoring='r2',
#                   param_grid={"n_neighbors": [5,15,20,25,30],
#                               "metric": ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan']})
#
# kr.fit(train, y)
# print kr.best_params_
# print kr.best_score_
# print kr.best_estimator_

# {'n_neighbors': 15, 'metric': 'cityblock'}
# 0.449594611604


clf = KNeighborsClassifier(n_neighbors = 15, metric = 'cityblock')
clf.fit(train, y)
print clf.predict(train)