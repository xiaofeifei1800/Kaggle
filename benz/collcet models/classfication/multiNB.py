import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import RobustScaler, normalize


path = "/Users/xiaofeifei/I/Kaggle/Benz/"
train = pd.read_csv(path+'train_start.csv')
# test = pd.read_csv(path+'test_start.csv')
train.drop(["y"], axis=1, inplace=True)

y = train["class"]

train = train.drop(["class"], axis = 1)
train = train.ix[:,9:377]
print train
# train = normalize(train)
# print train
# # poly
kr = GridSearchCV(MultinomialNB(alpha=1.0, fit_prior=True), cv=5, n_jobs = 6,verbose=1,scoring='r2',
                  param_grid={"alpha": [0.1,1,5,10,15]})

kr.fit(train, y)
print kr.best_params_
print kr.best_score_
print kr.best_estimator_

# {'alpha': 10}
# 0.379014466892