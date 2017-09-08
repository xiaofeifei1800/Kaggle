from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import log_loss

path = "/Users/xiaofeifei/I/Kaggle/Benz/"
train = pd.read_csv(path+'train_start.csv')
# test = pd.read_csv(path+'test_start.csv')
train.drop(["y"], axis=1, inplace=True)

y = train["class"]

train = train.drop(["class"], axis = 1)

# # poly

a= RobustScaler()
train = a.fit_transform(train,y)
kr = GridSearchCV(SVC(kernel='rbf', C=1.0), cv=5, n_jobs = 6,verbose=1,
                  param_grid={"C": [14,15,16,17]})

kr.fit(train, y)
print kr.best_params_
print kr.best_score_
print kr.best_estimator_

# {'C': 14}
# 0.740555951532