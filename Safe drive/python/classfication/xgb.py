from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

path = "/Users/xiaofeifei/I/Kaggle/Benz/"
train = pd.read_csv(path+'train_start.csv')
# test = pd.read_csv(path+'test_start.csv')
train.drop(["y"], axis=1, inplace=True)

y = train["class"]
y = y-1
train = train.drop(["class"], axis = 1)

# # poly
xgb_params = {
    'n_trees': 520,
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.93,
    'objective': 'multi:softmax',
    'eval_metric': 'merror',
    'silent': 0,
    "num_class":5
}


xgb_model = xgb.XGBClassifier(xgb_params)
kr = GridSearchCV(xgb_model, cv=5, n_jobs = 6,verbose=1,
                  param_grid={"max_depth": [4,6,8]
                              })

kr.fit(train, y)
print kr.best_params_
print kr.best_score_
print kr.best_estimator_

# {'max_depth': 4}
# 0.68210976479