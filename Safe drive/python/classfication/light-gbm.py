import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss


path = "/Users/xiaofeifei/I/Kaggle/Benz/"
train = pd.read_csv(path+'train_start.csv')
# test = pd.read_csv(path+'test_start.csv')
train.drop(["y"], axis=1, inplace=True)

y = train["class"]

train = train.drop(["class"], axis = 1)
model = LGBMClassifier(boosting_type='gbdt', num_leaves=10, max_depth=4, learning_rate=0.005, n_estimators=40,
                       objective="multiclass",max_bin=25, subsample=0.995, silent=True, nthread=6)

gbm = GridSearchCV(model, cv=5, n_jobs = 6,verbose=1,param_grid={
    "num_leaves": [5,10,15],
    "max_depth": [4,5,6]
})

gbm.fit(train, y)

print gbm.best_params_
print gbm.best_score_
print gbm.best_estimator_

# {'n_estimators': 20, 'learning_rate': 0.01}
# 0.738417676408