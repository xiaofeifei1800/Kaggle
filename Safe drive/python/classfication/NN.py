import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

path = "/Users/xiaofeifei/I/Kaggle/Benz/"
train = pd.read_csv(path+'train_start.csv')
# test = pd.read_csv(path+'test_start.csv')
train.drop(["y"], axis=1, inplace=True)

y = train["class"]

train = train.drop(["class"], axis = 1)
a= RobustScaler()
train = a.fit_transform(train,y)
# # poly
kr = GridSearchCV(MLPClassifier(hidden_layer_sizes=(150, 60), activation='logistic', alpha=0.0001,early_stopping=True, max_iter=1000),
                  cv=5, n_jobs = 6,verbose=1,
                  param_grid={"hidden_layer_sizes": [(25,50),(50,25)],
                              "activation": ['tanh', 'relu', 'logistic']})

kr.fit(train, y)
print kr.best_params_
print kr.best_score_
print kr.best_estimator_

# {'activation': 'relu', 'hidden_layer_sizes': (50, 25)}
# 0.736754573533
