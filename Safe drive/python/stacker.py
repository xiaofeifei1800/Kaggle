import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

train = pd.read_csv('/Users/xiaofeifei/I/Kaggle/Safe drive/train_stack_s1.csv')
test = pd.read_csv('/Users/xiaofeifei/I/Kaggle/Safe drive/test_stack_s1.csv')

id_test = test['id'].values
id_train = train['id'].values
y = train['target'].values

train = train.drop(['target','id'], axis = 1)
test = test.drop(['id'], axis = 1)

log_model = LogisticRegression()
results = cross_val_score(log_model, train, y, cv=3, scoring='roc_auc')

log_model.fit(train, y)

pred = log_model.predict_proba(test)[:,1]

sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = pred

sub.to_csv('/Users/xiaofeifei/I/Kaggle/Safe drive/stack_submit.csv', index=False)












