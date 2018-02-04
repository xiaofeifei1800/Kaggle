import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import *
import gc
import warnings

warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

#######################################
# Non stacking version, need to change to stacking.
# add new features reconstruct ps_reg_03 and ps_reg_03 interact with ps_car_13
# cate one_hot encoding
# train-gini:0.343229	valid-gini:0.289587 fold 0
#######################################

# Thanks Pascal and the1owl

# Pascal's Recovery https://www.kaggle.com/pnagel/reconstruction-of-ps-reg-03
# Froza's Baseline https://www.kaggle.com/the1owl/forza-baseline

# single XGB LB 0.285 will release soon.

#######################################

#### Load Data
data_path = '/Users/xiaofeifei/I/Kaggle/Safe drive/'
train = pd.read_csv(data_path + 'train_all.csv')
test = pd.read_csv(data_path + 'test_all.csv')

train_stack = pd.read_csv(data_path + 'train_stack.csv')
test_stack = pd.read_csv(data_path + 'test_stack.csv')

### merge
train = train.merge(train_stack, on = 'id', how = 'left')
test = test.merge(test_stack, on = 'id', how = 'left')
###
id_train = train['id'].values
y = train['target'].values
testid = test['id'].values

train.drop(['id', 'target'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

# ### Drop calc
# unwanted = train.columns[train.columns.str.startswith('ps_calc_')]
# train = train.drop(unwanted, axis=1)
# test = test.drop(unwanted, axis=1)
#
#
# ### Great Recovery from Pascal's materpiece
# ### Great Recovery from Pascal's materpiece
# ### Great Recovery from Pascal's materpiece
# ### Great Recovery from Pascal's materpiece
# ### Great Recovery from Pascal's materpiece
#
# def recon(reg):
#     integer = int(np.round((40 * reg) ** 2))
#     for a in range(32):
#         if (integer - a) % 31 == 0:
#             A = a
#     M = (integer - A) // 31
#     return A, M
#
#
# train['ps_reg_A'] = train['ps_reg_03'].apply(lambda x: recon(x)[0])
# train['ps_reg_M'] = train['ps_reg_03'].apply(lambda x: recon(x)[1])
# train['ps_reg_A'].replace(19, -1, inplace=True)
# train['ps_reg_M'].replace(51, -1, inplace=True)
# test['ps_reg_A'] = test['ps_reg_03'].apply(lambda x: recon(x)[0])
# test['ps_reg_M'] = test['ps_reg_03'].apply(lambda x: recon(x)[1])
# test['ps_reg_A'].replace(19, -1, inplace=True)
# test['ps_reg_M'].replace(51, -1, inplace=True)

### Froza's baseline
### Froza's baseline
### Froza's baseline
### Froza's baseline

d_median = train.median(axis=0)
d_mean = train.mean(axis=0)
d_skew = train.skew(axis=0)
one_hot = {c: list(train[c].unique()) for c in train.columns if c not in ['id', 'target']}


def transform_df(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id', 'target']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol] == -1).values, axis=1)
    # for c in dcol:
    #     if '_bin' not in c:  # standard arithmetic
    #         df[c + str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
    #         df[c + str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)

    for c in one_hot:
        if len(one_hot[c]) > 2 and len(one_hot[c]) < 7:
            for val in one_hot[c]:
                df[c + '_oh_' + str(val)] = (df[c].values == val).astype(np.int)
    return df


def multi_transform(df):
    print('Init Shape: ', df.shape)
    p = Pool(cpu_count())
    df = p.map(transform_df, np.array_split(df, cpu_count()))
    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
    p.close();
    p.join()
    print('After Shape: ', df.shape)
    return df


train = multi_transform(train)
test = multi_transform(test)


### Gini

def ginic(actual, pred):
    actual = np.asarray(actual)
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n


def gini_normalized(a, p):
    if p.ndim == 2:
        p = p[:, 1]
    return ginic(a, p) / ginic(a, a)


def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score


### XGB modeling

sub = pd.DataFrame()
sub['id'] = testid
params = {'eta': 0.025, 'max_depth': 4,
          'subsample': 0.9, 'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'min_child_weight': 100,
          'alpha': 4,
          'nthread': 6,
          'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 99, 'silent': True}
##################################
n_splits = 5
folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2016).split(train, y))

S_train = np.zeros((train.shape[0], 1))

S_test_i = np.zeros((test.shape[0], n_splits))

for j, (train_idx, test_idx) in enumerate(folds):

    X_train = train.iloc[train_idx,]
    y_train = y[train_idx,]
    X_holdout = train.iloc[test_idx,]
    y_holdout = y[test_idx,]
    watchlist = [(xgb.DMatrix(X_train, y_train), 'train'), (xgb.DMatrix(X_holdout, y_holdout), 'valid')]

    model = xgb.train(params, xgb.DMatrix(X_train, y_train), 1000, watchlist, feval=gini_xgb, maximize=True,
                  verbose_eval=100, early_stopping_rounds=70)
    print ("Fit %s fold", j)


    y_pred = model.predict(xgb.DMatrix(X_holdout))


    S_train[test_idx, 0] = y_pred

    S_test_i[:, j] = model.predict(xgb.DMatrix(test))

S_test = S_test_i.mean(axis=1)


##############################
val = pd.DataFrame()
val['id'] = id_train
val['target'] = S_train

data_out = '/Users/xiaofeifei/I/Kaggle/Safe drive/stack/'
val.to_csv(data_out+'froza_train_s1.csv', float_format='%.6f', index=False)

# Create submission file
sub = pd.DataFrame()
sub['id'] = testid
sub['target'] = S_test
sub.to_csv(data_out+'froza_test_s1.csv', float_format='%.6f', index=False)
