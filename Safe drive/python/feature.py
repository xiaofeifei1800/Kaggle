import pandas as pd
import numpy as np
from multiprocessing import *
import warnings
import time
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

#### Load Data
data_path = '/Users/xiaofeifei/I/Kaggle/Safe drive/'
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
print train.shape
print test.shape
# train_stack = pd.read_csv(data_path + 'train_stack.csv')
# test_stack = pd.read_csv(data_path + 'test_stack.csv')
#
# ### merge
# train = train.merge(train_stack, on = 'id', how = 'left')
# test = test.merge(test_stack, on = 'id', how = 'left')
# ###
id_train = train['id'].values
y = train['target'].values
testid = test['id'].values

train.drop(['id', 'target'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

### Drop calc and others
unwanted = train.columns[train.columns.str.startswith('ps_calc_')]

train = train.drop(unwanted, axis=1)
test = test.drop(unwanted, axis=1)

unwanted = ['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin']
train = train.drop(unwanted, axis=1)
test = test.drop(unwanted, axis=1)

### Great Recovery from Pascal's materpiece
### Great Recovery from Pascal's materpiece
### Great Recovery from Pascal's materpiece
### Great Recovery from Pascal's materpiece
### Great Recovery from Pascal's materpiece

def recon(reg):
    integer = int(np.round((40 * reg) ** 2))
    for a in range(32):
        if (integer - a) % 31 == 0:
            A = a
    M = (integer - A) // 31
    return A, M


train['ps_reg_A'] = train['ps_reg_03'].apply(lambda x: recon(x)[0])
train['ps_reg_M'] = train['ps_reg_03'].apply(lambda x: recon(x)[1])
train['ps_reg_A'].replace(19, -1, inplace=True)
train['ps_reg_M'].replace(51, -1, inplace=True)
test['ps_reg_A'] = test['ps_reg_03'].apply(lambda x: recon(x)[0])
test['ps_reg_M'] = test['ps_reg_03'].apply(lambda x: recon(x)[1])
test['ps_reg_A'].replace(19, -1, inplace=True)
test['ps_reg_M'].replace(51, -1, inplace=True)

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
    for c in dcol:
        if '_bin' not in c:  # standard arithmetic
            df[c + str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c + str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)

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

combs = [
    ('ps_reg_01', 'ps_car_02_cat'),
    ('ps_reg_01', 'ps_car_04_cat'),
    ("ps_ind_02_cat", "ps_ind_05_cat"),
    ("ps_ind_04_cat", "ps_ind_05_cat")
]

start = time.time()
for n_c, (f1, f2) in enumerate(combs):
    name1 = f1 + "_plus_" + f2
    print('current feature %60s %4d in %5.1f'
          % (name1, n_c + 1, (time.time() - start) / 60))
    print('\r' * 75)
    train[name1] = train[f1].apply(lambda x: str(x)) + "_" + train[f2].apply(lambda x: str(x))
    test[name1] = test[f1].apply(lambda x: str(x)) + "_" + test[f2].apply(lambda x: str(x))
    # Label Encode
    lbl = LabelEncoder()
    lbl.fit(list(train[name1].values) + list(test[name1].values))
    train[name1] = lbl.transform(list(train[name1].values))
    test[name1] = lbl.transform(list(test[name1].values))


print train.shape
print test.shape


data_path = '/Users/xiaofeifei/I/Kaggle/Safe drive/'

train['target'] = y
train['id'] = id_train
test['id'] = testid

train.to_csv(data_path+"train_all.csv",index=False)
test.to_csv(data_path+"test_all.csv",index=False)

