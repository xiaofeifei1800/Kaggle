# -*- coding: utf-8 -*-
# [399]	train-logloss:0.392087	eval-logloss:0.410417
import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
import operator
from sklearn.cross_validation import train_test_split
from collections import Counter
from nltk.corpus import stopwords
from matplotlib import pylab as plt

RS = 12357
ROUNDS = 315

print("Started")
np.random.seed(RS)
input_folder = '/Users/xiaofeifei/I/Kaggle/Quora/'

def train_xgb(X, y, params):
    print("Will train XGB for {} rounds, RandomSeed: {}".format(ROUNDS, RS))
    x, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RS)

    xg_train = xgb.DMatrix(x, label=y_train)
    xg_val = xgb.DMatrix(X_val, label=y_val)

    watchlist  = [(xg_train,'train'), (xg_val,'eval')]
    return xgb.train(params, xg_train, ROUNDS, watchlist)

def predict_xgb(clr, X_test):
    return clr.predict(xgb.DMatrix(X_test))

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def main():
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.11
    params['max_depth'] = 5
    params['silent'] = 1
    params['seed'] = RS
    params['n_jobs'] = 6


    df_train = pd.read_csv(input_folder + 'train.csv')
    df_test = pd.read_csv(input_folder + 'test.csv')
    x = pd.read_csv(input_folder + 'quora_all.csv')

    feature_names = list(x.columns.values)
    create_feature_map(feature_names)

    x_train = x[:df_train.shape[0]]
    x_test  = x[df_train.shape[0]:]
    y_train = df_train['is_duplicate'].values

    del x, df_train
    if 1: # Now we oversample the negative class - on your own risk of overfitting!
        pos_train = x_train[y_train == 1]
        neg_train = x_train[y_train == 0]
        print len(pos_train)
        print len(neg_train)
        print("Oversampling started for proportion: {}".format(0.1*len(pos_train) / (0.1*(len(pos_train) + len(neg_train)))))
        p = 0.165
        scale = ((0.1*len(pos_train) / (0.1*(len(pos_train) + len(neg_train)))) / p) - 1
        print scale
        while scale > 1:
            neg_train = pd.concat([neg_train, neg_train])
            scale -=1
        neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
        print("Oversampling done, new proportion: {}".format(0.1*len(pos_train) / (0.1*(len(pos_train) + len(neg_train)))))

        x_train = pd.concat([pos_train, neg_train])
        y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
        del pos_train, neg_train

    print("Training data: X_train: {}, Y_train: {}, X_test: {}".format(x_train.shape, len(y_train), x_test.shape))
    clr = train_xgb(x_train, y_train, params)
    preds = predict_xgb(clr, x_test)

    print("Writing output...")
    sub = pd.DataFrame()
    sub['test_id'] = df_test['test_id']
    sub['is_duplicate'] = preds
    sub.to_csv(input_folder + "xgb_seed{}_n{}_fix_add.csv".format(RS, ROUNDS), index=False)

    # print("Features importances...")
    # importance = clr.get_fscore(fmap='xgb.fmap')
    # importance = sorted(importance.items(), key=operator.itemgetter(1))
    # ft = pd.DataFrame(importance, columns=['feature', 'fscore'])
    #
    # ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
    # plt.gcf().savefig('features_importance.png')

main()