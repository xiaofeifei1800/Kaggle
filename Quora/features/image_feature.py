import csv
import pip
from gensim import corpora, models, similarities
import pandas as pd
import re
import nltk
import gensim
# from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.cm as cm
from scipy import *
from numpy import *
import numpy as np
from nltk.corpus import stopwords
from sklearn.cross_validation import cross_val_predict as cvp
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

# train_file = "/Users/xiaofeifei/I/Kaggle/Quora/test_with_ids.csv"
# df = pd.read_csv(train_file, index_col="test_id")
#
#
# # import matplotlib.pylab as plt
# questions = dict()
#
# for row in df.iterrows():
#     questions[row[1]['qid1']] = row[1]['question1']
#     questions[row[1]['qid2']] = row[1]['question2']
#
# def basic_cleaning(string):
#     string = str(string)
#     try:
#         string = string.decode('unicode-escape')
#     except Exception:
#         pass
#     string = string.lower()
#     string = re.sub(' +', ' ', string)
#     return string
# sentences = []
# for i in questions:
#     sentences.append(nltk.word_tokenize(basic_cleaning(questions[i])))
#
# model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
#
# tf = dict()
# docf = dict()
# total_docs = 0
# for qid in questions:
#     total_docs += 1
#     toks = nltk.word_tokenize(basic_cleaning(questions[qid]))
#     uniq_toks = set(toks)
#     for i in toks:
#         if i not in tf:
#             tf[i] = 1
#         else:
#             tf[i] += 1
#     for i in uniq_toks:
#         if i not in docf:
#             docf[i] = 1
#         else:
#             docf[i] += 1
#
#
#
# def idf(word):
#     return 1 - np.sqrt(docf[word]/total_docs)
#
# def basic_cleaning(string):
#     string = str(string)
#     string = string.lower()
#     string = re.sub('[0-9\(\)\!\^\%\$\'\"\.;,-\?\{\}\[\]\\/]', ' ', string)
#     string = ' '.join([i for i in string.split() if i not in ["a", "and", "of", "the", "to", "on", "in", "at", "is"]])
#     string = re.sub(' +', ' ', string)
#     return string
#
# def w2v_sim(w1, w2):
#     try:
#         return model.similarity(w1, w2)*idf(w1)*idf(w2)
#     except Exception:
#         return 0.0
#
#
# def surface(row):
#     s1 = row['question1']
#     s2 = row['question2']
#     t1 = list((basic_cleaning(s1)).split())
#     t2 = list((basic_cleaning(s2)).split())
#     print("Q1: "+ s1)
#     print("Q2: "+ s2)
#     print("Duplicate: " + str(row['is_duplicate']))
#
# def img_feature(row):
#     s1 = row['question1']
#     s2 = row['question2']
#     t1 = list((basic_cleaning(s1)).split())
#     t2 = list((basic_cleaning(s2)).split())
#     Z = [[w2v_sim(x, y) for x in t1] for y in t2]
#     a = np.array(Z, order='C')
#     return [np.resize(a,(10,10)).flatten()]
# s = df
#
# img = s.apply(img_feature, axis=1, raw=True)
# pix_col = [[] for y in range(100)]
# for k in img.iteritems():
#         for f in range(len(list(k[1][0]))):
#            pix_col[f].append(k[1][0][f])
#
#
# stops = set(stopwords.words("english"))
#
# def word_match_share(row):
#     q1words = {}
#     q2words = {}
#     for word in str(row['question1']).lower().split():
#         if word not in stops:
#             q1words[word] = 1
#     for word in str(row['question2']).lower().split():
#         if word not in stops:
#             q2words[word] = 1
#     if len(q1words) == 0 or len(q2words) == 0:
#         # The computer-generated chaff includes a few questions that are nothing but stopwords
#         return 0
#     shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
#     shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
#     R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
#     return R
#
# train_word_match = df.apply(word_match_share, axis=1, raw=True)
#
#
#
# x_train = pd.DataFrame()
#
#
#
#
#
#
#
# for g in range(len(pix_col)):
#     x_train['img'+str(g)] = pix_col[g]
#
#
# x_train['word_match'] = train_word_match
# x_train.to_csv("/Users/xiaofeifei/I/Kaggle/Quora/x_test_image.csv")
#
# print "shape of x_train",x_train.shape

X_train = pd.read_csv("/Users/xiaofeifei/I/Kaggle/Quora/x_train_image.csv", nrows = 10)
y_train = pd.read_csv("/Users/xiaofeifei/I/Kaggle/Quora/y_train copy.csv", nrows = 10)
y_train.columns = ["y"]
X_train  = pd.concat([X_train, y_train],axis=1)

pos_train = X_train[X_train["y"] == 1]
neg_train = X_train[X_train["y"] == 0]
# Now we oversample the negative class
# There is likely a much more elegant way to do this...
p = 0.165
scale = ((0.1*len(pos_train) / (0.1*(len(pos_train) + len(neg_train)))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
print(0.1*len(pos_train) / (0.1*len(pos_train) + len(neg_train)))

X_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
del pos_train, neg_train

from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
#
# x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

import xgboost as xgb

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 7
params['silent'] = 1
params['n_jobs'] = 6

y = []
X_train.reset_index(drop=True)
print X_train
kf = KFold(n_splits=10)
for train, test in kf.split(X_train):
    print train
    print X_train[train]
    x_train, x_valid = train_test_split(train, test_size=0.2, random_state=4242)
    # print x_train
    y_train = x_train["y"]
    y_valid = x_valid
    x_train = x_train.drop("y", axis=1,)
    x_valid = x_valid.drop("y", axis=1,)
    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    bst = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=100, verbose_eval=10)
    test = test.drop("y", axis=1,)
    preds = bst.predict(test)
    y.append(preds)

# x_test = pd.read_csv("/Users/xiaofeifei/I/Kaggle/Quora/x_test_image.csv")


# pred.to_csv("/Users/xiaofeifei/I/Kaggle/Quora/image_features.csv")