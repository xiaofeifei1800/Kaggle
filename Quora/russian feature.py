# -*- coding: utf-8 -*-
# import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', 500)
import seaborn as sns
# sns.set_style("dark")
# plt.rcParams['figure.figsize'] = 16, 12
from tqdm import tqdm, tqdm_notebook
import itertools as it
import pickle
import glob
import os
import string

from scipy import sparse

import nltk
# import spacy

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, make_scorer
from sklearn.decomposition import TruncatedSVD

from scipy.optimize import minimize

# import eli5
from IPython.display import display

import xgboost as xgb

def plot_real_feature(df, fname):
    fig = plt.figure()
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((3, 2), (2, 0))
    ax4 = plt.subplot2grid((3, 2), (2, 1))
    ax1.set_title('Distribution of %s' % fname, fontsize=20)
    sns.distplot(df.loc[ix_train][fname],
                 bins=50,
                 ax=ax1)
    sns.distplot(df.loc[ix_is_dup][fname],
                 bins=50,
                 ax=ax2,
                 label='is dup')
    sns.distplot(df.loc[ix_not_dup][fname],
                 bins=50,
                 ax=ax2,
                 label='not dup')
    ax2.legend(loc='upper right', prop={'size': 18})
    sns.boxplot(y=fname,
                x='is_duplicate',
                data=df.loc[ix_train],
                ax=ax3)
    sns.violinplot(y=fname,
                   x='is_duplicate',
                   data=df.loc[ix_train],
                   ax=ax4)
    plt.show()



input_folder = '/Users/xiaofeifei/I/Kaggle/Quora/'
# df = pd.read_csv(input_folder+"x_train_russian.csv")

df_train = pd.read_csv(input_folder + 'train_clean.csv',
                       dtype={
                           'question1': np.str,
                           'question2': np.str
                       })
df_train['test_id'] = -1
df_test = pd.read_csv(input_folder + 'test_clean.csv',
                      dtype={
                          'question1': np.str,
                          'question2': np.str
                      })
df_test['id'] = -1
df_test['qid1'] = -1
df_test['qid2'] = -1
df_test['is_duplicate'] = -1

df = pd.concat([df_train, df_test])
df['question1'] = df['question1'].fillna('')
df['question2'] = df['question2'].fillna('')
df['uid'] = np.arange(df.shape[0])
df = df.set_index(['uid'])

del(df_train, df_test)

ix_train = np.where(df['id'] >= 0)[0]
ix_test = np.where(df['id'] == -1)[0]
ix_is_dup = np.where(df['is_duplicate'] == 1)[0]
ix_not_dup = np.where(df['is_duplicate'] == 0)[0]


######### link_function
r1 = 0.174264424749
r0 = 0.825754788586
d = df[df['is_duplicate'] >= 0]['is_duplicate'].value_counts(normalize=True).to_dict()
gamma_0 = r0/d[0]
gamma_1 = r1/d[1]
def link_function(x):
    return gamma_1*x/(gamma_1*x + gamma_0*(1 - x))

def check_model(predictors, data=None, do_scaling=True):
    classifier = lambda: SGDClassifier(
        loss='log',
        penalty='elasticnet',
        fit_intercept=True,
        n_iter=100,
        shuffle=True,
        n_jobs=-1,
        class_weight=None)

    steps = []
    if do_scaling:
        steps.append(('ss', StandardScaler()))
    steps.append(('en', classifier()))

    model = Pipeline(steps=steps)

    parameters = {
        'en__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.02, 0.1, 0.5, 0.9, 1],
        'en__l1_ratio': [0, 0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.75, 0.9, 1]
    }

    folder = StratifiedKFold(n_splits=5, shuffle=True)

    grid_search = GridSearchCV(
        model,
        parameters,
        cv=folder,
        n_jobs=-1,
        verbose=1)
    if data is None:
        grid_search = grid_search.fit(df.loc[ix_train][predictors],
                                      df.loc[ix_train]['is_duplicate'])
    else:
        grid_search = grid_search.fit(data['X'],
                                      data['y'])

    return grid_search
#########
df['len1'] = df['question1'].str.len().astype(np.float32)
df['len2'] = df['question2'].str.len().astype(np.float32)
df['abs_diff_len1_len2'] = np.abs(df['len1'] - df['len2'])
#
# remove far away values
max_in_dup = df.loc[ix_is_dup]['abs_diff_len1_len2'].max()
max_in_not_dups = df.loc[ix_not_dup]['abs_diff_len1_len2'].max()
std_in_dups = df.loc[ix_is_dup]['abs_diff_len1_len2'].std()
replace_value = max_in_dup + 2*std_in_dups
df['abs_diff_len1_len2'] = df['abs_diff_len1_len2'].apply(lambda x: x if x < replace_value else replace_value)

# log abs diff
df['log_abs_diff_len1_len2'] = np.log(df['abs_diff_len1_len2'] + 1)

# ratio len1,2
df['ratio_len1_len2'] = df['len1'].apply(lambda x: x if x > 0.0 else 1.0)/\
                        df['len2'].apply(lambda x: x if x > 0.0 else 1.0)

max_in_dup = df.loc[ix_is_dup]['ratio_len1_len2'].max()
max_in_not_dups = df.loc[ix_not_dup]['ratio_len1_len2'].max()
std_in_dups = df.loc[ix_is_dup]['ratio_len1_len2'].std()
replace_value = max_in_dup + 2*std_in_dups
df['ratio_len1_len2'] = df['ratio_len1_len2'].apply(lambda x: x if x < replace_value else replace_value)
df['ratio_len1_len2'] = df['ratio_len1_len2'].apply(lambda x: x if x < replace_value else replace_value)

# log ratio
df['log_ratio_len1_len2'] = np.log(df['ratio_len1_len2'] + 1)

# build model
predictors = df.columns[7:].tolist()
print predictors

def check_model(predictors):
    classifier = lambda: SGDClassifier(
        loss='log',
        penalty='elasticnet',
        fit_intercept=True,
        n_iter=100,
        shuffle=True,
        n_jobs=-1,
        class_weight=None)

    model = Pipeline(steps=[
        ('ss', StandardScaler()),
        ('en', classifier())
    ])

    parameters = {
        'en__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.02, 0.1, 0.5, 0.9, 1],
        'en__l1_ratio': [0, 0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.75, 0.9, 1]
    }

    folder = StratifiedKFold(n_splits=5, shuffle=True)

    grid_search = GridSearchCV(
        model,
        parameters,
        cv=folder,
        n_jobs=-1,
        verbose=1)
    grid_search = grid_search.fit(df.loc[ix_train][predictors],
                                  df.loc[ix_train]['is_duplicate'])

    return grid_search
#
cv_char = CountVectorizer(ngram_range=(1, 3), analyzer='char')
ch_freq = np.array(cv_char.fit_transform(df['question1'].tolist() + df['question2'].tolist()).sum(axis=0))[0, :]
unigrams = dict([(k, v) for (k, v) in cv_char.vocabulary_.items() if len(k) == 1])
ix_unigrams = np.sort(unigrams.values())
print 'Unigrams:', len(unigrams)
bigrams = dict([(k, v) for (k, v) in cv_char.vocabulary_.items() if len(k) == 2])
ix_bigrams = np.sort(bigrams.values())
print 'Bigrams: ', len(bigrams)
trigrams = dict([(k, v) for (k, v) in cv_char.vocabulary_.items() if len(k) == 3])
ix_trigrams = np.sort(trigrams.values())
print 'Trigrams:', len(trigrams)

m_q1 = cv_char.transform(df['question1'].values)
m_q2 = cv_char.transform(df['question2'].values)


# v_num = (m_q1[:, ix_unigrams] > 0).minimum((m_q2[:, ix_unigrams] > 0)).sum(axis=1)
# v_den = (m_q1[:, ix_unigrams] > 0).maximum((m_q2[:, ix_unigrams] > 0)).sum(axis=1)
# v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]
#
# df['unigram_jaccard'] = v_score
#
# v_num = m_q1[:, ix_unigrams].minimum(m_q2[:, ix_unigrams]).sum(axis=1)
# v_den = m_q1[:, ix_unigrams].sum(axis=1) + m_q2[:, ix_unigrams].sum(axis=1)
# v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]
#
# df['unigram_all_jaccard'] = v_score
#
# v_num = m_q1[:, ix_unigrams].minimum(m_q2[:, ix_unigrams]).sum(axis=1)
# v_den = m_q1[:, ix_unigrams].maximum(m_q2[:, ix_unigrams]).sum(axis=1)
# v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]
#
# df['unigram_all_jaccard_max'] = v_score
#
# #Bigrams
# v_num = (m_q1[:, ix_bigrams] > 0).minimum((m_q2[:, ix_bigrams] > 0)).sum(axis=1)
# v_den = (m_q1[:, ix_bigrams] > 0).maximum((m_q2[:, ix_bigrams] > 0)).sum(axis=1)
# v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]
#
# df['bigram_jaccard'] = v_score
#
# df.loc[df['bigram_jaccard'] < -1.478751, 'bigram_jaccard'] = -1.478751
# df.loc[df['bigram_jaccard'] > 1.0, 'bigram_jaccard'] = 1.0
#
# v_num = m_q1[:, ix_bigrams].minimum(m_q2[:, ix_bigrams]).sum(axis=1)
# v_den = m_q1[:, ix_bigrams].sum(axis=1) + m_q2[:, ix_bigrams].sum(axis=1)
# v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]
#
# df['bigram_all_jaccard'] = v_score
#
# v_num = m_q1[:, ix_bigrams].minimum(m_q2[:, ix_bigrams]).sum(axis=1)
# v_den = m_q1[:, ix_bigrams].maximum(m_q2[:, ix_bigrams]).sum(axis=1)
# v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]
#
# df['bigram_all_jaccard_max'] = v_score

# trigrams
# m_q1 = m_q1[:, ix_trigrams]
# m_q2 = m_q2[:, ix_trigrams]

# v_num = (m_q1[:, ix_trigrams] > 0).minimum((m_q2[:, ix_trigrams] > 0)).sum(axis=1)
# v_den = (m_q1[:, ix_trigrams] > 0).maximum((m_q2[:, ix_trigrams] > 0)).sum(axis=1)
# v_den[np.where(v_den == 0)] = 1
# v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]
#
# df['trigram_jaccard'] = v_score
#
# v_num = m_q1[:, ix_trigrams].minimum(m_q2[:, ix_trigrams]).sum(axis=1)
# v_den = m_q1[:, ix_trigrams].sum(axis=1) + m_q2[:, ix_trigrams].sum(axis=1)
# v_den[np.where(v_den == 0)] = 1
# v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]
#
# df['trigram_all_jaccard'] = v_score
#
# v_num = m_q1[:, ix_trigrams].minimum(m_q2[:, ix_trigrams]).sum(axis=1)
# v_den = m_q1[:, ix_trigrams].maximum(m_q2[:, ix_trigrams]).sum(axis=1)
# v_den[np.where(v_den == 0)] = 1
# v_score = np.array(v_num.flatten()).astype(np.float32)[0, :]/np.array(v_den.flatten())[0, :]
#
# df['trigram_all_jaccard_max'] = v_score

# tfidf on tri
# 1,2,3, 2 and 3 same, not 1
# ix_unigrams, ix_bigrams, ix_trigrams
# m_q1 = m_q1[:, ix_unigrams]
# m_q2 = m_q2[:, ix_unigrams]
##########################
print "start unigram"

tft = TfidfTransformer(
    norm='l2',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1[:, ix_unigrams], m_q2[:, ix_unigrams])))
m_q1_tf = tft.transform(m_q1[:, ix_unigrams])
m_q2_tf = tft.transform(m_q2[:, ix_unigrams])

v_num = np.array(m_q1_tf.multiply(m_q2_tf).sum(axis=1))[:, 0]
v_den = np.array(np.sqrt(m_q1_tf.multiply(m_q1_tf).sum(axis=1)))[:, 0] * \
        np.array(np.sqrt(m_q2_tf.multiply(m_q2_tf).sum(axis=1)))[:, 0]
v_num[np.where(v_den == 0)] = 1
v_den[np.where(v_den == 0)] = 1

v_score = 1 - v_num/v_den

df['unigram_tfidf_cosine'] = v_score

# plot_real_feature(df, 'trigram_tfidf_cosine')

# 1, not too much diff, 2 and 3 same
tft = TfidfTransformer(
    norm='l2',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1[:, ix_unigrams], m_q2[:, ix_unigrams])))
m_q1_tf = tft.transform(m_q1[:, ix_unigrams])
m_q2_tf = tft.transform(m_q2[:, ix_unigrams])
#
v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

df['unigram_tfidf_l2_euclidean'] = v_score

tft = TfidfTransformer(
    norm='l1',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1[:, ix_unigrams], m_q2[:, ix_unigrams])))
m_q1_tf = tft.transform(m_q1[:, ix_unigrams])
m_q2_tf = tft.transform(m_q2[:, ix_unigrams])

v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

df['unigram_tfidf_l1_euclidean'] = v_score


tft = TfidfTransformer(
    norm='l2',
    use_idf=False,
    smooth_idf=True,
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1[:, ix_unigrams], m_q2[:, ix_unigrams])))
m_q1_tf = tft.transform(m_q1[:, ix_unigrams])
m_q2_tf = tft.transform(m_q2[:, ix_unigrams])

v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

df['unigram_tf_l2_euclidean'] = v_score

data = {
    'X_train': sparse.csc_matrix(sparse.hstack((m_q1_tf, m_q2_tf)))[ix_train, :],
    'y_train': df.loc[ix_train]['is_duplicate'],
    'X_test': sparse.csc_matrix(sparse.hstack((m_q1_tf, m_q2_tf)))[ix_test, :],
    'y_train_pred': np.zeros(ix_train.shape[0]),
    'y_test_pred': []
}

n_splits = 10
folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
for ix_first, ix_second in folder.split(np.zeros(data['y_train'].shape[0]), data['y_train']):
    # {'en__l1_ratio': 0.0001, 'en__alpha': 1e-05}
    model = SGDClassifier(
        loss='log',
        penalty='elasticnet',
        fit_intercept=True,
        n_iter=100,
        shuffle=True,
        n_jobs=-1,
        l1_ratio=0.0001,
        alpha=1e-05,
        class_weight=None)
    model = model.fit(data['X_train'][ix_first, :], data['y_train'][ix_first])
    data['y_train_pred'][ix_second] = model.predict_proba(data['X_train'][ix_second, :])[:, 1]
    data['y_test_pred'].append(model.predict_proba(data['X_test'])[:, 1])

print "finish prediction 1"

data['y_test_pred'] = np.array(data['y_test_pred']).T.mean(axis=1)

mp = np.mean(data['y_train_pred'])
print mp

def func(w):

    return (mp*data['y_test_pred'].shape[0] - np.sum(w[0]*data['y_test_pred']/(w[0]*data['y_test_pred'] +
                                                                               (1 - w[0]) * (1 - data['y_test_pred']))))**2

print func(np.array([1]))

res = minimize(func, np.array([1]), method='L-BFGS-B', bounds=[(0, 1)])

print res

w = res['x'][0]

def fix_function(x):
    return w*x/(w*x + (1 - w)*(1 - x))

data['y_test_pred_fixed'] = fix_function(data['y_test_pred'])

df['m_q1_q2_tf_oof_ui'] = np.zeros(df.shape[0])
df.loc[ix_train, 'm_q1_q2_tf_oof_ui'] = data['y_train_pred']
df.loc[ix_test, 'm_q1_q2_tf_oof_ui'] = data['y_test_pred_fixed']
del(data)

######## bi
print "start bigram"

tft = TfidfTransformer(
    norm='l2',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1[:, ix_bigrams], m_q2[:, ix_bigrams])))
m_q1_tf = tft.transform(m_q1[:, ix_bigrams])
m_q2_tf = tft.transform(m_q2[:, ix_bigrams])

v_num = np.array(m_q1_tf.multiply(m_q2_tf).sum(axis=1))[:, 0]
v_den = np.array(np.sqrt(m_q1_tf.multiply(m_q1_tf).sum(axis=1)))[:, 0] * \
        np.array(np.sqrt(m_q2_tf.multiply(m_q2_tf).sum(axis=1)))[:, 0]
v_num[np.where(v_den == 0)] = 1
v_den[np.where(v_den == 0)] = 1

v_score = 1 - v_num/v_den

df['bigram_tfidf_cosine'] = v_score


# plot_real_feature(df, 'trigram_tfidf_cosine')

# 1, not too much diff, 2 and 3 same
tft = TfidfTransformer(
    norm='l2',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1[:, ix_bigrams], m_q2[:, ix_bigrams])))
m_q1_tf = tft.transform(m_q1[:, ix_bigrams])
m_q2_tf = tft.transform(m_q2[:, ix_bigrams])
#
v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

df['bigram_tfidf_l2_euclidean'] = v_score

tft = TfidfTransformer(
    norm='l1',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1[:, ix_bigrams], m_q2[:, ix_bigrams])))
m_q1_tf = tft.transform(m_q1[:, ix_bigrams])
m_q2_tf = tft.transform(m_q2[:, ix_bigrams])

v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

df['bigram_tfidf_l1_euclidean'] = v_score


tft = TfidfTransformer(
    norm='l2',
    use_idf=False,
    smooth_idf=True,
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1[:, ix_bigrams], m_q2[:, ix_bigrams])))
m_q1_tf = tft.transform(m_q1[:, ix_bigrams])
m_q2_tf = tft.transform(m_q2[:, ix_bigrams])

v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

df['bigram_tf_l2_euclidean'] = v_score

data = {
    'X_train': sparse.csc_matrix(sparse.hstack((m_q1_tf, m_q2_tf)))[ix_train, :],
    'y_train': df.loc[ix_train]['is_duplicate'],
    'X_test': sparse.csc_matrix(sparse.hstack((m_q1_tf, m_q2_tf)))[ix_test, :],
    'y_train_pred': np.zeros(ix_train.shape[0]),
    'y_test_pred': []
}

n_splits = 10
folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
for ix_first, ix_second in folder.split(np.zeros(data['y_train'].shape[0]), data['y_train']):
    # {'en__l1_ratio': 0.0001, 'en__alpha': 1e-05}
    model = SGDClassifier(
        loss='log',
        penalty='elasticnet',
        fit_intercept=True,
        n_iter=100,
        shuffle=True,
        n_jobs=-1,
        l1_ratio=0.0001,
        alpha=1e-05,
        class_weight=None)
    model = model.fit(data['X_train'][ix_first, :], data['y_train'][ix_first])
    data['y_train_pred'][ix_second] = model.predict_proba(data['X_train'][ix_second, :])[:, 1]
    data['y_test_pred'].append(model.predict_proba(data['X_test'])[:, 1])

print "finish prediction 1"

data['y_test_pred'] = np.array(data['y_test_pred']).T.mean(axis=1)

mp = np.mean(data['y_train_pred'])
print mp

def func(w):

    return (mp*data['y_test_pred'].shape[0] - np.sum(w[0]*data['y_test_pred']/(w[0]*data['y_test_pred'] +
                                                                               (1 - w[0]) * (1 - data['y_test_pred']))))**2

print func(np.array([1]))

res = minimize(func, np.array([1]), method='L-BFGS-B', bounds=[(0, 1)])

print res

w = res['x'][0]

def fix_function(x):
    return w*x/(w*x + (1 - w)*(1 - x))

data['y_test_pred_fixed'] = fix_function(data['y_test_pred'])

df['m_q1_q2_tf_oof_bi'] = np.zeros(df.shape[0])
df.loc[ix_train, 'm_q1_q2_tf_oof_bi'] = data['y_train_pred']
df.loc[ix_test, 'm_q1_q2_tf_oof_bi'] = data['y_test_pred_fixed']
del(data)

######## tri
print "start trigram"
tft = TfidfTransformer(
    norm='l2',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1[:, ix_trigrams], m_q2[:, ix_trigrams])))
m_q1_tf = tft.transform(m_q1[:, ix_trigrams])
m_q2_tf = tft.transform(m_q2[:, ix_trigrams])

v_num = np.array(m_q1_tf.multiply(m_q2_tf).sum(axis=1))[:, 0]
v_den = np.array(np.sqrt(m_q1_tf.multiply(m_q1_tf).sum(axis=1)))[:, 0] * \
        np.array(np.sqrt(m_q2_tf.multiply(m_q2_tf).sum(axis=1)))[:, 0]
v_num[np.where(v_den == 0)] = 1
v_den[np.where(v_den == 0)] = 1

v_score = 1 - v_num/v_den

df['trigram_tfidf_cosine'] = v_score

# plot_real_feature(df, 'trigram_tfidf_cosine')

# 1, not too much diff, 2 and 3 same
tft = TfidfTransformer(
    norm='l2',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1[:, ix_trigrams], m_q2[:, ix_trigrams])))
m_q1_tf = tft.transform(m_q1[:, ix_trigrams])
m_q2_tf = tft.transform(m_q2[:, ix_trigrams])
#
v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

df['trigram_tfidf_l2_euclidean'] = v_score

tft = TfidfTransformer(
    norm='l1',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1[:, ix_trigrams], m_q2[:, ix_trigrams])))
m_q1_tf = tft.transform(m_q1[:, ix_trigrams])
m_q2_tf = tft.transform(m_q2[:, ix_trigrams])

v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

df['trigram_tfidf_l1_euclidean'] = v_score

df.to_csv(input_folder+"x_train_russian1.csv", index=False)

print "trigram l1"
data = {
    'X_train': sparse.csc_matrix(sparse.hstack((m_q1_tf, m_q2_tf)))[ix_train, :],
    'y_train': df.loc[ix_train]['is_duplicate'],
    'X_test': sparse.csc_matrix(sparse.hstack((m_q1_tf, m_q2_tf)))[ix_test, :],
    'y_train_pred': np.zeros(ix_train.shape[0]),
    'y_test_pred': []
}

n_splits = 10
folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
for ix_first, ix_second in folder.split(np.zeros(data['y_train'].shape[0]), data['y_train']):
    # {'en__l1_ratio': 0.0001, 'en__alpha': 1e-05}
    model = SGDClassifier(
        loss='log',
        penalty='elasticnet',
        fit_intercept=True,
        n_iter=100,
        shuffle=True,
        n_jobs=-1,
        l1_ratio=0.0001,
        alpha=1e-05,
        class_weight=None)
    model = model.fit(data['X_train'][ix_first, :], data['y_train'][ix_first])
    data['y_train_pred'][ix_second] = model.predict_proba(data['X_train'][ix_second, :])[:, 1]
    data['y_test_pred'].append(model.predict_proba(data['X_test'])[:, 1])

print "finish prediction 1"

data['y_test_pred'] = np.array(data['y_test_pred']).T.mean(axis=1)

mp = np.mean(data['y_train_pred'])
print mp

def func(w):

    return (mp*data['y_test_pred'].shape[0] - np.sum(w[0]*data['y_test_pred']/(w[0]*data['y_test_pred'] +
                                                                               (1 - w[0]) * (1 - data['y_test_pred']))))**2

print func(np.array([1]))

res = minimize(func, np.array([1]), method='L-BFGS-B', bounds=[(0, 1)])

print res

w = res['x'][0]

def fix_function(x):
    return w*x/(w*x + (1 - w)*(1 - x))

data['y_test_pred_fixed'] = fix_function(data['y_test_pred'])

df['m_q1_q2_tf_oof_tr'] = np.zeros(df.shape[0])
df.loc[ix_train, 'm_q1_q2_tf_oof_tr'] = data['y_train_pred']
df.loc[ix_test, 'm_q1_q2_tf_oof_tr'] = data['y_test_pred_fixed']
del(data)


tft = TfidfTransformer(
    norm='l2',
    use_idf=False,
    smooth_idf=True,
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1[:, ix_trigrams], m_q2[:, ix_trigrams])))
m_q1_tf = tft.transform(m_q1[:, ix_trigrams])
m_q2_tf = tft.transform(m_q2[:, ix_trigrams])

v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

df['trigram_tf_l2_euclidean'] = v_score

print "finish russain"

df.to_csv(input_folder+"x_train_russian1.csv", index=False)

data = {
    'X_train': sparse.csc_matrix(sparse.hstack((m_q1_tf, m_q2_tf)))[ix_train, :],
    'y_train': df.loc[ix_train]['is_duplicate'],
    'X_test': sparse.csc_matrix(sparse.hstack((m_q1_tf, m_q2_tf)))[ix_test, :],
    'y_train_pred': np.zeros(ix_train.shape[0]),
    'y_test_pred': []
}

n_splits = 10
folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
for ix_first, ix_second in folder.split(np.zeros(data['y_train'].shape[0]), data['y_train']):
    # {'en__l1_ratio': 0.0001, 'en__alpha': 1e-05}
    model = SGDClassifier(
        loss='log',
        penalty='elasticnet',
        fit_intercept=True,
        n_iter=100,
        shuffle=True,
        n_jobs=-1,
        l1_ratio=0.0001,
        alpha=1e-05,
        class_weight=None)
    model = model.fit(data['X_train'][ix_first, :], data['y_train'][ix_first])
    data['y_train_pred'][ix_second] = model.predict_proba(data['X_train'][ix_second, :])[:, 1]
    data['y_test_pred'].append(model.predict_proba(data['X_test'])[:, 1])

print "finish prediction 1"

data['y_test_pred'] = np.array(data['y_test_pred']).T.mean(axis=1)

mp = np.mean(data['y_train_pred'])
print mp

def func(w):

    return (mp*data['y_test_pred'].shape[0] - np.sum(w[0]*data['y_test_pred']/(w[0]*data['y_test_pred'] +
                                                                               (1 - w[0]) * (1 - data['y_test_pred']))))**2

print func(np.array([1]))

res = minimize(func, np.array([1]), method='L-BFGS-B', bounds=[(0, 1)])

print res

w = res['x'][0]

def fix_function(x):
    return w*x/(w*x + (1 - w)*(1 - x))

support = np.linspace(0, 1, 1000)
values = fix_function(support)

fig, ax = plt.subplots()
ax.plot(support, values)
ax.set_title('Fix transformation', fontsize=20)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
plt.show()

data['y_test_pred_fixed'] = fix_function(data['y_test_pred'])

df['m_q1_q2_tf_oof'] = np.zeros(df.shape[0])
df.loc[ix_train, 'm_q1_q2_tf_oof_tr'] = data['y_train_pred']
df.loc[ix_test, 'm_q1_q2_tf_oof_tr'] = data['y_test_pred_fixed']
del(data)

df.to_csv(input_folder+"x_train_russian1.csv", index=False)

# svd = TruncatedSVD(n_components=100)
# m_svd = svd.fit_transform(sparse.csc_matrix(sparse.hstack((m_q1_tf, m_q2_tf))))
#
# data={
#     'X_train': m_svd[ix_train, :],
#     'y_train': df.loc[ix_train]['is_duplicate'],
#     'X_test': m_svd[ix_test, :],
#     'y_train_pred': np.zeros(ix_train.shape[0]),
#     'y_test_pred': []
# }
# n_splits = 10
# folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
# for ix_first, ix_second in folder.split(np.zeros(data['y_train'].shape[0]), data['y_train']):
#     # {'en__l1_ratio': 0.5, 'en__alpha': 1e-05}
#     model = SGDClassifier(
#         loss='log',
#         penalty='elasticnet',
#         fit_intercept=True,
#         n_iter=100,
#         shuffle=True,
#         n_jobs=-1,
#         l1_ratio=0.5,
#         alpha=1e-05,
#         class_weight=None)
#     model = model.fit(data['X_train'][ix_first, :], data['y_train'][ix_first])
#     data['y_train_pred'][ix_second] = model.predict_proba(data['X_train'][ix_second, :])[:, 1]
#     data['y_test_pred'].append(model.predict_proba(data['X_test'])[:, 1])
#
# print "finish prediction 1"
# data['y_test_pred'] = np.array(data['y_test_pred']).T.mean(axis=1)
#
# mp = np.mean(data['y_train_pred'])
# print mp
#
# def func(w):
#     return (mp*data['y_test_pred'].shape[0] -
#             np.sum(w[0]*data['y_test_pred']/(w[0]*data['y_test_pred'] +
#                                              (1 - w[0]) * (1 - data['y_test_pred']))))**2
#
# print func(np.array([1]))
#
# res = minimize(func, np.array([1]), method='L-BFGS-B', bounds=[(0, 1)])
#
# print res
#
# w = res['x'][0]
#
# def fix_function(x):
#     return w*x/(w*x + (1 - w)*(1 - x))
#
# data['y_test_pred_fixed'] = fix_function(data['y_test_pred'])
#
# df['m_q1_q2_tf_svd100_oof'] = np.zeros(df.shape[0])
# df.loc[ix_train, 'm_q1_q2_tf_svd100_oof'] = data['y_train_pred']
# df.loc[ix_test, 'm_q1_q2_tf_svd100_oof'] = data['y_test_pred_fixed']
# del(data)
# df.to_csv(input_folder+"x_train_russian1.csv", index=False)
# del(m_q1, m_q2, m_svd)

# m_diff_q1_q2 = m_q1_tf - m_q2_tf
#
# data={
#     'X_train': m_diff_q1_q2[ix_train, :],
#     'y_train': df.loc[ix_train]['is_duplicate'],
#     'X_test': m_diff_q1_q2[ix_test, :],
#     'y_train_pred': np.zeros(ix_train.shape[0]),
#     'y_test_pred': []
# }
#
# n_splits = 10
#
# folder = StratifiedKFold(n_splits= n_splits, shuffle=True)
# for ix_first, ix_second in tqdm_notebook(folder.split(np.zeros(data['y_train'].shape[0]), data['y_train']),
#                                          total=n_splits):
#     # {'en__l1_ratio': 0.01, 'en__alpha': 0.001}
#     model = SGDClassifier(
#         loss='log',
#         penalty='elasticnet',
#         fit_intercept=True,
#         n_iter=100,
#         shuffle=True,
#         n_jobs=-1,
#         l1_ratio=0.01,
#         alpha=0.001,
#         class_weight=None)
#     model = model.fit(data['X_train'][ix_first, :], data['y_train'][ix_first])
#     data['y_train_pred'][ix_second] = model.predict_proba(data['X_train'][ix_second, :])[:, 1]
#     data['y_test_pred'].append(model.predict_proba(data['X_test'])[:, 1])
#
# print "finish prediction 3"
# data['y_test_pred'] = np.array(data['y_test_pred']).T.mean(axis=1)
#
# df['m_diff_q1_q2_tf_oof'] = np.zeros(df.shape[0])
# df.loc[ix_train, 'm_diff_q1_q2_tf_oof'] = data['y_train_pred']
# df.loc[ix_test, 'm_diff_q1_q2_tf_oof'] = data['y_test_pred']
# del(data)
#
# df.to_csv(input_folder+"x_train_russian1.csv", index=False)
# m_svd_q1 = m_svd[:m_svd.shape[0]/2, :]
# m_svd_q2 = m_svd[m_svd.shape[0]/2:, :]
#
# df['m_vstack_svd_q1_q1_euclidean'] = ((m_svd_q1 - m_svd_q2)**2).mean(axis=1)
# num = (m_svd_q1*m_svd_q2).sum(axis=1)
# den = np.sqrt((m_svd_q1**2).sum(axis=1))*np.sqrt((m_svd_q2**2).sum(axis=1))
# num[np.where(den == 0)] = 0
# den[np.where(den == 0)] = 1
# df['m_vstack_svd_q1_q1_cosine'] = 1 - num/den
#
# m_svd = m_svd_q1*m_svd_q2
#
# data={
#     'X_train': m_svd[ix_train, :],
#     'y_train': df.loc[ix_train]['is_duplicate'],
#     'X_test': m_svd[ix_test, :],
#     'y_train_pred': np.zeros(ix_train.shape[0]),
#     'y_test_pred': []
# }
# del(m_svd)
#
# n_splits = 10
# folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
# for ix_first, ix_second in folder.split(np.zeros(data['y_train'].shape[0]), data['y_train']):
#     # {'en__l1_ratio': 1, 'en__alpha': 1e-05}
#     model = SGDClassifier(
#         loss='log',
#         penalty='elasticnet',
#         fit_intercept=True,
#         n_iter=100,
#         shuffle=True,
#         n_jobs=-1,
#         l1_ratio=1.0,
#         alpha=1e-05,
#         class_weight=None)
#     model = model.fit(data['X_train'][ix_first, :], data['y_train'][ix_first])
#     data['y_train_pred'][ix_second] = model.predict_proba(data['X_train'][ix_second, :])[:, 1]
#     data['y_test_pred'].append(model.predict_proba(data['X_test'])[:, 1])
# print "finish prediction 4"
# data['y_test_pred'] = np.array(data['y_test_pred']).T.mean(axis=1)
#
# mp = np.mean(data['y_train_pred'])
# print mp
#
# def func(w):
#     return (mp*data['y_test_pred'].shape[0] -
#             np.sum(w[0]*data['y_test_pred']/(w[0]*data['y_test_pred'] +
#                                              (1 - w[0]) * (1 - data['y_test_pred']))))**2
#
# print func(np.array([1]))
#
# res = minimize(func, np.array([1]), method='L-BFGS-B', bounds=[(0, 1)])
#
# print res
#
# w = res['x'][0]
#
# def fix_function(x):
#     return w*x/(w*x + (1 - w)*(1 - x))
#
# data['y_test_pred_fixed'] = fix_function(data['y_test_pred'])
#
# df['m_vstack_svd_mult_q1_q2_oof'] = np.zeros(df.shape[0])
# df.loc[ix_train, 'm_vstack_svd_mult_q1_q2_oof'] = data['y_train_pred']
# df.loc[ix_test, 'm_vstack_svd_mult_q1_q2_oof'] = data['y_test_pred_fixed']
# del(data)
#
# df.to_csv(input_folder+"x_train_russian1.csv", index=False)
#
# m_svd = np.abs(m_svd_q1 - m_svd_q2)
#
# data={
#     'X_train': m_svd[ix_train, :],
#     'y_train': df.loc[ix_train]['is_duplicate'],
#     'X_test': m_svd[ix_test, :],
#     'y_train_pred': np.zeros(ix_train.shape[0]),
#     'y_test_pred': []
# }
# del(m_svd, m_svd_q1, m_svd_q2)
#
# n_splits = 10
# folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
# for ix_first, ix_second in folder.split(np.zeros(data['y_train'].shape[0]), data['y_train']):
#     # {'en__l1_ratio': 0.01, 'en__alpha': 1e-05}
#     model = SGDClassifier(
#         loss='log',
#         penalty='elasticnet',
#         fit_intercept=True,
#         n_iter=100,
#         shuffle=True,
#         n_jobs=-1,
#         l1_ratio=0.01,
#         alpha=1e-05,
#         class_weight=None)
#     model = model.fit(data['X_train'][ix_first, :], data['y_train'][ix_first])
#     data['y_train_pred'][ix_second] = model.predict_proba(data['X_train'][ix_second, :])[:, 1]
#     data['y_test_pred'].append(model.predict_proba(data['X_test'])[:, 1])
#
# print "finish prediction 5"
# data['y_test_pred'] = np.array(data['y_test_pred']).T.mean(axis=1)
#
# mp = np.mean(data['y_train_pred'])
# print mp
#
# def func(w):
#     return (mp*data['y_test_pred'].shape[0] -
#             np.sum(w[0]*data['y_test_pred']/(w[0]*data['y_test_pred'] +
#                                              (1 - w[0]) * (1 - data['y_test_pred']))))**2
#
# print func(np.array([1]))
#
# res = minimize(func, np.array([1]), method='L-BFGS-B', bounds=[(0, 1)])
#
# print res
#
# w = res['x'][0]
#
# def fix_function(x):
#     return w*x/(w*x + (1 - w)*(1 - x))
#
# df['m_vstack_svd_absdiff_q1_q2_oof'] = np.zeros(df.shape[0])
# df.loc[ix_train, 'm_vstack_svd_absdiff_q1_q2_oof'] = data['y_train_pred']
# df.loc[ix_test, 'm_vstack_svd_absdiff_q1_q2_oof'] = data['y_test_pred']
# del(data)
# df.to_csv(input_folder+"x_train_russian1.csv", index=False)

# whole data
nlp = spacy.load('en')
df.head()['question1'].apply(lambda s: ' '.join([c.lemma_ for c in nlp(unicode(s)) if c.lemma_  != '?']))

SYMBOLS = set(' '.join(string.punctuation).split(' ') + ['...', '“', '”', '\'ve'])

q1 = []

for doc in nlp.pipe(df['question1'].str.decode('utf-8'), n_threads=6, batch_size=10000):
    q1.append([c.lemma_ for c in doc if c.lemma_ not in SYMBOLS])

q2 = []

for doc in nlp.pipe(df['question2'].str.decode('utf-8'), n_threads=6, batch_size=10000):
    q2.append([c.lemma_ for c in doc if c.lemma_ not in SYMBOLS])

cv_words = CountVectorizer(ngram_range=(1, 1), analyzer='word')
w_freq = np.array(cv_words.fit_transform([' '.join(s) for s in q1] + [' '.join(s) for s in q2]).sum(axis=0))[0, :]

m_q1 = cv_words.transform([' '.join(s) for s in q1])
m_q2 = cv_words.transform([' '.join(s) for s in q2])

tft = TfidfTransformer(
    norm='l2',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1, m_q2)))
m_q1_tf = tft.transform(m_q1)
m_q2_tf = tft.transform(m_q2)

v_num = np.array(m_q1_tf.multiply(m_q2_tf).sum(axis=1))[:, 0]
v_den = np.array(np.sqrt(m_q1_tf.multiply(m_q1_tf).sum(axis=1)))[:, 0] * \
        np.array(np.sqrt(m_q2_tf.multiply(m_q2_tf).sum(axis=1)))[:, 0]
v_num[np.where(v_den == 0)] = 1
v_den[np.where(v_den == 0)] = 1

v_score = 1 - v_num/v_den

df['1wl_tfidf_cosine'] = v_score

tft = TfidfTransformer(
    norm='l2',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1, m_q2)))
m_q1_tf = tft.transform(m_q1)
m_q2_tf = tft.transform(m_q2)

v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

df['1wl_tfidf_l2_euclidean'] = v_score

tft = TfidfTransformer(
    norm='l2',
    use_idf=False,
    smooth_idf=True,
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1, m_q2)))
m_q1_tf = tft.transform(m_q1)
m_q2_tf = tft.transform(m_q2)

v_score = (m_q1_tf - m_q2_tf)
v_score = np.sqrt(np.array(v_score.multiply(v_score).sum(axis=1))[:, 0])

df['1wl_tf_l2_euclidean'] = v_score

tft = TfidfTransformer(
    norm='l2',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=False)

tft = tft.fit(sparse.vstack((m_q1, m_q2)))
m_q1_tf = tft.transform(m_q1)
m_q2_tf = tft.transform(m_q2)

data={
    'X_train': sparse.csc_matrix(sparse.hstack((m_q1_tf, m_q2_tf)))[ix_train, :],
    'y_train': df.loc[ix_train]['is_duplicate'],
    'X_test': sparse.csc_matrix(sparse.hstack((m_q1_tf, m_q2_tf)))[ix_test, :],
    'y_train_pred': np.zeros(ix_train.shape[0]),
    'y_test_pred': []
}
del(m_q1_tf, m_q2_tf)

n_splits = 10
folder = StratifiedKFold(n_splits=n_splits, shuffle=True)
for ix_first, ix_second in folder.split(np.zeros(data['y_train'].shape[0]), data['y_train']):
    # {'en__l1_ratio': 0.0001, 'en__alpha': 1e-05}
    model = SGDClassifier(
        loss='log',
        penalty='elasticnet',
        fit_intercept=True,
        n_iter=100,
        shuffle=True,
        n_jobs=-1,
        l1_ratio=0.0001,
        alpha=1e-05,
        class_weight=None)
    model = model.fit(data['X_train'][ix_first, :], data['y_train'][ix_first])
    data['y_train_pred'][ix_second] = model.predict_proba(data['X_train'][ix_second, :])[:, 1]
    data['y_test_pred'].append(model.predict_proba(data['X_test'])[:, 1])
print "finish prediction all"
data['y_test_pred'] = np.array(data['y_test_pred']).T.mean(axis=1)

mp = np.mean(data['y_train_pred'])
print mp

def func(w):
    return (mp*data['y_test_pred'].shape[0] -
            np.sum(w[0]*data['y_test_pred']/(w[0]*data['y_test_pred'] +
                                             (1 - w[0]) * (1 - data['y_test_pred']))))**2

print func(np.array([1]))

res = minimize(func, np.array([1]), method='L-BFGS-B', bounds=[(0, 1)])

print res

w = res['x'][0]
df['m_w1l_tfidf_oof'] = np.zeros(df.shape[0])
df.loc[ix_train, 'm_w1l_tfidf_oof'] = data['y_train_pred']
df.loc[ix_test, 'm_w1l_tfidf_oof'] = data['y_test_pred_fixed']
del(data)
df.to_csv(input_folder+"x_train_russian1.csv", index=False)