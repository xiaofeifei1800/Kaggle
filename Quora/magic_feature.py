import numpy as np
import pandas as pd
import timeit

input_folder = '/Users/xiaofeifei/I/Kaggle/Quora/'
# https://www.kaggle.com/jturkewitz/magic-features-0-03-gain
train_orig =  pd.read_csv(input_folder + 'train_clean.csv', header=0)
test_orig =  pd.read_csv(input_folder + 'test_clean.csv', header=0)

x_train  = pd.read_csv(input_folder + 'x_train.csv')
x_test  = pd.read_csv(input_folder + 'x_test.csv')


tic0=timeit.default_timer()
df1 = train_orig[['question1']].copy()
df2 = train_orig[['question2']].copy()
df1_test = test_orig[['question1']].copy()
df2_test = test_orig[['question2']].copy()

df2.rename(columns = {'question2':'question1'},inplace=True)
df2_test.rename(columns = {'question2':'question1'},inplace=True)

train_questions = df1.append(df2)
train_questions = train_questions.append(df1_test)
train_questions = train_questions.append(df2_test)
#train_questions.drop_duplicates(subset = ['qid1'],inplace=True)
train_questions.drop_duplicates(subset = ['question1'],inplace=True)

train_questions.reset_index(inplace=True,drop=True)
questions_dict = pd.Series(train_questions.index.values,index=train_questions.question1.values).to_dict()
train_cp = train_orig.copy()
test_cp = test_orig.copy()
train_cp.drop(['qid1','qid2'],axis=1,inplace=True)

test_cp['is_duplicate'] = -1
test_cp.rename(columns={'test_id':'id'},inplace=True)
comb = pd.concat([train_cp,test_cp])

comb['q1_hash'] = comb['question1'].map(questions_dict)
comb['q2_hash'] = comb['question2'].map(questions_dict)

q1_vc = comb.q1_hash.value_counts().to_dict()
q2_vc = comb.q2_hash.value_counts().to_dict()

def try_apply_dict(x,dict_to_apply):
    try:
        return dict_to_apply[x]
    except KeyError:
        return 0
#map to frequency space
comb['q1_freq_clean'] = comb['q1_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
comb['q2_freq_clean'] = comb['q2_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))

train_comb = comb[comb['is_duplicate'] >= 0][['id','q1_hash','q2_hash','q1_freq_clean','q2_freq_clean','is_duplicate']]
test_comb = comb[comb['is_duplicate'] < 0][['id','q1_hash','q2_hash','q1_freq_clean','q2_freq_clean']]

x_train = pd.concat([x_train, train_comb[["q1_freq_clean", "q2_freq_clean"]]], axis=1)
x_test = pd.concat([x_test, test_comb[["q1_freq_clean", "q2_freq_clean"]]], axis=1)

# ############## feature 2
# print "feature 2"
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from collections import defaultdict
#
# # train_orig =  pd.read_csv('../input/train.csv', header=0)
# # test_orig =  pd.read_csv('../input/test.csv', header=0)
#
# ques = pd.concat([train_orig[['question1', 'question2']], \
#         test_orig[['question1', 'question2']]], axis=0).reset_index(drop='index')
#
# q_dict = defaultdict(set)
# for i in range(ques.shape[0]):
#         q_dict[ques.question1[i]].add(ques.question2[i])
#         q_dict[ques.question2[i]].add(ques.question1[i])
#
# def q1_q2_intersect(row):
#     return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))
#
# train_orig['q1_q2_intersect'] = train_orig.apply(q1_q2_intersect, axis=1, raw=True)
# test_orig['q1_q2_intersect'] = test_orig.apply(q1_q2_intersect, axis=1, raw=True)
#
# x_train = pd.concat([x_train, train_orig["q1_q2_intersect"]], axis=1)
# x_test = pd.concat([x_test, test_orig["q1_q2_intersect"]], axis=1)
#
#
x_train.to_csv(input_folder+"x_train.csv", index=False)
x_test.to_csv(input_folder+"x_test.csv", index=False)
