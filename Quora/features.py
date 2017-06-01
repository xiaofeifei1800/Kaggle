# """
# Detecting duplicate quora questions
# feature engineering
# @author: Abhishek Thakur
# """
#
# import cPickle
# import pandas as pd
# import numpy as np
# import gensim
# from fuzzywuzzy import fuzz
# from nltk.corpus import stopwords
# from tqdm import tqdm
# from scipy.stats import skew, kurtosis
# from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
# from nltk import word_tokenize
# stop_words = stopwords.words('english')
#
#
#
# def sent2vec(s):
#     words = str(s).lower().decode('utf-8')
#     words = word_tokenize(words)
#     words = [w for w in words if not w in stop_words]
#     words = [w for w in words if w.isalpha()]
#     M = []
#     for w in words:
#         try:
#             M.append(model[w])
#         except:
#             continue
#     M = np.array(M)
#     v = M.sum(axis=0)
#     return v / np.sqrt((v ** 2).sum())
#
#
#
# input_folder = '/Users/xiaofeifei/I/Kaggle/Quora/'
# df_train = pd.read_csv(input_folder + 'train.csv')
# df_test  = pd.read_csv(input_folder + 'test.csv')
# data = pd.concat([df_train, df_test])
#
#
# model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
#
#
# question1_vectors = np.zeros((data.shape[0], 300))
# error_count = 0
#
# for i, q in tqdm(enumerate(data.question1.values)):
#     question1_vectors[i, :] = sent2vec(q)
#
# question2_vectors  = np.zeros((data.shape[0], 300))
# for i, q in tqdm(enumerate(data.question2.values)):
#     question2_vectors[i, :] = sent2vec(q)
#
# data['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
#                                                           np.nan_to_num(question2_vectors))]
#
# data['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
#                                                           np.nan_to_num(question2_vectors))]
#
# data['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
#                                                           np.nan_to_num(question2_vectors))]
#
# data['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
#                                                           np.nan_to_num(question2_vectors))]
#
# data['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
#                                                           np.nan_to_num(question2_vectors))]
#
# data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
#                                                           np.nan_to_num(question2_vectors))]
#
# data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
#                                                           np.nan_to_num(question2_vectors))]
#
# data['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
# data['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
# data['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
# data['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]
#
# cPickle.dump(question1_vectors, open('data/q1_w2v.pkl', 'wb'), -1)
# cPickle.dump(question2_vectors, open('data/q2_w2v.pkl', 'wb'), -1)
#
# data.to_csv('data/quora_features.csv', index=False)
#
#
# ###############################################
# import cPickle
# import pandas as pd
# import numpy as np
# import gensim
# from fuzzywuzzy import fuzz
# from nltk.corpus import stopwords
# from tqdm import tqdm
# from scipy.stats import skew, kurtosis
# from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
# from nltk import word_tokenize
# stop_words = stopwords.words('english')
#
#
# def wmd(s1, s2):
#     s1 = str(s1).lower().split()
#     s2 = str(s2).lower().split()
#     stop_words = stopwords.words('english')
#     s1 = [w for w in s1 if w not in stop_words]
#     s2 = [w for w in s2 if w not in stop_words]
#     return model.wmdistance(s1, s2)
#
#
# def norm_wmd(s1, s2):
#     s1 = str(s1).lower().split()
#     s2 = str(s2).lower().split()
#     stop_words = stopwords.words('english')
#     s1 = [w for w in s1 if w not in stop_words]
#     s2 = [w for w in s2 if w not in stop_words]
#     return norm_model.wmdistance(s1, s2)
#
#
# def sent2vec(s):
#     words = str(s).lower().decode('utf-8')
#     words = word_tokenize(words)
#     words = [w for w in words if not w in stop_words]
#     words = [w for w in words if w.isalpha()]
#     M = []
#     for w in words:
#         try:
#             M.append(model[w])
#         except:
#             continue
#     M = np.array(M)
#     v = M.sum(axis=0)
#     return v / np.sqrt((v ** 2).sum())
#
#
# data = pd.read_csv('data/quora_duplicate_questions.tsv', sep='\t')
# data = data.drop(['id', 'qid1', 'qid2'], axis=1)
#
#
# data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
# data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
# data['diff_len'] = data.len_q1 - data.len_q2
# data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
# data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
# data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
# data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
# data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
# data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
# data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
# data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
# data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
# data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
# data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
# data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
#
#
# model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
# data['wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)
#
#
# norm_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
# norm_model.init_sims(replace=True)
# data['norm_wmd'] = data.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)
#
# question1_vectors = np.zeros((data.shape[0], 300))
# error_count = 0
#
# for i, q in tqdm(enumerate(data.question1.values)):
#     question1_vectors[i, :] = sent2vec(q)
#
# question2_vectors  = np.zeros((data.shape[0], 300))
# for i, q in tqdm(enumerate(data.question2.values)):
#     question2_vectors[i, :] = sent2vec(q)
#
# data['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
#                                                           np.nan_to_num(question2_vectors))]
#
# data['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
#                                                           np.nan_to_num(question2_vectors))]
#
# data['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
#                                                           np.nan_to_num(question2_vectors))]
#
# data['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
#                                                           np.nan_to_num(question2_vectors))]
#
# data['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
#                                                           np.nan_to_num(question2_vectors))]
#
# data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
#                                                           np.nan_to_num(question2_vectors))]
#
# data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
#                                                           np.nan_to_num(question2_vectors))]
#
# data['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
# data['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
# data['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
# data['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]
#
# cPickle.dump(question1_vectors, open('data/q1_w2v.pkl', 'wb'), -1)
# cPickle.dump(question2_vectors, open('data/q2_w2v.pkl', 'wb'), -1)
#
# data.to_csv('data/quora_features.csv', index=False)

########################
# # add more positive
# import numpy as np
# import pandas as pd
# from IPython.display import  display
# from collections import defaultdict
# from itertools import combinations
# pd.set_option('display.max_colwidth',-1)
# input_folder = '/Users/xiaofeifei/I/Kaggle/Quora/'
# train_df=pd.read_csv(input_folder+ 'train.csv')
# print train_df
# train_df.head(2)
# # only duplicated questions
# ddf=train_df[train_df.is_duplicate==1]
# print('Duplicated questions shape:',ddf.shape)
# ddf.head(2)
#
# # get all duplicated questions
# clean_ddf1=ddf[['qid1','question1']].drop_duplicates()
# clean_ddf1.columns=['qid','question']
# clean_ddf2=ddf[['qid2','question2']].drop_duplicates()
# clean_ddf2.columns=['qid','question']
# all_dqdf=clean_ddf1.append(clean_ddf2,ignore_index=True)
# print(all_dqdf.shape)
# all_dqdf.head(2)
#
# # groupby qid1, and then we get all the combinations of id in each group
# dqids12=ddf[['qid1','qid2']]
# df12list=dqids12.groupby('qid1', as_index=False)['qid2'].agg({'dlist':(lambda x: list(x))})
# print(len(df12list))
# d12list=df12list.values
# d12list=[[i]+j for i,j in d12list]
# # get all the combinations of id, like (id1,id2)...
# d12ids=set()
# for ids in d12list:
#     ids_len=len(ids)
#     for i in range(ids_len):
#         for j in range(i+1,ids_len):
#             d12ids.add((ids[i],ids[j]))
# print(len(d12ids))
#
# # the same operation of qid2
# dqids21=ddf[['qid2','qid1']]
# display(dqids21.head(2))
# df21list=dqids21.groupby('qid2', as_index=False)['qid1'].agg({'dlist':(lambda x: list(x))})
# print(len(df21list))
# ids2=df21list.qid2.values
# d21list=df21list.values
# d21list=[[i]+j for i,j in d21list]
# d21ids=set()
# for ids in d21list:
#     ids_len=len(ids)
#     for i in range(ids_len):
#         for j in range(i+1,ids_len):
#             d21ids.add((ids[i],ids[j]))
# len(d21ids)
#
# # merge two set
# dids=list(d12ids | d21ids)
# len(dids)
#
# # let's define union-find function
# def indices_dict(lis):
#     d = defaultdict(list)
#     for i,(a,b) in enumerate(lis):
#         d[a].append(i)
#         d[b].append(i)
#     return d
#
# def disjoint_indices(lis):
#     d = indices_dict(lis)
#     sets = []
#     while len(d):
#         que = set(d.popitem()[1])
#         ind = set()
#         while len(que):
#             ind |= que
#             que = set([y for i in que
#                          for x in lis[i]
#                          for y in d.pop(x, [])]) - ind
#         sets += [ind]
#     return sets
#
# def disjoint_sets(lis):
#     return [set([x for i in s for x in lis[i]]) for s in disjoint_indices(lis)]
#
# # split data into groups, so that each question in each group are duplicated
# did_u=disjoint_sets(dids)
# new_dids=[]
# for u in did_u:
#     new_dids.extend(list(combinations(u,2)))
# len(new_dids)
#
# new_ddf=pd.DataFrame(new_dids,columns=['qid1','qid2'])
# print('New duplicated shape:',new_ddf.shape)
# display(new_ddf.head(2))
#
# # merge with all_dqdf to get question1 description
# new_ddf=new_ddf.merge(all_dqdf,left_on='qid1',right_on='qid',how='left')
# new_ddf.drop('qid',inplace=True,axis=1)
# new_ddf.columns=['qid1','qid2','question1']
# new_ddf.drop_duplicates(inplace=True)
# print(new_ddf.shape)
# new_ddf.head(2)
#
# # the same operation with qid2
# new_ddf=new_ddf.merge(all_dqdf,left_on='qid2',right_on='qid',how='left')
# new_ddf.drop('qid',inplace=True,axis=1)
# new_ddf.columns=['qid1','qid2','question1','question2']
# new_ddf.drop_duplicates(inplace=True)
# print(new_ddf.shape)
# new_ddf.head(2)
#
# new_ddf['is_duplicate']=1
# new_ddf.head(2)
#
# new_ddf.to_csv(input_folder+"extra_dup.csv")


## generate pid for test
# DATA_DIR = '/Users/xiaofeifei/I/Kaggle/Quora/'
#
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from collections import defaultdict
#
# train_orig =  pd.read_csv(DATA_DIR + "train.csv", header=0)
# test_orig =  pd.read_csv(DATA_DIR + "test.csv", header=0)
#
# # "id","qid1","qid2","question1","question2","is_duplicate"
# df_id1 = train_orig[["qid1", "question1"]].drop_duplicates(keep="first").copy().reset_index(drop=True)
# df_id2 = train_orig[["qid2", "question2"]].drop_duplicates(keep="first").copy().reset_index(drop=True)
#
# df_id1.columns = ["qid", "question"]
# df_id2.columns = ["qid", "question"]
#
# print(df_id1.shape, df_id2.shape)
#
# df_id = pd.concat([df_id1, df_id2]).drop_duplicates(keep="first").reset_index(drop=True)
# print(df_id1.shape, df_id2.shape, df_id.shape)
#
#
# import csv
# dict_questions = df_id.set_index('question').to_dict()
# dict_questions = dict_questions["qid"]
#
# new_id = 538000 # df_id["qid"].max() ==> 537933
#
# def get_id(question):
#     global dict_questions
#     global new_id
#
#     if question in dict_questions:
#         return dict_questions[question]
#     else:
#         new_id += 1
#         dict_questions[question] = new_id
#         return new_id
#
# rows = []
# max_lines = 10
# if True:
#     with open(DATA_DIR + "test.csv", 'r') as infile:
#         reader = csv.reader(infile, delimiter=",")
#         header = next(reader)
#         header.append('qid1')
#         header.append('qid2')
#
#         if True:
#             print(header)
#             pos, max_lines = 0, 10*1000*1000
#             for row in reader:
#                 # "test_id","question1","question2"
#                 question1 = row[1]
#                 question2 = row[2]
#
#                 qid1 = get_id(question1)
#                 qid2 = get_id(question2)
#                 row.append(qid1)
#                 row.append(qid2)
#
#                 pos += 1
#                 if pos >= max_lines:
#                     break
#                 rows.append(row)
#
# rows = pd.DataFrame(rows)
# rows.to_csv(DATA_DIR + "test_with_ids.csv")

################
# kcore
import numpy as np
import pandas as pd
import networkx as nx

DATA_DIR = '/Users/xiaofeifei/I/Kaggle/Quora/'

df_train = pd.read_csv(DATA_DIR + "train.csv", usecols=["qid1", "qid2"])

df_test = pd.read_csv(DATA_DIR + "test_with_ids.csv", usecols=["qid1", "qid2"])

df_all = pd.concat([df_train, df_test])

print("df_all.shape:", df_all.shape) # df_all.shape: (2750086, 2)

df = df_all

g = nx.Graph()

g.add_nodes_from(df.qid1)

edges = list(df[['qid1', 'qid2']].to_records(index=False))

g.add_edges_from(edges)

g.remove_edges_from(g.selfloop_edges())

print(len(set(df.qid1)), g.number_of_nodes()) # 4789604

print(len(df), g.number_of_edges()) # 2743365 (after self-edges)

df_output = pd.DataFrame(data=g.nodes(), columns=["qid"])

print("df_output.shape:", df_output.shape)

NB_CORES = 20

for k in range(2, NB_CORES + 1):

    fieldname = "kcore{}".format(k)

    print("fieldname = ", fieldname)

    ck = nx.k_core(g, k=k).nodes()

    print("len(ck) = ", len(ck))

    df_output[fieldname] = 0

    df_output.ix[df_output.qid.isin(ck), fieldname] = k

df_output.to_csv("question_kcores.csv", index=None)


df_cores = pd.read_csv("question_kcores.csv", index_col="qid")

df_cores.index.names = ["qid"]

df_cores['max_kcore'] = df_cores.apply(lambda row: max(row), axis=1)

df_cores[['max_kcore']].to_csv("question_max_kcores.csv") # with index