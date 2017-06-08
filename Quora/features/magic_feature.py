
import pandas as pd
import timeit
from collections import defaultdict
import networkx as nx

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

############## feature 2
print "feature 2"

# train_orig =  pd.read_csv('../input/train.csv', header=0)
# test_orig =  pd.read_csv('../input/test.csv', header=0)

ques = pd.concat([train_orig[['question1', 'question2']], \
        test_orig[['question1', 'question2']]], axis=0).reset_index(drop='index')

q_dict = defaultdict(set)
for i in range(ques.shape[0]):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])

def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

train_orig['q1_q2_intersect'] = train_orig.apply(q1_q2_intersect, axis=1, raw=True)
test_orig['q1_q2_intersect'] = test_orig.apply(q1_q2_intersect, axis=1, raw=True)

x_train = pd.concat([x_train, train_orig["q1_q2_intersect"]], axis=1)
x_test = pd.concat([x_test, test_orig["q1_q2_intersect"]], axis=1)


#####################
print ("feature 3")

df_all = pd.concat([train_orig, test_orig])

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

df_output.index.names = ["qid"]

df_output['max_kcore'] = df_output.apply(lambda row: max(row), axis=1)

x_train.to_csv(input_folder+"x_train.csv", index=False)
x_test.to_csv(input_folder+"x_test.csv", index=False)

data = pd.concat([x_train,x_test])
data['max_kcore'] = df_output[['max_kcore']]

data.to_csv(input_folder+"magic_feature.csv", index=False)