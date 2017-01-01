# -*- coding: utf-8 -*-

import csv
import sys
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
import pandas
import pickle

from pyfm import pylibfm

from sklearn.feature_extraction import DictVectorizer

csv.field_size_limit(sys.maxsize)

# TL; DR, the main training process starts on line: 250,
# you may want to start reading the code from there


##############################################################################
# parameters #################################################################
##############################################################################

# A, paths
data_path = "/Users/xiaofeifei/I/Kaggle/Outbrain/"
train = data_path+'clicks_train.csv'               # path to training file
test = data_path+'clicks_test.csv'                 # path to testing file
submission = 'sub_proba_FM.csv'  # path of to be outputted submission file

# B, model
alpha = .1  # learning rate
beta = 0.   # smoothing parameter for adaptive learning rate
L1 = 0.    # L1 regularization, larger value means more regularized
L2 = 0.     # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 20             # number of weights to use
interaction = None     # whether to enable poly2 f gleature interactions

# D, training/validations
epoch = 1       # learn training data for N passes
holdafter = None   # data after date N (exclusive) are used as validation
holdout = None  # use every N training instance for holdout validation


##############################################################################
# class, function, generator definitions #####################################
##############################################################################

class ftrl_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = {}

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g


def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


def data(path, D,prcont_dict,prcont_header,event_dict,event_header,leak_uuid_dict):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''

    for t, row in enumerate(DictReader(open(path))):
        # process id
        disp_id = int(row['display_id'])
        ad_id = int(row['ad_id'])

        # process clicks
        y = 0.
        if 'clicked' in row:
            if row['clicked'] == '1':
                y = 1.
            del row['clicked']

        x = {}
        for key in row:
            x[key]=row[key]

        row = prcont_dict.get(ad_id, [0,0,0])

        # build x
        ad_doc_id = -1
        for ind, val in enumerate(row):
            if ind==0:
                ad_doc_id = int(val)
            x[prcont_header[ind]]=val


        row = event_dict.get(disp_id, [0,0,0,0,0,0,0])

        ## build x
        disp_doc_id = -1
        for ind, val in enumerate(row):
            if ind==0:
                uuid_val = val
            if ind==1:
                disp_doc_id = int(val)
            x[event_header[ind]]=val


        if (ad_doc_id in leak_uuid_dict) and (uuid_val in leak_uuid_dict[ad_doc_id]):

            x['leakage_row_found'] = 1
        else:
            x['leakage_row_found'] = 0

        yield t, disp_id, ad_id, x, y


##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()

# initialize ourselves a learner
learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)

print("Leakage file..")

leak_uuid_dict= {}
with open(data_path+"leak.csv") as infile:
    doc = csv.reader(infile)
    doc.next()
    for ind, row in enumerate(doc):
        doc_id = int(row[0])
        leak_uuid_dict[doc_id] = set(row[1].split(' '))
        if ind%100000==0:
            print("Leakage file : ", ind)
    print(len(leak_uuid_dict))
del doc

print("Content..")
with open(data_path + "promoted_content.csv") as infile:
    prcont = csv.reader(infile)
    #prcont_header = (prcont.next())[1:]
    prcont_header = next(prcont)[1:]
    prcont_dict = {}
    for ind,row in enumerate(prcont):
        prcont_dict[int(row[0])] = row[1:]
        if ind%100000 == 0:
            print(ind)
        # if ind==10000:
        #     break
    print(len(prcont_dict))
del prcont

print("Events..")
with open(data_path + "events.csv") as infile:
    events = csv.reader(infile)
    #events.next()
    next(events)
    event_header = ['uuid', 'document_id', 'platform', 'geo_location', 'loc_country', 'loc_state', 'loc_dma']
    event_dict = {}
    for ind,row in enumerate(events):
        tlist = row[1:3] + row[4:6]
        loc = row[5].split('>')
        if len(loc) == 3:
            tlist.extend(loc[:])
        elif len(loc) == 2:
            tlist.extend( loc[:]+[''])
        elif len(loc) == 1:
            tlist.extend( loc[:]+['',''])
        else:
            tlist.append(['','',''])
        event_dict[int(row[0])] = tlist[:]
        if ind%100000 == 0:
            print("Events : ", ind)
        # if ind==10000:
        #     break
    print(len(event_dict))
del events

# start training
a = []
b = []
for t, disp_id, ad_id, x, y in data(train, D, prcont_dict, prcont_header, event_dict, event_header, leak_uuid_dict):
    # if t > 100:
    #     break
    a.append(x)
    b.append(y)

v = DictVectorizer()
X = v.fit_transform(a)
fm = pylibfm.FM()
fm.fit(X,b)

filename = 'finalized_model.sav'
pickle.dump(fm, open(filename, 'wb'))

# test_x=[]
# disp_id_list = []
# ad_id_list = []
# for t, disp_id, ad_id, x, y in data(test, D, prcont_dict, prcont_header, event_dict, event_header, leak_uuid_dict):
#     # if t > 100:
#     #     break
#     test_x.append(x)
#     disp_id_list.append(disp_id)
#     ad_id_list.append(ad_id)
#
# prob = fm.predict(v.transform(test_x))
#
# percentile_list = pd.DataFrame(
#     {'disp_id': disp_id_list,
#      'ad_id': ad_id_list,
#      'clicked': prob
#     })
#
# percentile_list.to_csv(submission)
