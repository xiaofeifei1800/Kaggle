# -*- coding: utf-8 -*-
"""
Thanks to tinrtgu for the wonderful base script
Use pypy for faster computations.!
"""
import csv
import sys
from datetime import datetime
from math import log

import Outbrain.Python.FTRL
from Outbrain.Python.FTRL.combine_feature import process_data

csv.field_size_limit(sys.maxsize)


def logloss(p, y):
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)

##############################################################################
# parameters #################################################################
######################################################-66########################

# A, paths
data_path = "/Users/xiaofeifei/I/Kaggle/Outbrain/"
train = data_path+'new_train.csv'               # path to training file
test = data_path+'clicks_test.csv'                 # path to testing file
submission = 'sub_proba.csv'  # path of to be outputted submission file

# B, model
alpha = 0.1  # learning rate
beta = 0   # smoothing parameter for adaptive learning rate
L1 = 0.0    # L1 regularization, larger value means more regularized
L2 = 0.0         # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 20             # number of weights to use
interaction = None     # whether to enable poly2 f gleature interactions

# D, training/validations
epoch = 10      # learn training data for N passes
holdafter = None   # data after date N (exclusive) are used as validation
holdout = None  # use every N training instance for holdout validation

inter = []

# initialize ourselves a learner
learner = Outbrain.Python.FTRL.ftrl_proximal(alpha, beta, L1, L2, D, interaction)

##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()

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

print("document..")
with open(data_path + "documents_meta.csv") as infile:
    document = csv.reader(infile)
    #prcont_header = (prcont.next())[1:]
    document_header = next(document)[1:]
    document_dict = {}
    for ind,row in enumerate(document):
        document_dict[int(row[0])] = row[1:]
        if ind%100000 == 0:
            print("document file : ", ind)
        if ind==10:
            break

    print(len(document_dict))

del document

print("document..")
with open(data_path + "documents_categories.csv") as infile:
    document = csv.reader(infile)
    #prcont_header = (prcont.next())[1:]
    document_cate_header = next(document)[1:]
    document_cate_dict = {}
    for ind,row in enumerate(document):
        document_cate_dict[int(row[0])] = row[1:]
        if ind%100000 == 0:
            print("document file : ", ind)
        if ind==10:
            break
    print(len(document_cate_dict))

del document
print("document..")
with open(data_path + "documents_entities.csv") as infile:
    document = csv.reader(infile)
    #prcont_header = (prcont.next())[1:]
    document_en_header = next(document)[1:]
    document_en_dict = {}
    for ind,row in enumerate(document):
        document_en_dict[int(row[0])] = row[1:]
        if ind%100000 == 0:
            print("document file : ", ind)
        if ind==10:
            break
    print(len(document_en_dict))

del document
print("document..")
with open(data_path + "documents_topics.csv") as infile:
    document = csv.reader(infile)
    #prcont_header = (prcont.next())[1:]
    document_top_header = next(document)[1:]
    document_top_dict = {}
    for ind,row in enumerate(document):
        document_top_dict[int(row[0])] = row[1:]
        if ind%100000 == 0:
            print("document file : ", ind)
        if ind==10:
            break
    print(len(document_top_dict))

del document

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
        if ind==10:
            break
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
        if ind==10000:
            break
    print(len(event_dict))
del events

# start training
for e in range(epoch):
    loss = 0.
    count = 0
    date = 0

    for t, disp_id, ad_id, x, y in process_data(train, D,prcont_dict,prcont_header,event_dict,event_header,leak_uuid_dict,document_header):
        # data is a generator
        #    t: just a instance counter
        # date: you know what this is
        #   ID: id provided in original data
        #    x: features
        #    y: label (click)

        # step 1, get prediction from learner

        p = learner.predict(x)

        if t>4:
            break
        if (holdafter and date > holdafter) or (holdout and t % holdout == 0):
            # step 2-1, calculate validation loss
            #           we do not train with the validation data so that our
            #           validation loss is an accurate estimation
            #
            # holdafter: train instances from day 1 to day N
            #            validate with instances from day N + 1 and after
            #
            # holdout: validate with every N instance, train with others
            loss += logloss(p, y)
            count += 1
        else:
            # step 2-2, update learner with label (click) information
            learner.update(x, p, y)

        if t%1000000 == 0:
            print("Processed : ", t, datetime.now())
        # if t == 100:
        #     break


##############################################################################
# start testing, and build Kaggle's submission file ##########################
##############################################################################

with open(submission, 'w') as outfile:
    outfile.write('display_id,ad_id,clicked\n')
    for t, disp_id, ad_id, x, y in process_data(test, D,prcont_dict,prcont_header,event_dict,event_header,leak_uuid_dict):
        p = learner.predict(x)
        outfile.write('%s,%s,%s\n' % (disp_id, ad_id, str(p)))
        if t%1000000 == 0:
            print("Processed : ", t, datetime.now())

print(datetime.now()-start)