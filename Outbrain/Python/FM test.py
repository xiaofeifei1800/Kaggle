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

test_x=[]
disp_id_list = []
ad_id_list = []
for t, disp_id, ad_id, x, y in data(test, D, prcont_dict, prcont_header, event_dict, event_header, leak_uuid_dict):
    # if t > 100:
    #     break
    test_x.append(x)

with open('file.pkl', 'wb') as pickle_file:
    pickle.dump(test_x, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

# prob = fm.predict(v.transform(test_x))
#
# percentile_list = pd.DataFrame(
#     {'disp_id': disp_id_list,
#      'ad_id': ad_id_list,
#      'clicked': prob
#     })

# percentile_list.to_csv(submission)
