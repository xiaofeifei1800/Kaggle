# -*- coding: utf-8 -*-
"""
Thanks to tinrtgu for the wonderful base script
Use pypy for faster computations.!
"""
import csv
import sys
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
import pandas as pd
csv.field_size_limit(sys.maxsize)

# TL; DR, the main training process starts on line: 250,
# you may want to start reading the code from there


##############################################################################
# parameters #################################################################
######################################################-66########################

# A, paths
data_path = "/Users/xiaofeifei/I/Kaggle/Outbrain/"
train = data_path+'new_train.csv'               # path to training file
test = data_path+'clicks_test.csv'                 # path to testing file
submission = 'sub_proba.csv'  # path of to be outputted submission file
output = "../all_train.csv"

# C, feature/hash trick
D = 2 ** 20             # number of weights to use
interaction = None     # whether to enable poly2 f gleature interactions

def data(path,D,prcont_dict,event_dict,leak_uuid_dict,document_dict,document_cate_dict,document_en_dict,
         document_top_dict):
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
# column name ['display_id', 'ad_id', 'document_id', 'campaign_id', 'advertiser_id', 'uuid', 'document_id', 'platform',
    # 'geo_location', 'loc_country', 'loc_state', 'loc_dma', 'source_id', 'publisher_id', 'publish_time', 'category_id',
    # 'confidence_level','entity_id', 'confidence_level', 'topic_id', 'confidence_level', 'leakage', 'click']
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

        x = []
        for key in row:
            x.append(row[key])

        row = prcont_dict.get(ad_id, [0,0,0])
        # build x
        ad_doc_id = -1
        for ind, val in enumerate(row):
            if ind==0:
                ad_doc_id = int(val)
            x.append(val)
        document_id = int(row[0])

        row = event_dict.get(disp_id, [0,0,0,0,0,0,0,0,0])

        ## build x
        disp_doc_id = -1
        for ind, val in enumerate(row):
            if ind==0:
                uuid_val = val
            if ind==1:
                disp_doc_id = int(val)
            x.append(val)


        row = document_dict.get(document_id, [0,0,0,0,0,0])

        for ind, val in enumerate(row):
            x.append(val)

        # cate
        row = document_cate_dict.get(document_id, [0,0])

        for ind, val in enumerate(row):
            x.append(val)

        # entities
        row = document_en_dict.get(document_id, [0,0])

        for ind, val in enumerate(row):
            x.append(val)

        # topics
        row = document_top_dict.get(document_id, [0,0])

        for ind, val in enumerate(row):
            x.append(val)

        if (ad_doc_id in leak_uuid_dict) and (uuid_val in leak_uuid_dict[ad_doc_id]):
            x.append(1)
        else:
            x.append(0)

        yield t, disp_id, ad_id, x, y


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
    document_header = next(document)[1:3]
    document_header.append("year")
    document_header.append("month")
    document_header.append("day")
    document_header.append("hour")
    document_dict = {}
    for ind,row in enumerate(document):
        date = row[3]
        document_dict[int(row[0])] = row[1:3]
        year = date[0:4]
        month = date[5:7]
        day = date[8:10]
        hour = date[11:13]
        document_dict[int(row[0])].append(year)
        document_dict[int(row[0])].append(month)
        document_dict[int(row[0])].append(day)
        document_dict[int(row[0])].append(hour)
        if ind%100000 == 0:
            print("document file : ", ind)
        # if ind==3:
        #     break

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
        # if ind==10:
        #     break
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
        # if ind==10:
        #     break
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
        # if ind==10:
        #     break
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
        # if ind==10000:
        #     break
    print(len(prcont_dict))
del prcont

print("Events..")
with open(data_path + "events.csv") as infile:
    events = csv.reader(infile)
    #events.next()
    next(events)
    event_header = ['uuid', 'document_id', 'platform', 'geo_location', 'loc_country', 'loc_state', 'loc_dma'
                    ,'event_hour', 'event_day']
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
        if tlist[2] == '\\N':
            hour = -1
            day = -1
        else:
            time = int(tlist[2])
            hour = (time // (3600 * 1000)) % 24
            day = time // (3600 * 24 * 1000)
        tlist.append(hour)
        tlist.append(day)


        event_dict[int(row[0])] = tlist[:]
        if ind%100000 == 0:
            print("Events : ", ind)
        # if ind==10000:
        #     break
    print(len(event_dict))
del events

train_x = []
for e in range(1):
    loss = 0.
    count = 0
    date = 0

    for t, disp_id, ad_id, x, y in data(train, D,prcont_dict,event_dict,leak_uuid_dict,document_dict,document_cate_dict,document_en_dict,
         document_top_dict):
        # data is a generator
        #    t: just a instance counter
        # date: you know what this is
        #   ID: id provided in original data
        #    x: features
        #    y: label (click)
        y = int(y)

        x.append(y)
        train_x.append(x)
        if t%1000000 == 0:
            print("Processed : ", t, datetime.now())
        # if t == 10:
        #     break
train_x = pd.DataFrame(train_x)

train_x.columns = ['display_id', 'ad_id', 'document_id', 'campaign_id', 'advertiser_id', 'uuid', 'event_document_id',
                   'platform',
    'geo_location', 'loc_country', 'loc_state', 'loc_dma','event_hour','event_day','source_id', 'publisher_id',
    "year","month","day",'hour', 'category_id','confidence_level','entity_id', 'confidence_level', 'topic_id',
                   'confidence_level', 'leakage', 'clicked']

train_x.to_csv(output)
