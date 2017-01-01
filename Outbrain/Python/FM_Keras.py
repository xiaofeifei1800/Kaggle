#############################################################################################################
# Created by qqgeogor
# https://www.kaggle.com/qqgeogor
#############################################################################################################

import sys
import numpy as np
from sklearn.base import BaseEstimator
from keras.layers import Input, Embedding, Dense,Flatten, merge,Activation
from keras.models import Model
from keras.regularizers import l2 as l2_reg
from keras import initializations
import itertools
import csv
from datetime import datetime
from csv import DictReader
import pandas as pd
import pickle
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer as DV

csv.field_size_limit(sys.maxsize)

data_path = "/Users/xiaofeifei/I/Kaggle/Outbrain/"
train = data_path+'clicks_train.csv'               # path to training file
test = data_path+'clicks_test.csv'                 # path to testing file

D = 2**20
def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]


def batch_generator(X,y,batch_size=128,shuffle=True):
    sample_size = X[0].shape[0]
    index_array = np.arange(sample_size)
    while 1:
        if shuffle:
            np.random.shuffle(index_array)
        batches = make_batches(sample_size, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            X_batch = [X[i][batch_ids] for i in range(len(X))]
            y_batch = y[batch_ids]
            yield X_batch,y_batch


def test_batch_generator(X,y,batch_size=128):
    sample_size = X[0].shape[0]
    index_array = np.arange(sample_size)
    batches = make_batches(sample_size, batch_size)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start:batch_end]
        X_batch = [X[i][batch_ids] for i in range(len(X))]
        y_batch = y[batch_ids]
        yield X_batch,y_batch


def predict_batch(model,X_t,batch_size=128):
    outcome = []
    for X_batch,y_batch in test_batch_generator(X_t,np.zeros(X_t[0].shape[0]),batch_size=batch_size):
        outcome.append(model.predict(X_batch,batch_size=batch_size))
    outcome = np.concatenate(outcome).ravel()
    return outcome



def build_model(max_features,K=8,solver='adam',l2=0.0,l2_fm = 0.0):

    inputs = []
    flatten_layers=[]
    columns = range(len(max_features))
    for c in columns:
        inputs_c = Input(shape=(1,), dtype='int32',name = 'input_%s'%c)
        num_c = max_features[c]

        embed_c = Embedding(
                        num_c,
                        K,
                        input_length=1,
                        name = 'embed_%s'%c,
                        W_regularizer=l2_reg(l2_fm)
                        )(inputs_c)

        flatten_c = Flatten()(embed_c)

        inputs.append(inputs_c)
        flatten_layers.append(flatten_c)

    fm_layers = []
    for emb1,emb2 in itertools.combinations(flatten_layers, 2):
        dot_layer = merge([emb1,emb2],mode='dot',dot_axes=1)
        fm_layers.append(dot_layer)

    for c in columns:
        num_c = max_features[c]
        embed_c = Embedding(
                        num_c,
                        1,
                        input_length=1,
                        name = 'linear_%s'%c,
                        W_regularizer=l2_reg(l2)
                        )(inputs[c])

        flatten_c = Flatten()(embed_c)

        fm_layers.append(flatten_c)
        
        
    flatten = merge(fm_layers,mode='sum')
    outputs = Activation('sigmoid',name='outputs')(flatten)

    model = Model(input=inputs, output=outputs)

    model.compile(
                optimizer=solver,
                loss= 'binary_crossentropy'
              )

    return model


class KerasFM(BaseEstimator):
    def __init__(self,max_features=[],K=8,solver='adam',l2=0.0,l2_fm = 0.0):
        self.model = build_model(max_features,K,solver,l2=l2,l2_fm = l2_fm)

    def fit(self,X,y,batch_size=128,nb_epoch=10,shuffle=True,verbose=1,validation_data=None):
        self.model.fit(X,y,batch_size=batch_size,nb_epoch=nb_epoch,shuffle=shuffle,verbose=verbose,validation_data=None)

    def fit_generator(self,X,y,batch_size=128,nb_epoch=10,shuffle=True,verbose=1,validation_data=None,callbacks=None):
        tr_gen = batch_generator(X,y,batch_size=batch_size,shuffle=shuffle)
        if validation_data:
            X_test,y_test = validation_data
            te_gen = batch_generator(X_test,y_test,batch_size=batch_size,shuffle=False)
            nb_val_samples = X_test[-1].shape[0]
        else:
            te_gen = None
            nb_val_samples = None

        self.model.fit_generator(
                tr_gen, 
                samples_per_epoch=X[-1].shape[0], 
                nb_epoch=nb_epoch, 
                verbose=verbose, 
                callbacks=callbacks, 
                validation_data=te_gen, 
                nb_val_samples=nb_val_samples, 
                max_q_size=10
                )

    def predict(self,X,batch_size=128):
        y_preds = predict_batch(self.model,X,batch_size=batch_size)
        return y_preds


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
        x = []
        for key in row:
            x.append(abs(hash(key + '_' + row[key])) % D)

        row = prcont_dict.get(ad_id, ['0','0','0'])
        # build x
        ad_doc_id = -1
        for ind, val in enumerate(row):
            if ind==0:
                ad_doc_id = int(val)
            x.append(abs(hash(prcont_header[ind] + '_' + val)) % D)

        row = event_dict.get(disp_id, ['0','0','0','0','0','0','0'])
        ## build x
        disp_doc_id = -1
        for ind, val in enumerate(row):
            if ind==0:
                uuid_val = val
            if ind==1:
                disp_doc_id = int(val)
            x.append(abs(hash(event_header[ind] + '_' + val)) % D)

        if (ad_doc_id in leak_uuid_dict) and (uuid_val in leak_uuid_dict[ad_doc_id]):
            x.append(abs(hash('leakage_row_found_1'))%D)
        else:
            x.append(abs(hash('leakage_row_not_found'))%D)

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
a = []
b = []
c = []
for t, disp_id, ad_id, x, y in data(train, D, prcont_dict, prcont_header, event_dict, event_header, leak_uuid_dict):
    # if t > 105:
    #     break

    a.append(x)
    b.append(y)

a = pd.DataFrame(a)

print "split train"
data1 = a.ix[:,0]
data2 = a.ix[:,1]
data3 = a.ix[:,2]
data4 = a.ix[:,3]
data5 = a.ix[:,4]
data6 = a.ix[:,5]
data7 = a.ix[:,6]
data8 = a.ix[:,7]
data9 = a.ix[:,8]
data10 = a.ix[:,9]

del a

d = [2000000,2000000,2000000,2000000,2000000,2000000,2000000,2000000,
     2000000,2000000]

ml = KerasFM(max_features=d)

print "fit model"
ml.fit([data1,data2,data3,data4,data5,data6,data7,data8,data9,data10],b)

del data1,data2,data3,data4,data5,data6,data7,data8,data9,data10

print "start predict"
a = []
for t, disp_id, ad_id, x, y in data(test, D, prcont_dict, prcont_header, event_dict, event_header, leak_uuid_dict):
    # if t > 105:
    #     break
    a.append(x)
    print y

a = pd.DataFrame(a)

print "split test"
data1 = a.ix[:,0]
data2 = a.ix[:,1]
data3 = a.ix[:,2]
data4 = a.ix[:,3]
data5 = a.ix[:,4]
data6 = a.ix[:,5]
data7 = a.ix[:,6]
data8 = a.ix[:,7]
data9 = a.ix[:,8]
data10 = a.ix[:,9]

del a

prob = ml.predict([data1,data2,data3,data4,data5,data6,data7,data8,data9,data10])

with open('Keras_FM.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(prob)
