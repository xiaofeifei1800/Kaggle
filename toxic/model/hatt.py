# author - Richard Liao
# Dec 26 2016
import numpy as np
import pandas as pd

import re

from bs4 import BeautifulSoup

import sys
import os

os.environ['KERAS_BACKEND'] = 'theano'

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializations
from keras.optimizers import Nadam
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from nltk import tokenize


MAX_SENT_LENGTH = 100
MAX_SENTS = 15
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


data_train = pd.read_csv('/Users/guoxinli/I/data/data/train.csv')
data_test = pd.read_csv("/Users/guoxinli/I/data/data/test.csv")
submission = pd.read_csv('/Users/guoxinli/I/data/data/sample_submission.csv')
print(data_train.shape)

labels = data_train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values


reviews = []
texts = []

test_reviews = []
test_texts = []
for idx in range(data_train.comment_text.shape[0]):
    text = BeautifulSoup(data_train.comment_text[idx])
    text = clean_str(text.get_text())
    texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    reviews.append(sentences)

for idx in range(data_test.comment_text.shape[0]):
    text = BeautifulSoup(data_test.comment_text[idx])
    text = clean_str(text.get_text())
    test_texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    test_reviews.append(sentences)



tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(list(texts) + list(test_texts))

data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(reviews):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if word not in tokenizer.word_index:
                    continue
                if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                    data[i, j, k] = tokenizer.word_index[word]
                    k = k + 1

test_data = np.zeros((len(test_texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(test_reviews):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if word not in tokenizer.word_index:
                    continue
                if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                    test_data[i, j, k] = tokenizer.word_index[word]
                    k = k + 1

word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))

# labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


GLOVE_DIR = "/Users/guoxinli/I/data/data/glove.6B"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

# embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector

# embedding_layer = Embedding(len(word_index) + 1,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SENT_LENGTH,
#                             trainable=True)
#
# sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sentence_input)
# l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
# sentEncoder = Model(sentence_input, l_lstm)
#
# review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
# review_encoder = TimeDistributed(sentEncoder)(review_input)
# l_lstm_sent = Bidirectional(LSTM(100))(review_encoder)
# preds = Dense(2, activation='softmax')(l_lstm_sent)
# model = Model(review_input, preds)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['acc'])
#
# print("model fitting - Hierachical LSTM")
# print
# model.summary()
# model.fit(x_train, y_train, validation_data=(x_val, y_val),
#           nb_epoch=10, batch_size=50)

# building Hierachical Attention network
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)


class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializations.get('normal')
        # self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        # self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')

        weighted_input = x * weights.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])


sentence_input = Input(shape=(MAX_SENT_LENGTH,))
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(80, return_sequences=True))(embedded_sequences)
l_dense = TimeDistributed(Dense(100))(l_lstm)
l_att = AttLayer()(l_dense)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH))
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(80, return_sequences=True))(review_encoder)
l_dense_sent = TimeDistributed(Dense(100))(l_lstm_sent)
l_att_sent = AttLayer()(l_dense_sent)
preds = Dense(6, activation="sigmoid")(l_att_sent)
model = Model(review_input, preds)

model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(lr=0.005),
                  metrics=['accuracy'])

print("model fitting - Hierachical attention network")
RocAuc = RocAucEvaluation(validation_data=(x_val, y_val), interval=1)
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=2, batch_size=32,callbacks=[RocAuc])

y_pred = model.predict(test_data, batch_size=1024)

submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('/Users/guoxinli/I/data/data/submission_glove_hatt.csv', index=False)