import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GRU, Embedding, Dropout, Activation, GlobalAveragePooling1D, concatenate, \
    SpatialDropout1D, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

EMBEDDING_FILE='/Users/guoxinli/I/data/data/glove.twitter.27B/glove.twitter.27B.200d.txt'
TRAIN_DATA_FILE="/Users/guoxinli/I/data/data/clean_train_v1.csv"
TEST_DATA_FILE="/Users/guoxinli/I/data/data/clean_test_v1.csv"

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))

embed_size = 200 # how big is each word vector
max_features = 20000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use

train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["Toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


# def get_coefs(word,*arr):
#     return word, np.asarray(arr, dtype='float32')

embeddings_index = {}

for line in open(EMBEDDING_FILE):
    values = line.strip().split()
    if len(values)!=201:
        continue
    word = values[0]
    coefs = np.asarray(values[1:],  dtype='float32')
    embeddings_index[word] = coefs

# embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(GRU(100, return_sequences=True))(x)
max_pool = GlobalMaxPool1D()(x)
avg_pool = GlobalAveragePooling1D()(x)
x = concatenate([avg_pool, max_pool])
x = Dense(100, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_t, y, batch_size=32, epochs=4, validation_split=0.1)

y_test = model.predict([X_te], batch_size=1024, verbose=1)
sample_submission = pd.read_csv('/Users/guoxinli/I/data/data/sample_submission.csv')
sample_submission[list_classes] = y_test
sample_submission.to_csv('/Users/guoxinli/I/data/data/submission_twitter_v1_con_gru.csv', index=False)