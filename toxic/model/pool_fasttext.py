import numpy as np

np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.optimizers import Nadam
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D
from keras.layers import concatenate, BatchNormalization, Average, Dropout, merge
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, PReLU
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import initializers, regularizers, constraints
from keras.engine.topology import Layer, InputSpec
from keras import backend as K

import warnings

warnings.filterwarnings('ignore')

import os

os.environ['OMP_NUM_THREADS'] = '4'

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

EMBEDDING_FILE = '/Users/guoxinli/I/data/data/glove.twitter.27B/glove.twitter.27B.200d.txt'
EMBEDDING_FILE2 = '/Users/guoxinli/I/data/data/crawl-300d-2M.vec'

train = pd.read_csv("/Users/guoxinli/I/data/data/clean_train_v1.csv")
test = pd.read_csv("/Users/guoxinli/I/data/data/clean_test_v1.csv")
submission = pd.read_csv('/Users/guoxinli/I/data/data/sample_submission.csv')

X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values

max_features = 30000
maxlen = 100
embed_size = 200
embed_size2 = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
print(embeddings_index.get('wtf'))
# for word, i in word_index.items():
#     if i >= max_features: continue
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None: embedding_matrix[i] = embedding_vector

# embeddings_index2 = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE2))

# word_index = tokenizer.word_index
# nb_words = min(max_features, len(word_index))
# embedding_matrix2 = np.zeros((nb_words, embed_size2))
# for word, i in word_index.items():
#     if i >= max_features: continue
#     embedding_vector2 = embeddings_index2.get(word)
#     if embedding_vector2 is not None: embedding_matrix2[i] = embedding_vector2


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


def get_model():
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.5)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(lr=1e-3),
                  metrics=['accuracy'])

    # inp = Input(shape=(maxlen,))
    # inp2 = Input(shape=(maxlen,))
    #
    # x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    # x = SpatialDropout1D(0.5)(x)
    # x2 = Embedding(max_features, embed_size2, weights=[embedding_matrix2])(inp2)
    # x2 = SpatialDropout1D(0.5)(x2)
    #
    # x = Bidirectional(GRU(80, return_sequences=True))(x)
    # x2 = Bidirectional(GRU(80, return_sequences=True))(x2)
    #
    # x = BatchNormalization()(x)
    # x2 = BatchNormalization()(x2)
    #
    # avg_pool = GlobalAveragePooling1D()(x)
    # avg_pool2 = GlobalAveragePooling1D()(x2)
    # max_pool = GlobalMaxPooling1D()(x)
    # max_pool2 = GlobalMaxPooling1D()(x2)
    #
    # x = Attention(50)(x)
    # x2 = Attention(50)(x2)
    #
    # conc = concatenate([x, max_pool,avg_pool,avg_pool2, max_pool2, x2])
    # conc = BatchNormalization()(conc)
    # outp = Dense(6, activation="sigmoid")(conc)
    #
    # model = Model(inputs=inp, outputs=outp)
    # model.compile(loss='binary_crossentropy',
    #               optimizer=Nadam(lr=1e-3),
    #               metrics=['accuracy'])

    return model


model = get_model()

batch_size = 32
epochs = 2

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                 callbacks=[RocAuc])

y_pred = model.predict(x_test, batch_size=1024)
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('/Users/guoxinli/I/data/data/submission_twitter_v1_fasttext_dropout_s.csv', index=False)