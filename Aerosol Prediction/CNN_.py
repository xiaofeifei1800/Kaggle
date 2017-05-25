import numpy as np
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
import pandas as pd
import csv
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import KFold

X_train = []
y_train = []
X_test = []

with open("/Users/xiaofeifei/I/Kaggle/a/CNN.csv", 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    i = 1
    spamreader.next()
    for row in spamreader:
        if i <=980*7:
            a = np.zeros(shape=(10,10))
            a[0] = row[0:10]
            a[1] = row[10:20]
            a[2] = row[20:30]
            a[3] = row[30:40]
            a[4] = row[40:50]
            a[5] = row[50:60]
            a[6] = row[60:70]
            a[7] = row[70:80]
            a[8] = row[80:90]
            a[9] = row[90:100]
            X_train.append(a)
            y_train.append(row[100])
        else:
            a = np.zeros(shape=(10,10))
            a[0] = row[0:10]
            a[1] = row[10:20]
            a[2] = row[20:30]
            a[3] = row[30:40]
            a[4] = row[40:50]
            a[5] = row[50:60]
            a[6] = row[60:70]
            a[7] = row[70:80]
            a[8] = row[80:90]
            a[9] = row[90:100]
            X_test.append(a)

        i += 1



X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)

X_train = X_train.reshape(X_train.shape[0], 1, 10, 10)
X_test = X_test.reshape(X_test.shape[0], 1, 10, 10)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# def baseline_model():
#     model = Sequential()
#     model.add(Convolution2D(32, 2, 2, activation='sigmoid', input_shape=(1,10,10)))
#
#     model.add(Convolution2D(32, 2, 2, activation='sigmoid'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.25))
#
#     model.add(Flatten())
#     model.add(Dense(128, activation='sigmoid'))
#     model.add(Dropout(0.5))
#     model.add(Dense(1, activation='linear'))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model

# estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=1, batch_size=32, verbose=1)
#
#
# kfold = KFold(n=6860, n_folds=5, shuffle=False)
#
# results = cross_val_predict(estimator, X_train, y_train, cv=kfold)
# print results
# df = pd.DataFrame(results)
# df.to_csv("predictions_train.csv",index=False)


model = Sequential()
model.add(Convolution2D(32, 2, 2, activation='sigmoid', input_shape=(1, 10, 10)))

model.add(Convolution2D(32, 2, 2, activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train,
          batch_size=32, nb_epoch=200, verbose=1)


predictions_train = model.predict(X_train)
predictions_test = model.predict(X_test)

df = pd.DataFrame(predictions_train)
df.to_csv("predictions_train.csv",index=False)

df = pd.DataFrame(predictions_test)
df.to_csv("predictions_test.csv",index=False)
