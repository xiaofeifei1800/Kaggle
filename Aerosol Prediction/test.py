import pickle
from wmil import SGDWeights
from sklearn.metrics import mean_absolute_error
data = pickle.load(open('/Users/xiaofeifei/I/Kaggle/a/ted_comments.p'))
size = len(data['X'])
k = int(size*0.5)
x_train = data['X'][1:2]
y_train = data['Y'][:k]
print (data['X'][3])

# x_test = data['X'][k:]
# y_test = data['Y'][k:]
# model = SGDWeights(alpha=0.4, momentum=0.0, minib=50)
# model.fit(x_train, y_train)