import numpy as np
import pandas as pd

from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
import re
import string



class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# train = pd.read_csv('/Users/guoxinli/I/data/data/train.csv').fillna(' ')
# test = pd.read_csv('/Users/guoxinli/I/data/data/test.csv').fillna(' ')

zpolarity = {0:'zero',1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine',10:'ten'}
zsign = {-1:'negative',  0.: 'neutral', 1:'positive'}

# coly = [c for c in train.columns if c not in ['id','comment_text']]
# y = train[coly]
# tid = test['id'].values

# train['polarity'] = train['comment_text'].map(lambda x: int(TextBlob(x).sentiment.polarity * 10))
# test['polarity'] = test['comment_text'].map(lambda x: int(TextBlob(x).sentiment.polarity * 10))

# train.to_csv('/Users/guoxinli/I/data/data/train_polarity.csv', index=False)
# test.to_csv('/Users/guoxinli/I/data/data/test_polarity.csv', index=False)

train = pd.read_csv('/Users/guoxinli/I/data/data/train_polarity.csv')
test = pd.read_csv('/Users/guoxinli/I/data/data/test_polarity.csv')

train['comment_text'] = train.apply(lambda r: str(r['comment_text']) + ' polarity' +  zsign[np.sign(r['polarity'])] + zpolarity[np.abs(r['polarity'])], axis=1)
test['comment_text'] = test.apply(lambda r: str(r['comment_text']) + ' polarity' +  zsign[np.sign(r['polarity'])] + zpolarity[np.abs(r['polarity'])], axis=1)

print('finishi polarity')

train_text = train['clean_comment']
test_text = test['clean_comment']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

print('finish 1 tfidf')
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

print('finish 2,6 tfidf')

train_features = hstack([train_char_features, train_word_features,train_char_features])
test_features = hstack([test_char_features, test_word_features,test_char_features])

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
for class_name in class_names:
    print(class_name)
    train_target = train[class_name]
    classifier = LogisticRegression(solver='sag')

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))

submission.to_csv('/Users/guoxinli/I/data/data/submission_log_add_ftr.csv', index=False)