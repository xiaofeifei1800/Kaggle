import numpy as np
import pandas as pd
from sklearn import *
from textblob import TextBlob

zpolarity = {0:'zero',1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine',10:'ten'}
zsign = {-1:'negative',  0.: 'neutral', 1:'positive'}
#
train = pd.read_csv("/Users/guoxinli/I/data/data/clean_train_v1.csv")
test = pd.read_csv("/Users/guoxinli/I/data/data/clean_test_v1.csv")
submission = pd.read_csv('/Users/guoxinli/I/data/data/sample_submission.csv')
sub1 = pd.read_csv('/Users/guoxinli/I/data/data/submission_ensemble.csv')
#
coly = [c for c in train.columns if c not in ['id','comment_text']]
y = train[coly]
tid = test['id'].values

# train['polarity'] = train['comment_text'].map(lambda x: int(TextBlob(x).sentiment.polarity * 10) if type(x) is not float else 0 )
# test['polarity'] = test['comment_text'].map(lambda x: int(TextBlob(x).sentiment.polarity * 10) if type(x) is not float else 0)
#
# train['comment_text'] = train.apply(lambda r: str(r['comment_text']) + ' polarity' +  zsign[np.sign(r['polarity'])] + zpolarity[np.abs(r['polarity'])], axis=1)
# test['comment_text'] = test.apply(lambda r: str(r['comment_text']) + ' polarity' +  zsign[np.sign(r['polarity'])] + zpolarity[np.abs(r['polarity'])], axis=1)
#
# train.to_csv('/Users/guoxinli/I/data/data/clean_train_polarity.csv', index=False)
# test.to_csv('/Users/guoxinli/I/data/data/clean_test_polarity.csv', index=False)

train = pd.read_csv('/Users/guoxinli/I/data/data/clean_train_polarity.csv')
test = pd.read_csv('/Users/guoxinli/I/data/data/clean_test_polarity.csv')

df = pd.concat([train['comment_text'], test['comment_text']], axis=0)
df = df.fillna("unknown")
nrow = train.shape[0]

tfidf = feature_extraction.text.TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=800000)
data = tfidf.fit_transform(df)

# char_vectorizer = feature_extraction.text.TfidfVectorizer(
#     sublinear_tf=True,
#     strip_accents='unicode',
#     analyzer='char',
#     stop_words='english',
#     ngram_range=(2, 6),
#     max_features=50000)
# char_vectorizer.fit(df)
# train_char_features = char_vectorizer.transform(train_text)
# test_char_features = char_vectorizer.transform(test_text)

model = ensemble.ExtraTreesClassifier(n_jobs=-1, random_state=3)
model.fit(data[:nrow], y)
print(1- model.score(data[:nrow], y))
sub2 = model.predict_proba(data[nrow:])
sub2 = pd.DataFrame([[c[1] for c in sub2[row]] for row in range(len(sub2))]).T
sub2.columns = coly
sub2['id'] = tid
for c in coly:
    sub2[c] = sub2[c].clip(0+1e12, 1-1e12)

#blend 1
sub2.columns = [x+'_' if x not in ['id'] else x for x in sub2.columns]
blend = pd.merge(sub1, sub2, how='left', on='id')
for c in coly:
    blend[c] = blend[c] * 0.8 + blend[c+'_'] * 0.2
    blend[c] = blend[c].clip(0+1e12, 1-1e12)
blend = blend[sub1.columns]

#blend 2
sub2 = blend[:]
sub2.columns = [x+'_' if x not in ['id'] else x for x in sub2.columns]
blend = pd.merge(sub1, sub2, how='left', on='id')
for c in coly:
    blend[c] = np.sqrt(blend[c] * blend[c+'_'])
    blend[c] = blend[c].clip(0+1e12, 1-1e12)
blend = blend[sub1.columns]
blend.to_csv('/Users/guoxinli/I/data/data/submission_avenger_14.csv', index=False)