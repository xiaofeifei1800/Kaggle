import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('/Users/guoxinli/I/data/data/train.csv').fillna(' ')
test = pd.read_csv('/Users/guoxinli/I/data/data/test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
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

print("vector 1")
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
print("vector 2-6")

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})

params = {
    "objective": "binary",
    'metric': {'auc'},
    "boosting_type": "gbdt",
    "num_threads": 4,
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8,
    "learning_rate": 0.05,
    "reg_alpha": .1,
    'max_depth':7,
    'verbose': 1
}


for class_name in class_names:
    print(class_name)
    train_target = train[class_name]

    X_train, X_test, y_train, y_test = train_test_split(train_features, train_target, test_size=0.05, random_state=233)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=500,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=20)

    submission[class_name] = gbm.predict(test_features)

print('Total CV score is {}'.format(np.mean(scores))    )

submission.to_csv('/Users/guoxinli/I/data/data/submission_lgbm.csv', index=False)