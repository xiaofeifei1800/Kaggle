import numpy as np
import pandas as pd
from contextlib import contextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import time
import re
import string
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from sklearn.model_selection import train_test_split

cont_patterns = [
    (b'US', b'United States'),
    (b'IT', b'Information Technology'),
    (b'(W|w)on\'t', b'will not'),
    (b'(C|c)an\'t', b'can not'),
    (b'(I|i)\'m', b'i am'),
    (b'(A|a)in\'t', b'is not'),
    (b'(\w+)\'ll', b'\g<1> will'),
    (b'(\w+)n\'t', b'\g<1> not'),
    (b'(\w+)\'ve', b'\g<1> have'),
    (b'(\w+)\'s', b'\g<1> is'),
    (b'(\w+)\'re', b'\g<1> are'),
    (b'(\w+)\'d', b'\g<1> would'),
]
patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]

def prepare_for_char_n_gram(text):
    # 1. Go to lower case (only good for english)
    clean = bytes(text.lower(), encoding="utf-8")
    # 2. Drop \n and  \t
    clean = clean.replace(b"\n", b" ")
    clean = clean.replace(b"\t", b" ")
    clean = clean.replace(b"\b", b" ")
    clean = clean.replace(b"\r", b" ")
    # 3. Replace contractions
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)
    # 4. Drop puntuation
    exclude = re.compile(b'[%s]' % re.escape(bytes(string.punctuation, encoding='utf-8')))
    clean = b" ".join([exclude.sub(b'', token) for token in clean.split()])
    # 6. Drop numbers
    clean = re.sub(b"\d+", b" ", clean)
    # 7. Remove extra spaces
    clean = re.sub(b'\s+', b' ', clean)
    # 5. Now replace words by words surrounded by # signs
    # e.g. my name is bond would become #my# #name# #is# #bond#
    clean = re.sub(b" ", b"# #", clean)

    # Remove ending space if any
    clean = re.sub(b'\s+$', b'', clean)

    return str(clean, 'utf-8')


def count_regexp_occ(regexp="", text=None):
    return len(re.findall(regexp, text))


def perform_nlp(df):
    # Check all sorts of content as it may help find toxic comment
    df["ant_slash_n"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\n", x))

    df["raw_word_len"] = df["comment_text"].apply(lambda x: len(x.split()))
    df["raw_char_len"] = df["comment_text"].apply(lambda x: len(x))

    df["nb_upper"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[A-Z]", x))

    # Number of fuck - f..k contains folk, fork,
    df["nb_fk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ff]\S{2}[Kk]", x))

    # Number of suck
    df["nb_sk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ss]\S{2}[Kk]", x))
    df["nb_dk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[dD]ick", x))
    df["nb_you"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Yy]ou\W", x))
    df["nb_mother"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wmother\W", x))
    df["nb_nigger"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wnigger\W", x))

    df["start_with_columns"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"^\:+", x))

    # Check for time stamp
    df["has_timestamp"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\d{2}|:\d{2}", x))
    # Remove timestamp

    # Check for dates 18:44, 8 December 2010
    df["has_date_long"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}", x))
    # Remove dates

    # Check for date short 8 December 2010
    df["has_date_short"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{1,2} \w+ \d{4}", x))

    # Check for http links
    df["has_http"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"http[s]{0,1}://\S+", x))
    # Remove http links

    # check for mail
    df["has_mail"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\S+\@\w+\.\w+", x))
    # remove mail

    df["has_emphasize_equal"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\={2}.+\={2}", x))

    df["has_emphasize_quotes"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\"{4}\S+\"{4}", x))

    df["word_len"] = df["comment_text"].apply(lambda x: len(x.split()))
    df["unique_word_len"] = df["comment_text"].apply(lambda x: float(len(set(x.split()))))
    df["word_ratio"] = df["unique_word_len"] / (df["word_len"] + 1)

    # Now clean comments
    # go lower case
    df["clean_comment"] = df["comment_text"].apply(lambda x: prepare_for_char_n_gram(x))

    # Get the exact length in words and characters
    df["clean_word_len"] = df["clean_comment"].apply(lambda x: len(x.split()))
    df["clean_char_len"] = df["clean_comment"].apply(lambda x: len(x))

    df["clean_chars"] = df["clean_comment"].apply(lambda x: len(set(x)))
    df["clean_chars_ratio"] = df["clean_comment"].apply(lambda x: len(set(x))) / df["clean_comment"].apply(
        lambda x: 1 + min(99, len(x)))

    df["clean_unique_word_len"] = df["clean_comment"].apply(lambda x: float(len(set(x.split()))))
    df["clean_word_ratio"] = df["clean_unique_word_len"] / (df["clean_word_len"] + 1)


def char_analyzer(text):
    tokens = text.split()
    return [token[i: i + 3] for token in tokens for i in range(len(token) - 2)]

if __name__ == '__main__':
    zpolarity = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight',
                 9: 'nine', 10: 'ten'}
    zsign = {-1: 'negative', 0.: 'neutral', 1: 'positive'}
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    train = pd.read_csv('/Users/guoxinli/I/data/data/train_polarity.csv').fillna(' ')
    test = pd.read_csv('/Users/guoxinli/I/data/data/test_polarity.csv').fillna(' ')

    perform_nlp(train)
    perform_nlp(test)

    train_text = train['clean_comment']
    test_text = test['clean_comment']
    all_text = pd.concat([train_text, test_text])

    num_features = [f_ for f_ in train.columns if f_ not in ["comment_text", "clean_comment", "id", "remaining_chars",
                                                             'has_ip_address'] + class_names]
    skl = MinMaxScaler()
    train_num_features = csr_matrix(skl.fit_transform(train[num_features]))
    test_num_features = csr_matrix(skl.fit_transform(test[num_features]))

    train['comment_text'] = train.apply(
        lambda r: str(r['comment_text']) + ' polarity' + zsign[np.sign(r['polarity'])] + zpolarity[
            np.abs(r['polarity'])], axis=1)
    test['comment_text'] = test.apply(
        lambda r: str(r['comment_text']) + ' polarity' + zsign[np.sign(r['polarity'])] + zpolarity[
            np.abs(r['polarity'])], axis=1)

    word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english',
            ngram_range=(1, 2),
            max_features=20000)
    word_vectorizer.fit(all_text)
    train_word_features = word_vectorizer.transform(train_text)
    test_word_features = word_vectorizer.transform(test_text)

    char_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            tokenizer=char_analyzer,
            analyzer='word',
            ngram_range=(1, 1),
            max_features=50000)
    char_vectorizer.fit(all_text)
    train_char_features = char_vectorizer.transform(train_text)
    test_char_features = char_vectorizer.transform(test_text)

    print((train_char_features > 0).sum(axis=1).max())

    csr_trn = hstack([train_char_features, train_word_features, train_num_features]).tocsr()
    csr_sub = hstack([test_char_features,test_word_features, test_num_features]).tocsr()

    submission = pd.DataFrame.from_dict({'id': test['id']})

    params = {
            "objective": "binary",
            'metric': {'auc'},
            "boosting_type": "gbdt",
            "verbosity": -1,
            "num_threads": 4,
            "bagging_fraction": 0.8,
            "feature_fraction": 0.8,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "verbose": -1,
            "min_split_gain": .1,
            "reg_alpha": .1
        }

    for class_name in class_names:

        train_target = train[class_name]
        X_train, X_test, y_train, y_test = train_test_split(csr_trn, train_target, test_size=0.05, random_state=233)

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=700,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=20)
        submission[class_name] = gbm.predict(csr_sub)


    submission.to_csv("lvl0_lgbm_clean_sub_polar.csv", index=False, float_format="%.8f")