"""
Ridge Regression on TfIDF of text features and One-Hot-Encoded Categoricals
"""

import pandas as pd
import numpy as np
import scipy
import math

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer

import gc


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))

def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")

NUM_BRANDS = 2500
NAME_MIN_DF = 1
MAX_FEAT_DESCP = 50000

print("Reading in Data")

df_train = pd.read_csv('/Users/guoxinli/I/data/mercari/train.tsv', sep='\t')
df_test = pd.read_csv('/Users/guoxinli/I/data/mercari/test.tsv', sep='\t', nrows=100)

df = pd.concat([df_train, df_test], 0)
nrow_train = df_train.shape[0]
y_train = np.log1p(df_train["price"])

del df_train
gc.collect()


df["category_name"] = df["category_name"].fillna("Other").astype("category")
# df["level_1_cat"] = df["level_1_cat"].fillna("Other").astype("category")
# df["level_2_cat"] = df["level_2_cat"].fillna("Other").astype("category")
# df["level_3_cat"] = df["level_3_cat"].fillna("Other").astype("category")

df["brand_name"] = df["brand_name"].fillna("unknown")

pop_brands = df["brand_name"].value_counts().index[:NUM_BRANDS]
df.loc[~df["brand_name"].isin(pop_brands), "brand_name"] = "Other"

df["item_description"] = df["item_description"].fillna("None")
df["item_condition_id"] = df["item_condition_id"].astype("category")
df["brand_name"] = df["brand_name"].astype("category")

print("Encodings")
count = CountVectorizer(min_df=NAME_MIN_DF)
X_name = count.fit_transform(df["name"])

print("Category Encoders")
unique_categories = pd.Series("/".join(df["category_name"].unique().astype("str")).split("/")).unique()
count_category = CountVectorizer()
X_category = count_category.fit_transform(df["category_name"])

vect_cat = LabelBinarizer(sparse_output=True)
# X_level_1 = vect_cat.fit_transform(df["level_1_cat"])
# X_level_2 = vect_cat.fit_transform(df["level_2_cat"])
# X_level_3 = vect_cat.fit_transform(df["level_3_cat"])

print("Descp encoders")
count_descp = TfidfVectorizer(max_features = MAX_FEAT_DESCP,
                              ngram_range = (1,3),
                              stop_words = "english")
X_descp = count_descp.fit_transform(df["item_description"])

print("Brand encoders")
vect_brand = LabelBinarizer(sparse_output=True)
X_brand = vect_brand.fit_transform(df["brand_name"])

print("Dummy Encoders")
X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(df[[
    "item_condition_id", "shipping"]], sparse = True).values)

X = scipy.sparse.hstack((X_dummies,
                         X_descp,
                         X_brand,
                         X_category,
                         # X_level_1,
                         # X_level_2,
                         # X_level_3,
                         X_name)).tocsr()

print([X_dummies.shape, X_category.shape,
       X_name.shape, X_descp.shape, X_brand.shape])

X_train = X[:nrow_train]
model = Ridge(solver = "lsqr", fit_intercept=False)

print("hold out dataset")
X_train, X_hold, y_train, y_hold = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
print("Fitting Model")
model.fit(X_train, y_train)

print("evaluation")
preds_hold = model.predict(X_hold)

print(rmsle(y=np.expm1(y_hold.values), y0=np.expm1(preds_hold)))

X_test = X[nrow_train:]
preds = model.predict(X_test)

df_test["price"] = np.expm1(preds)
# df_test[["test_id", "price"]].to_csv("submission_ridge.csv", index = False)