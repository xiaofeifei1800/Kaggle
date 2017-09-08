import re
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances, linear_kernel
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline, Pipeline, _name_estimators
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoLarsCV
from matplotlib import pylab as plt
import operator
import csv

train = pd.read_csv("/Users/xiaofeifei/I/Kaggle/Basket/products.csv")
print train


def review_to_words(raw_review):
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return (" ".join(meaningful_words))


# Get the number of reviews based on the dataframe column size
num_reviews = train["product_name"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list
for i in xrange(0, num_reviews):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append(review_to_words(train["product_name"][i]))

print "Cleaning and parsing the training set movie reviews...\n"
clean_train_reviews = []
for i in xrange(0, num_reviews):
    # If the index is evenly divisible by 1000, print a message
    if ((i + 1) % 1000 == 0):
        print "Review %d of %d\n" % (i + 1, num_reviews)
    clean_train_reviews.append(review_to_words(train["product_name"][i]))

print "Creating the bag of words...\n"

vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                             ngram_range=(1, 1), max_features=100)
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
train_data_features = pd.DataFrame(train_data_features)

n_comp = 5

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train_data_features)

pca2_results_train = pd.DataFrame(pca2_results_train)
pca2_results_train = pd.concat([train["product_id"],pca2_results_train], axis=1)

pca2_results_train.columns = ["product_id", "pca_n_1", "pca_n_2","pca_n_3", "pca_n_4", "pca_n_5"]

pca2_results_train.to_csv("/Users/xiaofeifei/I/Kaggle/Basket/pca_name.csv", index=False)