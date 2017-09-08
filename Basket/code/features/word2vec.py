import pandas as pd
import numpy as np
import gensim
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

print "load data"
train_orders = pd.read_csv("/Users/xiaofeifei/I/Kaggle/Basket/order_products__train.csv")
prior_orders = pd.read_csv("/Users/xiaofeifei/I/Kaggle/Basket/order_products__prior.csv")
products = pd.read_csv("/Users/xiaofeifei/I/Kaggle/Basket/products.csv").set_index('product_id')

print "transform num to str"
train_orders["product_id"] = train_orders["product_id"].astype(str)
prior_orders["product_id"] = prior_orders["product_id"].astype(str)

train_products = train_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())
prior_products = prior_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())


sentences = prior_products.append(train_products)
longest = np.max(sentences.apply(len))
sentences = sentences.values

print "train model"
model = gensim.models.Word2Vec(sentences, size=100, window=longest, min_count=2, workers=4)

vocab = list(model.wv.vocab.keys())
vocab = pd.DataFrame(vocab)

print "train PCA"
pca = PCA(n_components=2)
embeds = pca.fit_transform(model.wv.syn0)
embeds = pd.DataFrame(embeds)

vocab = pd.concat([vocab,embeds], axis=1)
vocab.columns = ["product_id", "pca_1", "pca_2"]

print "save"

vocab.to_csv("/Users/xiaofeifei/I/Kaggle/Basket/word2vec.csv", index=False)

