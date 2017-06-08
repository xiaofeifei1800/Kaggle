import operator
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
from matplotlib import pylab as plt


def create_feature_map(features):
        outfile = open('xgb.fmap', 'w')
        i = 0
        for feat in features:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1
        outfile.close()

def main():
    input_folder = '/Users/xiaofeifei/I/Kaggle/Quora/'

    X_train  = pd.read_csv(input_folder + 'x_train_git.csv')
    y_train = pd.read_csv(input_folder + "y_train copy.csv")
    y_train = y_train.values
    feature_names = list(X_train.columns.values)
    create_feature_map(feature_names)


    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=4242)

    #UPDownSampling
    pos_train = X_train[y_train == 1]
    neg_train = X_train[y_train == 0]
    X_train = pd.concat((neg_train, pos_train.iloc[:int(0.8*len(pos_train))], neg_train))
    y_train = np.array([0] * neg_train.shape[0] + [1] * pos_train.iloc[:int(0.8*len(pos_train))].shape[0] + [0] * neg_train.shape[0])
    print(np.mean(y_train))
    del pos_train, neg_train

    pos_valid = X_valid[y_valid == 1]
    neg_valid = X_valid[y_valid == 0]
    X_valid = pd.concat((neg_valid, pos_valid.iloc[:int(0.8 * len(pos_valid))], neg_valid))
    y_valid = np.array([0] * neg_valid.shape[0] + [1] * pos_valid.iloc[:int(0.8 * len(pos_valid))].shape[0] + [0] * neg_valid.shape[0])
    print(np.mean(y_valid))
    del pos_valid, neg_valid


    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.02
    params['max_depth'] = 7
    params['subsample'] = 0.6
    params['base_score'] = 0.2
    params['silent'] = 1

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_valid, label=y_valid)
    del X_train, X_valid
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    bst = xgb.train(params, d_train, 2500, watchlist, early_stopping_rounds=10, verbose_eval=50)
    print(log_loss(y_valid, bst.predict(d_valid)))
    del d_train, d_valid

    print('Start predicting')

    x_test = pd.read_csv(input_folder + "x_test_git.csv")


    d_test = xgb.DMatrix(x_test)
    del x_test
    p_test = bst.predict(d_test)
    del d_test
    sub = pd.DataFrame()
    df_test  = pd.read_csv(input_folder + 'test.csv')
    sub['test_id'] = df_test['test_id']
    sub['is_duplicate'] = p_test
    sub.to_csv(input_folder + 'predictions_clean_0.15.csv', index=False)


    print("Features importances...")
    importance = bst.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    ft = pd.DataFrame(importance, columns=['feature', 'fscore'])
    ft.to_csv(input_folder + 'feature_important.csv')
    ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
    plt.gcf().savefig('features_importance.png')

if __name__ == '__main__':
    main()

