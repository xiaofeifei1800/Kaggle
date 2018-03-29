'''
The core idea behind all blends is "diversity".
By blending some moderate model results (with some weights), we can create a more "diverse" stable results.
Errors from one model will be covered by others. Same goes for all the models.
So, we get more stable results after blendings.
0.9868
'''
import pandas as pd
import numpy as np


# 0.9835 # v2 0.9861
best = pd.read_csv('/Users/guoxinli/I/data/data/toxic-hight-of-blending/hight_of_blending.csv')

# 0.9788
tidy = pd.read_csv('/Users/guoxinli/I/data/data/tidy-xgboost-glmnet-text2vec-lsa/tidy_xgb_glm.csv')

#0.9823
gruglo = pd.read_csv("/Users/guoxinli/I/data/data/pooled-gru-glove-with-preprocessing/submission.csv")

#avenger 0.9825
ave = pd.read_csv("/Users/guoxinli/I/data/data/toxic-avenger/submission.csv")

# textcnn 0.9831
textcnn = pd.read_csv('/Users/guoxinli/I/data/data/submission_twitter_v1_textcnn.csv')

# 0.9837
supbl = pd.read_csv('/Users/guoxinli/I/data/data/blend-of-blends-1/superblend_1.csv')

# 0.9858
oofs = pd.read_csv('/Users/guoxinli/I/data/data/oof-stacking-regime/submission.csv')

# 0.9860
supbl2 = pd.read_csv('/Users/guoxinli/I/data/data/blend-of-blends-2/hight_of_blend_v2.csv')

################## new
# 0.9772
nbsvm = pd.read_csv('/Users/guoxinli/I/data/data/nb-svm/submission.csv')

# 0.9844
r = pd.read_csv('/Users/guoxinli/I/data/data/r/submission.csv')

# 0.9852
avg_en = pd.read_csv('/Users/guoxinli/I/data/data/LGB-GRU-LR Average-Ensemble/submission.csv')

# 0.9860
low_rank = pd.read_csv('/Users/guoxinli/I/data/data/Low Correlation/preprocessed_blend.csv')
######################
# 0.9804
log2 = pd.read_csv('/Users/guoxinli/I/data/data/log2/submission.csv')

# 0.9821
capsule = pd.read_csv('/Users/guoxinli/I/data/data/Capsule_net/submission.csv')
#####################
# 0.9794
log = pd.read_csv('/Users/guoxinli/I/data/data/submission_log_add_ftr.csv')

# 0.9796
lgbm = pd.read_csv('/Users/guoxinli/I/data/data/lgbm-with-words-and-chars-n-gram/lvl0_lgbm_clean_sub.csv')

# 0.9812
wordbtch = pd.read_csv('/Users/guoxinli/I/data/data/wordbatch-fm-ftrl-using-mse-lb-0-9804/lvl0_wordbatch_clean_sub.csv')

# 0.9841
grucnn = pd.read_csv('/Users/guoxinli/I/data/data/bi-gru-cnn-poolings/submission.csv')

# 0.9841
bilst = pd.read_csv('/Users/guoxinli/I/data/data/bidirectional-lstm-with-convolution/submission.csv')

# 0.9842
fast = pd.read_csv('/Users/guoxinli/I/data/data/submission_twitter_v1_fasttext_cells.csv')

#
blend4 = pd.read_csv("/Users/guoxinli/I/data/data/blend_it_all4.csv")
blend5 = pd.read_csv("/Users/guoxinli/I/data/data/blend_it_all5.csv")
blend6 = pd.read_csv("/Users/guoxinli/I/data/data/blend_it_all6.csv")
blend7 = pd.read_csv("/Users/guoxinli/I/data/data/blend_it_all7.csv")

b1 = best.copy()
col = best.columns

col = col.tolist()
col.remove('id')
# for i in col:
#     b1[i] = (fast[i] * 4 + gruglo[i] * 2 + grucnn[i] * 4 + ave[i] + supbl2[i] * 4 + log[i] + wordbtch[i] * 2 + lgbm[
#         i] * 2 + tidy[i] + bilst[i] * 4 + oofs[i] * 5 + textcnn[i]*2 + r[i]*4 + nbsvm[i] + low_rank[i]*5
#              + avg_en[i]*4 + log2[i]*2 + capsule[i]) / 49

for i in col:
    b1[i] = (blend4[i]+blend5[i]+blend6[i]+blend7[i])/4

b1.to_csv('/Users/guoxinli/I/data/data/blend_it_all8.csv', index=False)

