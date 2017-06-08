import re
import numpy as np
from nlp_utils import stopwords, english_stemmer, stem_tokens
from feat_utils import try_divide, dump_feat_name
from param_config import config
import pandas as pd



def get_position_list(target, obs):
    """
        Get the list of positions of obs in target
    """
    pos_of_obs_in_target = [0]
    if len(obs) != 0:
        pos_of_obs_in_target = [j for j,w in enumerate(obs, start=1) if w in target]
        if len(pos_of_obs_in_target) == 0:
            pos_of_obs_in_target = [0]
    return pos_of_obs_in_target

def ngrams(words, n):
    return ['_'.join(words[i:i+n]) for i in range(len(words)-n+1)]

######################
## Pre-process data ##
######################
token_pattern = r"(?u)\b\w\w+\b"
#token_pattern = r'\w{1,}'
#token_pattern = r"\w+"
#token_pattern = r"[\w']+"
def preprocess_data(line,
                    token_pattern=token_pattern,
                    exclude_stopword=config.cooccurrence_word_exclude_stopword,
                    encode_digit=False):
    token_pattern = re.compile(token_pattern, flags = re.UNICODE | re.LOCALE)
    ## tokenize
    tokens = [x.lower() for x in token_pattern.findall(line)]
    ## stem
    tokens_stemmed = stem_tokens(tokens, english_stemmer)
    if exclude_stopword:
        tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]
    return tokens_stemmed


def extract_feat(df):
  ## unigram
    print "generate unigram"
    df["query_unigram"] = df['question1'].apply(lambda x: preprocess_data(x))
    df["title_unigram"] = df['question2'].apply(lambda x: preprocess_data(x))
    # df["description_unigram"] = list(df.apply(lambda x: preprocess_data(x["product_description"]), axis=1))
    ## bigram
    print "generate bigram"
    join_str = "_"
    df["query_bigram"] = df['question1'].apply(lambda x: ngrams(x.split(), 2))
    df["title_bigram"] = df['question2'].apply(lambda x: ngrams(x.split(), 2))
    # df["title_bigram"] = list(df.apply(lambda x: ngrams(x["question2"].split(), 2), axis=1))

    # df["description_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["description_unigram"], join_str), axis=1))
    ## trigram
    print "generate trigram"
    join_str = "_"
    df["query_trigram"] = df['question1'].apply(lambda x: ngrams(x.split(), 3))
    df["title_trigram"] = df['question2'].apply(lambda x: ngrams(x.split(), 3))
    # df["description_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["description_unigram"], join_str), axis=1))


    ################################
    ## word count and digit count ##
    ################################
    print "generate word counting features"
    feat_names = ["query", "title"]
    grams = ["unigram", "bigram", "trigram"]
    count_digit = lambda x: sum([1. for w in x if w.isdigit()])
    for feat_name in feat_names:
        for gram in grams:
            ## word count
            df["count_of_%s_%s"%(feat_name,gram)] = list(df.apply(lambda x: len(x[feat_name+"_"+gram]), axis=1))
            df["count_of_unique_%s_%s"%(feat_name,gram)] = list(df.apply(lambda x: len(set(x[feat_name+"_"+gram])), axis=1))
            df["ratio_of_unique_%s_%s"%(feat_name,gram)] = map(try_divide, df["count_of_unique_%s_%s"%(feat_name,gram)], df["count_of_%s_%s"%(feat_name,gram)])

        ## digit count
        df["count_of_digit_in_%s"%feat_name] = list(df.apply(lambda x: count_digit(x[feat_name+"_unigram"]), axis=1))
        df["ratio_of_digit_in_%s"%feat_name] = map(try_divide, df["count_of_digit_in_%s"%feat_name], df["count_of_%s_unigram"%(feat_name)])

    ## description missing indicator
    # df["description_missing"] = list(df.apply(lambda x: int(x["description_unigram"] == ""), axis=1))


    ##############################
    ## intersect word count ##
    ##############################
    print "generate intersect word counting features"
    #### unigram
    for gram in grams:
        for obs_name in feat_names:
            for target_name in feat_names:
                if target_name != obs_name:
                    ## query
                    df["count_of_%s_%s_in_%s"%(obs_name,gram,target_name)] = list(df.apply(lambda x: sum([1. for w in x[obs_name+"_"+gram] if w in set(x[target_name+"_"+gram])]), axis=1))
                    df["ratio_of_%s_%s_in_%s"%(obs_name,gram,target_name)] = map(try_divide, df["count_of_%s_%s_in_%s"%(obs_name,gram,target_name)], df["count_of_%s_%s"%(obs_name,gram)])

        ## some other feat
        df["title_%s_in_query_div_query_%s"%(gram,gram)] = map(try_divide, df["count_of_title_%s_in_query"%gram], df["count_of_query_%s"%gram])
        df["title_%s_in_query_div_query_%s_in_title"%(gram,gram)] = map(try_divide, df["count_of_title_%s_in_query"%gram], df["count_of_query_%s_in_title"%gram])
        # df["description_%s_in_query_div_query_%s"%(gram,gram)] = map(try_divide, df["count_of_description_%s_in_query"%gram], df["count_of_query_%s"%gram])
        # df["description_%s_in_query_div_query_%s_in_description"%(gram,gram)] = map(try_divide, df["count_of_description_%s_in_query"%gram], df["count_of_query_%s_in_description"%gram])


    ######################################
    ## intersect word position feat ##
    ######################################
    print "generate intersect word position features"
    for gram in grams:
        for target_name in feat_names:
            for obs_name in feat_names:
                if target_name != obs_name:
                    pos = list(df.apply(lambda x: get_position_list(x[target_name+"_"+gram], obs=x[obs_name+"_"+gram]), axis=1))
                    ## stats feat on pos
                    df["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = map(np.min, pos)
                    df["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = map(np.mean, pos)
                    df["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = map(np.median, pos)
                    df["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = map(np.max, pos)
                    df["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = map(np.std, pos)
                    ## stats feat on normalized_pos
                    df["normalized_pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
                    df["normalized_pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
                    df["normalized_pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
                    df["normalized_pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
                    df["normalized_pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] , df["count_of_%s_%s" % (obs_name, gram)])


if __name__ == "__main__":

    input_folder = '/Users/xiaofeifei/I/Kaggle/Quora/'
    train  = pd.read_csv(input_folder + 'train_clean.csv')
    test = pd.read_csv(input_folder + 'test_clean.csv')

    dfTrain = pd.concat([train,test])
    dfTrain['question1'] = dfTrain['question1'].fillna('')
    dfTrain['question2'] = dfTrain['question2'].fillna('')
    #######################
    ## Generate Features ##
    #######################
    print("==================================================")
    print("Generate counting features...")


    extract_feat(dfTrain)
    dfTrain.to_csv(input_folder+"count_feature.csv", index=False)
