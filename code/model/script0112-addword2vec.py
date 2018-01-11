# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math  
import scipy
import nltk
import psutil  # to get process information
import time
import gc  # to collect garbage in the memory
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge, Lasso
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ----- utility functions -----
# convert seconds to 'xx seconds'/'yy minutes'/etc.
def get_duration_str(secs):
    if secs <= 60:
        ret = "{} seconds".format(round(secs, 1))
    elif secs <= 3600:
        ret = "{} minutes".format(round(secs/60.0, 1))
    else:
        ret = "{} hours".format(round(secs/3600.0, 1))
    return ret


# Copied from stackoverflow, to declare the specific source later
# convert bytes to 'xx MB'/'yy GB'/etc.
def get_size_str(num_bytes, units=None):
    assert num_bytes>=0, "Negative size is not allowd"
    if not units:
        units = ['B', 'KB', 'MB', 'GB']
    return "{:.2f}{}".format(num_bytes, units[0]) if num_bytes < 1024 else get_size_str(num_bytes/1024.0, units[1:])


# get number of bytes occupied by the current process
def get_memory_bytes():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


# get the string representation of memory occupied by the current process
def get_memory_str():
    return get_size_str(get_memory_bytes())


# convert size difference in bytes to '+xxMB'/'-yyKB'/etc.
def get_size_diff_str(num_bytes):
    sign = '+' if num_bytes >= 0 else '-'
    return "{}{}".format(sign, get_size_str(abs(num_bytes)))


# run a function with its execution time and memory usage printed
# TODO: to use as a wrapper
def run_func(func, **params):
    # TODO: do not take idle time into account
    _t = time.time()  # start time
    _m = get_memory_bytes()
    ret = func(**params)
    _m2 = get_memory_bytes()
    print('time consumption: {}'.format(get_duration_str(time.time() - _t)))
    print('memory usage: {}({})'.format(get_size_str(_m2), get_size_diff_str(_m2 - _m)))
    print()
    return ret


# Copied from: https://www.kaggle.com/marknagelberg/rmsle-function
# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5


# ----- load data -----
# kaggle = True
# TRAIN_PATH = '../input/train.tsv' if kaggle else '../../data/raw/train.tsv'
# TEST_PATH = '../input/test.tsv' if kaggle else '../../data/raw/test.tsv'
TRAIN_PATH = 'E:/kaggle/train1.tsv'
TEST_PATH = 'E:/kaggle/test1.tsv'
# TRAIN_PATH = '/home/xuqingyao/kaggle/train.tsv'
# TEST_PATH = '/home/xuqingyao/kaggle/test.tsv'

print('reading data...')
df_train = pd.read_table(TRAIN_PATH)
df_test = pd.read_table(TEST_PATH)
print('memory usage: ', get_memory_str())


# ----- prepare variables  -----
# -- set constant --
TRAIN_SIZE = df_train.shape[0]
TEST_SIZE = df_test.shape[0]
TOTAL_SIZE = TRAIN_SIZE + TEST_SIZE

# -- split id and target --
print('spliting and merging data...')
id_train = df_train['train_id']
id_test = df_test['test_id']
y_train = df_train['price']
y_log_train = np.log1p(y_train)

# -- concatenate dataframe --
df_train.drop(['train_id', 'price'], axis=1, inplace=True)
df_test.drop(['test_id'], axis=1, inplace=True)
df_all = pd.concat([df_train, df_test], axis=0)
del df_train
del df_test
gc.collect()
print('memory usage: ', get_memory_str())
print('data shape: ', df_all.shape)


# ----- fill missing values -----
def fill_missing(df):
    df['item_description'].fillna(value='missing', inplace=True)
    df['brand_name'].fillna(value='missing', inplace=True)
    df['category_name'].fillna(value='missing', inplace=True)


print('filling missing data...')
run_func(fill_missing, df=df_all)


# ----- get dummies of non-text features ---
# -- make combination --
def combine_ship_cond(row):
    return "{}_{}".format(row['shipping'], row['item_condition_id'])

def make_comb(df):
    df['ship_cond_comb'] = df.apply(combine_ship_cond, axis=1)


print('getting combination of shipping and item_condition_id...')
run_func(make_comb, df=df_all)  # use _ to store redundant return
print('data shape: ', df_all.shape)

# -- get dummies --
cols = ['shipping', 'item_condition_id', 'ship_cond_comb']
print('getting dummies of shipping, item_condition_id and their combination...')
df_all = run_func(pd.get_dummies, data=df_all, columns=cols, prefix=cols)


# ----- replace words -----
# -- replace specific '/' in category_name --
cat_map = {
    'Electronics/Computers & Tablets/iPad/Tablet/eBook Readers':
        'Electronics/Computers & Tablets/iPad or Tablet or eBook Readers',
    'Electronics/Computers & Tablets/iPad/Tablet/eBook Access':
        'Electronics/Computers & Tablets/iPad or Tablet or eBook Access',
    'Sports & Outdoors/Exercise/Dance/Ballet':
        'Sports & Outdoors/Exercise/Dance or Ballet',
    'Sports & Outdoors/Outdoors/Indoor/Outdoor Games':
        'Sports & Outdoors/Outdoors/Indoor or Outdoor Games',
    'Men/Coats & Jackets/Flight/Bomber':
        'Men/Coats & Jackets/Flight or Bomber',
    'Men/Coats & Jackets/Varsity/Baseball':
        'Men/Coats & Jackets/Varsity or Baseball',
    'Handmade/Housewares/Entertaining/Serving':
        'Handmade/Housewares/Entertaining or Serving'
}

def replace_cat(df, map=cat_map):
    df['category_name'] = df['category_name'].apply(lambda x: map[x] if x in map.keys() else x)


print('replacing category_name...')
run_func(replace_cat, df=df_all)
succ = ((df_all['category_name'] == 'Electronics/Computers & Tablets/iPad/Tablet/eBook Readers').sum() == 0)
print('replacement success: ', succ)


# ----- split category into levels -----
def get_levels_helper(cat):
    levels = cat.split('/')
    num_levels = len(levels)
    if levels == [""]:
        levels = ['missing']
    if len(levels) == 1:
        levels.append(levels[0])
    if len(levels) == 2:
        levels.append(levels[1])
    return num_levels, levels[0], levels[1], levels[2]


def get_levels(df):
    df['num_cat_levles'], df['cat_level_0'], df['cat_level_1'], df['cat_level_2'] = zip(*df['category_name'].apply(get_levels_helper))
    return df


# -- get level 0,1 and 2 --
print('splitting category into levels...')
df_all = run_func(get_levels, df=df_all)
print("Number of unique level 0 category: ", df_all['cat_level_0'].nunique())
print("Number of unique level 1 category: ", df_all['cat_level_1'].nunique())
print("Number of unique level 2 category: ", df_all['cat_level_2'].nunique())


# ----- cutting minority in brand and category -----
BRAND_CUTOFF = 50
CAT_1_CUTOFF = 100  # modify to 300?
CAT_2_CUTOFF = 5


# cut off by threshold number
def get_reserved_list(df, col, cutoff):
    counts = df[col].value_counts()
    return list(counts[counts>=cutoff].index)


def get_reserved_brand_list(df, cutoff=BRAND_CUTOFF):
    return get_reserved_list(df, col='brand_name', cutoff=cutoff)


def get_reserved_cat1_list(df, cutoff=CAT_1_CUTOFF):
    return get_reserved_list(df, col='cat_level_1', cutoff=cutoff)
    
    
def get_reserved_cat2_list(df, cutoff=CAT_2_CUTOFF):
    return get_reserved_list(df, col='cat_level_2', cutoff=cutoff)
    
    
def cut_brand_cat(df, brand_cutoff=BRAND_CUTOFF, cat1_cutoff=CAT_1_CUTOFF, cat2_cutoff=CAT_2_CUTOFF):
    brand_list = get_reserved_brand_list(df, brand_cutoff)
    cat_1_list = get_reserved_cat1_list(df, cat1_cutoff)
    cat_2_list = get_reserved_cat2_list(df, cat2_cutoff)
    df['brand_name'] = df['brand_name'].apply(lambda x: x if x in brand_list else "other")
    df['cat_level_1'] = df['cat_level_1'].apply(lambda x: x if x in cat_1_list else "other")
    df['cat_level_2'] = df['cat_level_2'].apply(lambda x: x if x in cat_2_list else "other")
    del brand_list, cat_1_list, cat_2_list
    gc.collect()
    return df
    
    
df_all = run_func(cut_brand_cat, df=df_all)
print("Number of unique brand left: ", df_all['brand_name'].nunique())
print("Number of unique level 1 category left: ", df_all['cat_level_1'].nunique())
print("Number of unique level 2 category left: ", df_all['cat_level_2'].nunique())


# ----- get dummies of text feature -----
# only get dummies from brand so far
print('getting dummies of brand_name...')
brand_dummies = run_func(pd.get_dummies, data=df_all['brand_name'], sparse=True)
df_all.drop('brand_name', axis=1, inplace=True)
gc.collect()
print(df_all.shape)
print(list(df_all.columns))
print('size of brand_dummies:', get_size_str(brand_dummies.memory_usage(deep=True).sum()))


# -- get dummies of the levels --
print('getting dummies of level 0, 1 and 2 categories...')
cat_0_dummies = run_func(pd.get_dummies, data=df_all['cat_level_0'], sparse=True)
cat_1_dummies = run_func(pd.get_dummies, data=df_all['cat_level_1'], sparse=True)
cat_2_dummies = run_func(pd.get_dummies, data=df_all['cat_level_2'], sparse=True)

# -- clean memory --
print("df_all size:", get_size_str(df_all.memory_usage(deep=True).sum()))
cols_to_drop = ['category_name', 'cat_level_0', 'cat_level_1', 'cat_level_2']
print('dropping category_name, cat_level_0, cat_level_1 and cat_level_2...')
df_all.drop(cols_to_drop, axis=1, inplace=True)
gc.collect()
print("df_all size:", get_size_str(df_all.memory_usage(deep=True).sum()))




# ----- get Bag-of-Words/Tf-Idf features for text -----
# -- Bag-of-Words for name --
# parameters for CountVectorizer
NAME_MAX_FEAT = 50000  # min_df
NAME_NGRAM_RANGE = (1, 2)
NAME_DTYPE = np.int8  # to save memory


def get_name_bow(df, **params):
    bow_transformer = CountVectorizer(**params)
    X = bow_transformer.fit_transform(df["name"])
    del bow_transformer
    gc.collect()
    return X


print('getting Bag-of-Words representation of name...')
name_bow= run_func(get_name_bow, df=df_all, max_features=NAME_MAX_FEAT, ngram_range=NAME_NGRAM_RANGE, dtype=NAME_DTYPE)
print("name_bow dimension:", name_bow.shape[1])


# # -- Bag-of-Words for description --
# # parameters for TfidfVectorizer
# DESC_BOW_MAX_FEAT = 10000  # min_df
# DESC_BOW_NGRAM_RANGE = (1, 2)
# DESC_BOW_DTYPE = np.int16  # to save memory


# def get_desc_bow(df, **params):
#     bow_transformer = CountVectorizer(**params)
#     X = bow_transformer.fit_transform(df["item_description"])
#     del bow_transformer
#     gc.collect()
#     return X


# print('getting Bag-of-Words representation of description...')
# desc_bow= run_func(get_desc_bow, df=df_all, max_features=DESC_BOW_MAX_FEAT, ngram_range=DESC_BOW_NGRAM_RANGE, dtype=DESC_BOW_DTYPE)
# print("desc_bow dimension:", desc_bow.shape[1])


# -- Tf-Idf for description --
# parameters for TfidfVectorizer
DESC_TFIDF_MAX_FEAT = 100000  # max_features
DESC_TFIDF_NGRAM_RANGE = (1, 3)  # ngram_range
DESC_NORM = 'l2'  # norm
DESC_TFIDF_DTYPE = np.float32  # to save memory

import gensim
from gensim.models.word2vec import Word2Vec
# from gensim.models.doc2vec import Doc2Vec,LabeledSentence
# LabeledSentence = gensim.models.doc2vec.LabeledSentence #doc2vec
# from nltk.stem import WordNetLemmatizer
# wordnet_lemmatizer = WordNetLemmatizer()
from nltk.tokenize import word_tokenize
import re

def text_preprocessing(df):
    no_point_data = [re.sub('[^a-zA-Z]', ' ', each_text) for each_text in df]   #drop the punctuations
    processed_text = [each_text.lower() for each_text in no_point_data]     #change to the lowercases
    df_words = [word_tokenize(x) for x in processed_text]
    #reduct_df = [wordnet_lemmatizer.lemmatize(each_word) for each_word in df_words]
    return df_words

df_words = run_func(text_preprocessing, df=df_all['item_description'])

#   parameters of word2vec
num_features = 128
min_word_count = 10
context = 5
downsampling = 1e-3

model = Word2Vec(df_words, size=num_features, window=context, min_count=min_word_count, sample=downsampling)
index = set(model.wv.index2word)

def feature_to_vector(words, model, num_features):
    feature_vec = np.zeros(num_features, dtype='float32')
    n_words = 0
    index_word_set = set(model.wv.index2word)
    for word in words:
        if word in index_word_set:
            n_words = n_words + 1
            feature_vec = np.add(feature_vec, model[word])
    feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

def text_to_feature_vector(data, model, num_features):
    counter = 0
    text_features = np.zeros((len(data), num_features), dtype= 'float32')
    for each_data in data:
        text_features[counter] = feature_to_vector(each_data, model, num_features)
        counter += 1
    return text_features

print('getting word2vec features...')
w2v_vectors = run_func(text_to_feature_vector, data = df_words, model = model, num_features = num_features)


def get_desc_tfidf(df, **params):
    tfidf_transformer = TfidfVectorizer(**params)
    X = tfidf_transformer.fit_transform(df["item_description"])
    return X


print('getting Tf-Idf representation of item_description...')
desc_tfidf = run_func(get_desc_tfidf, df=df_all, max_features=DESC_TFIDF_MAX_FEAT, ngram_range=DESC_TFIDF_NGRAM_RANGE,
                        norm=DESC_NORM, dtype=DESC_TFIDF_DTYPE)
cols_to_drop = ['name', 'item_description']
df_all.drop(cols_to_drop, axis=1, inplace=True)
gc.collect()


# ----- prepare data for models -----
# some data are in sparse matrix format, so we need to combine them using horizontal stack
def get_stack_data():
    X = scipy.sparse.hstack((df_all,
                             brand_dummies,
                             cat_0_dummies,
                             cat_1_dummies,
                             cat_2_dummies,
                             name_bow,
                             w2v_vectors,
                            #  desc_bow,
                             desc_tfidf)).tocsr()
    return X


print("stacking data together...")
X = run_func(get_stack_data)
print((df_all.shape,
      brand_dummies.shape,
      cat_0_dummies.shape,
      cat_1_dummies.shape,
      cat_2_dummies.shape,
      name_bow.shape,
      w2v_vectors,
    #   desc_bow.shape,
      desc_tfidf.shape))
gc.collect()
print('final data shape:', X.shape)
VALID_RATIO = 0.05
y_sub_valid = y_train[:round(VALID_RATIO*TRAIN_SIZE)]
y_sub_train = y_train[round(VALID_RATIO*TRAIN_SIZE):TRAIN_SIZE]
y_log_sub_valid = y_log_train[:round(VALID_RATIO*TRAIN_SIZE)]
y_log_sub_train = y_log_train[round(VALID_RATIO*TRAIN_SIZE):TRAIN_SIZE]
X_valid = X[:round(VALID_RATIO*TRAIN_SIZE)]
X_train = X[round(VALID_RATIO*TRAIN_SIZE):TRAIN_SIZE]
X_test = X[TRAIN_SIZE:]
print("memory usage:", get_memory_str())
print("releasing memory...")
del X  # comment this line if you want to do cross validation
gc.collect()
print("memory usage:", get_memory_str())
print("training subset shape:", X_train.shape)
print("validation subset shape:", X_valid.shape)


# # ----- train model -----
# # -- Ridge Regression --
# # Note: Each weight is a double-precision float number occupying 8B memory, so a Ridge model
# # with 100,000 weights can take up nearly 1GB memory. therefore we need to delete the model and
# # free memory after using the model
# def get_ridge(X, y, **params):
#     model = Ridge(**params)
#     model.fit(X, y)
#     return model


# print('training Ridge Regression model...')
# num_ridge = 2
# for i in range(num_ridge):
#     print("model #{}".format(i))
#     # todo: to introduce randomness by random subset instead of 'random_state'
#     # the standard deviation of scores turned out to be 0, so seemingly 'random_state' didn't make the models different
#     ridge = run_func(get_ridge, X=X_train, y=y_log_sub_train, alpha=2+2*i, tol=0.05, solver = 'lsqr', fit_intercept=False)
#     pred_valid = ridge.predict(X_valid)
#     pred = ridge.predict(X_test).astype(np.float32)
#     pred_ridge = pred if i==0 else pred_ridge+pred
#     score = rmsle(np.expm1(np.asarray(pred_valid)), y_sub_valid)
#     ridge_eval = [score] if i==0 else ridge_eval+[score]
#     del ridge, pred_valid, pred, score
#     gc.collect()
# print("Ridge validation score: {:.6}(+/-{:.6})".format(np.mean(ridge_eval), np.std(ridge_eval)))
# pred_ridge /= num_ridge
# gc.collect()


# # -- Lasso Regression --
# # you may find that the memory usage of Lasso model is much lower than Ridge,
# # this is because Lasso emphasizes "sparsity", then many weights turn out to be zero indeed,
# # and won't occupy memory
# def get_lasso(X, y, **params):
#     model = Lasso(**params)
#     model.fit(X, y)
#     return model


# print('training Lasso Regression model...')
# num_lasso = 3
# for i in range(num_lasso):
#     print("model #{}".format(i))
#     # todo: to introduce randomness by random subset instead of 'random_state'
#     # the standard deviation of scores turned out to be 0, so seemingly 'random_state' didn't make the models different
#     lasso = run_func(get_lasso, X=X_train, y=y_log_sub_train, alpha=0.8+i*0.4, fit_intercept=False)
#     pred_valid = lasso.predict(X_valid)
#     pred = lasso.predict(X_test).astype(np.float32)
#     pred_lasso = pred if i==0 else pred_lasso+pred
#     score = rmsle(np.expm1(np.asarray(pred_valid)), y_sub_valid)
#     lasso_eval = [score] if i==0 else lasso_eval+[score]
#     del lasso, pred_valid, pred, score
#     gc.collect()
# print("Lasso validation score: {:.6}(+/-{:.6})".format(np.mean(lasso_eval), np.std(lasso_eval)))
# pred_lasso /= num_lasso
# gc.collect()


# -- LightGBM model --
def get_lgbm(**params):
    model = lgb.train(**params)
    return model


# prepare data
d_train = lgb.Dataset(X_train, label=y_log_sub_train, max_bin=8192)
d_valid = lgb.Dataset(X_valid, label=y_log_sub_valid, max_bin=8192)
watchlist = [d_train, d_valid]

# best accuracy, hopefully :)
# 'learning_rate':0.55 + 'num_leaves':105 --> 0.451953(stop at 5911 rounds)
LGB_PARAMS0 = {
    'learning_rate': 0.4,
    'application': 'regression',
    'max_depth': 3,
    'num_leaves': 105,
    # 'feature_fraction': 0.7,
    # 'colsample_bytree': 0.9,
    'lambda_l2': 0.001,
    'verbosity': -1,
    'metric': 'RMSE',
    #'nthread': 4 if kaggle else -1
}
print('training LightGBM #0 model...')
lgbm = run_func(get_lgbm, params=LGB_PARAMS0, train_set=d_train, num_boost_round=8200, valid_sets=watchlist,
                  early_stopping_rounds=50, verbose_eval=100)
pred_valid = lgbm.predict(X_valid)
pred_lgbm0 = lgbm.predict(X_test).astype(np.float32)
lgb_eval = rmsle(np.expm1(np.asarray(pred_valid)), y_sub_valid)
print("LightGBM #0 score: {:.6}".format(lgb_eval))
del lgbm
del pred_valid
gc.collect()


# # to avoid overfitting and make fast prediction, hopefully :)
# LGB_PARAMS1 = {
#     'learning_rate': 0.85,
#     'application': 'regression',
#     'max_depth': 3,
#     'num_leaves': 120,  # 80 seems a bit small
#     'feature_fraction': 0.8,
#     'colsample_bytree': 0.9,
#     'lambda_l2': 0.1,
#     'verbosity': -1,
#     'metric': 'RMSE',
#     'nthread': 4 if kaggle else -1
# }
# print('training LightGBM #1 model...')
# lgbm = run_func(get_lgbm, params=LGB_PARAMS1, train_set=d_train, num_boost_round=3000, valid_sets=watchlist,
#                   early_stopping_rounds=50, verbose_eval=100)
# pred_valid = lgbm.predict(X_valid)
# pred_lgbm1 = lgbm.predict(X_test).astype(np.float32)
# lgb_eval = rmsle(np.expm1(np.asarray(pred_valid)), y_sub_valid)
# print("LightGBM #1 score: {:.6}".format(lgb_eval))
# del lgbm
# del pred_valid
# gc.collect()


# # fast predicting
# LGB_PARAMS2 = {
#     'learning_rate': 1.1,
#     'application': 'regression',
#     'max_depth': 3,
#     'num_leaves': 100,
#     'feature_fraction': 0.7,
#     'colsample_bytree': 0.8,
#     'verbosity': -1,
#     'metric': 'RMSE',
#     'nthread': 4 if kaggle else -1
# }
# print('training LightGBM #2 model...')
# lgbm = run_func(get_lgbm, params=LGB_PARAMS2, train_set=d_train, num_boost_round=8000, valid_sets=watchlist,
#                   early_stopping_rounds=50, verbose_eval=100)
# pred_valid = lgbm.predict(X_valid)
# pred_lgbm2 = lgbm.predict(X_test).astype(np.float32)
# lgb_eval = rmsle(np.expm1(np.asarray(pred_valid)), y_sub_valid)
# print("LightGBM #2 score: {:.6}".format(lgb_eval))
# del lgbm
# del pred_valid
# gc.collect()


# ----- blend model -----
print('blending predictions...')

subm = pred_lgbm0
subm = pd.DataFrame({'test_id':id_test.values, 'price':np.expm1(subm)})
subm.to_csv("baseline5_7_cutoff_textplusplus_lgbm.csv", index=False)
