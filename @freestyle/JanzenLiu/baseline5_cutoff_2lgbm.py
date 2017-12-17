# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math  
import scipy
import psutil  # to get process information
import time
import gc  # to collect garbage in the memory
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge, Lasso
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


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
# convert bytes to 'xxMB'/'yyGB'/etc.
def get_size_str(num_bytes, units=None):
    assert num_bytes >= 0, "Negative size is not allowed"
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
def rmsle(y, y_pred_):
    assert len(y) == len(y_pred_)
    terms_to_sum = [(math.log(y_pred_[i_] + 1) - math.log(y[i_] + 1)) ** 2.0 for i_, pred_ in enumerate(y_pred_)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5


# ----- load data -----
kaggle = True
TRAIN_PATH = '../input/train.tsv' if kaggle else '../../data/raw/train.tsv'
TEST_PATH = '../input/test.tsv' if kaggle else '../../data/raw/test.tsv'

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


def replace_cat(df, maps=cat_map):
    df['category_name'] = df['category_name'].apply(lambda x: maps[x] if x in maps.keys() else x)


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
    df['num_cat_levles'], df['cat_level_0'], df['cat_level_1'], df['cat_level_2'] = \
        zip(*df['category_name'].apply(get_levels_helper))
    return df


# -- get level 0,1 and 2 --
print('splitting category into levels...')
df_all = run_func(get_levels, df=df_all)
print("Number of unique level 0 category: ", df_all['cat_level_0'].nunique())
print("Number of unique level 1 category: ", df_all['cat_level_1'].nunique())
print("Number of unique level 2 category: ", df_all['cat_level_2'].nunique())


# ----- cutting minority in brand and category -----
BRAND_CUTOFF = 150
CAT_1_CUTOFF = 300  # modify to 300?
CAT_2_CUTOFF = 8


# cut off by threshold number
def get_reserved_list(df, col_, cutoff):
    counts = df[col_].value_counts()
    return list(counts[counts >= cutoff].index)


def get_reserved_brand_list(df, cutoff=BRAND_CUTOFF):
    return get_reserved_list(df, col_='brand_name', cutoff=cutoff)


def get_reserved_cat1_list(df, cutoff=CAT_1_CUTOFF):
    return get_reserved_list(df, col_='cat_level_1', cutoff=cutoff)
    
    
def get_reserved_cat2_list(df, cutoff=CAT_2_CUTOFF):
    return get_reserved_list(df, col_='cat_level_2', cutoff=cutoff)
    
    
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
NAME_MIN_DF = 5  # min_df
NAME_DTYPE = np.int8  # to save memory


def get_name_bow(df, **params):
    bow_transformer = CountVectorizer(**params)
    vec = bow_transformer.fit_transform(df["name"])  # to mute IDE warning
    return vec


print('getting Bag-of-Words representation of name...')
name_bow = run_func(get_name_bow, df=df_all, min_df=NAME_MIN_DF, dtype=NAME_DTYPE)
print("name_bow dimension:", name_bow.shape[1])


# -- Tf-Idf for description --
# parameters for TfidfVectorizer
DESC_MAX_FEAT = 60000  # max_features
DESC_NGRAM_RANGE = (1, 3)  # ngram_range
DESC_NORM = 'l2'  # norm
DESC_DTYPE = np.float32  # to save memory


def get_desc_tfidf(df, **params):
    tfidf_transformer = TfidfVectorizer(**params)
    vec = tfidf_transformer.fit_transform(df["item_description"])
    return vec


print('getting Tf-Idf representation of item_description...')
desc_tfidf = run_func(get_desc_tfidf, df=df_all, max_features=DESC_MAX_FEAT, ngram_range=DESC_NGRAM_RANGE,
                      norm=DESC_NORM, dtype=DESC_DTYPE)
cols_to_drop = ['name', 'item_description']
df_all.drop(cols_to_drop, axis=1, inplace=True)
gc.collect()


# ----- prepare data for models -----
# some data are in sparse matrix format, so we need to combine them using horizontal stack
def get_stack_data():
    mat = scipy.sparse.hstack(
        (df_all,
         brand_dummies,
         cat_0_dummies,
         cat_1_dummies,
         cat_2_dummies,
         name_bow,
         desc_tfidf)
    ).tocsr()
    return mat


print("stacking data together...")
X_stack = run_func(get_stack_data)
print((df_all.shape, 
      brand_dummies.shape, 
      cat_0_dummies.shape, 
      cat_1_dummies.shape, 
      cat_2_dummies.shape, 
      name_bow.shape, 
      desc_tfidf.shape))
gc.collect()
print('final data shape:', X_stack.shape)
VALID_RATIO = 0.05
y_sub_valid = y_train[:round(VALID_RATIO*TRAIN_SIZE)]
y_sub_train = y_train[round(VALID_RATIO*TRAIN_SIZE):TRAIN_SIZE]
y_log_sub_valid = y_log_train[:round(VALID_RATIO*TRAIN_SIZE)]
y_log_sub_train = y_log_train[round(VALID_RATIO*TRAIN_SIZE):TRAIN_SIZE]
X_valid = X_stack[:round(VALID_RATIO * TRAIN_SIZE)]
X_train = X_stack[round(VALID_RATIO * TRAIN_SIZE):TRAIN_SIZE]
X_test = X_stack[TRAIN_SIZE:]
print("memory usage:", get_memory_str())
print("releasing memory...")
del X_stack  # comment this line if you want to do cross validation
gc.collect()
print("memory usage:", get_memory_str())
print("training subset shape:", X_train.shape)
print("validation subset shape:", X_valid.shape)


# ----- train model -----
# -- Ridge Regression --
# Note: Each weight is a double-precision float number occupying 8B memory, so a Ridge model 
# with 100,000 weights can take up nearly 1GB memory. therefore we need to delete the model and 
# free memory after using the model
def get_ridge(X, y, **params):
    model = Ridge(**params)
    model.fit(X, y)
    return model


print('training Ridge Regression model...')
num_ridge = 2
pred_ridge = 0  # to mute IDE warning
ridge_eval = []
for i in range(num_ridge):
    print("model #{}".format(i))
    # TODO: to introduce randomness by random subset instead of 'random_state'
    # the standard deviation of scores turned out to be 0, so seemingly 'random_state' didn't make the models different
    ridge = run_func(get_ridge, X=X_train, y=y_log_sub_train, alpha=0.8**i, solver='lsqr', fit_intercept=False)
    pred_valid = ridge.predict(X_valid)
    pred = ridge.predict(X_test).astype(np.float32)
    pred_ridge = pred if i == 0 else pred_ridge + pred
    score = rmsle(np.expm1(np.asarray(pred_valid)), y_sub_valid)
    ridge_eval += [score]
    del ridge, pred_valid, pred, score
    gc.collect()
print("Ridge validation score: {:.6}(+/-{:.6})".format(np.mean(ridge_eval), np.std(ridge_eval)))
pred_ridge /= num_ridge
gc.collect()


# -- Lasso Regression --
# you may find that the memory usage of Lasso model is much lower than Ridge,
# this is because Lasso emphasizes "sparsity", then many weights turn out to be zero indeed,
# and won't occupy memory
def get_lasso(X, y, **params):
    model = Lasso(**params)
    model.fit(X, y)
    return model


print('training Lasso Regression model...')
num_lasso = 3
pred_lasso = 0  # to mute IDE warning
lasso_eval = []
for i in range(num_lasso):
    print("model #{}".format(i))
    # todo: to introduce randomness by random subset instead of 'alpha'
    lasso = run_func(get_lasso, X=X_train, y=y_log_sub_train, alpha=1.2**i, fit_intercept=False)
    pred_valid = lasso.predict(X_valid)
    pred = lasso.predict(X_test).astype(np.float32)
    pred_lasso = pred if i == 0 else pred_lasso + pred
    score = rmsle(np.expm1(np.asarray(pred_valid)), y_sub_valid)
    lasso_eval += [score]
    del lasso, pred_valid, pred, score
    gc.collect()
print("Lasso validation score: {:.6}(+/-{:.6})".format(np.mean(lasso_eval), np.std(lasso_eval)))
pred_lasso /= num_lasso
gc.collect()


# -- LightGBM model --
def get_lgbm(**params):
    model = lgb.train(**params)
    return model


# prepare data
d_train = lgb.Dataset(X_train, label=y_log_sub_train, max_bin=8192)
d_valid = lgb.Dataset(X_valid, label=y_log_sub_valid, max_bin=8192)
watchlist = [d_train, d_valid]

# best accuracy, hopefully :)
# 'learning_rate':0.75 + 'num_leaves':
LGB_PARAMS0 = {
    'learning_rate': 0.55,
    'application': 'regression',
    'max_depth': 3,
    'num_leaves': 105,
    # 'feature_fraction': 0.7,
    # 'colsample_bytree': 0.9,
    'lambda_l2': 0.001,
    'verbosity': -1,
    'metric': 'RMSE',
    'nthread': 4 if kaggle else -1
}
print('training LightGBM #0 model...')
lgbm = run_func(get_lgbm, params=LGB_PARAMS0, train_set=d_train, num_boost_round=8000, valid_sets=watchlist, 
                early_stopping_rounds=50, verbose_eval=100)
pred_valid = lgbm.predict(X_valid)
pred_lgbm0 = lgbm.predict(X_test).astype(np.float32)
lgb_eval = rmsle(np.expm1(np.asarray(pred_valid)), y_sub_valid)
print("LightGBM #0 score: {:.6}".format(lgb_eval))
del lgbm
del pred_valid
gc.collect()


# to avoid overfitting, hopefully :)
LGB_PARAMS1 = {
    'learning_rate': 0.85,  # to speed up
    'application': 'regression',
    'max_depth': 3,
    'num_leaves': 90,  # 80 seems a bit small
    'lambda_l2': 0.2,  # to reduce overfitting
    'verbosity': -1,
    'metric': 'RMSE',
    'nthread': 4 if kaggle else -1
}
print('training LightGBM #1 model...')
lgbm = run_func(get_lgbm, params=LGB_PARAMS1, train_set=d_train, num_boost_round=8000, valid_sets=watchlist, 
                early_stopping_rounds=50, verbose_eval=100)
pred_valid = lgbm.predict(X_valid)
pred_lgbm1 = lgbm.predict(X_test).astype(np.float32)
lgb_eval = rmsle(np.expm1(np.asarray(pred_valid)), y_sub_valid)
print("LightGBM #1 score: {:.6}".format(lgb_eval))
del lgbm
del pred_valid
gc.collect()


# ----- blend model -----
print('blending predictions...')
weights = {'ridge': 0.08, 'lasso': 0.02, 'lgbm0': 0.6, 'lgbm1': 0.3}
subm = weights['ridge']*pred_ridge + weights['lasso']*pred_lasso + \
       weights['lgbm0']*pred_lgbm0 + weights['lgbm1']*pred_lgbm1
subm = pd.DataFrame({'test_id': id_test.values, 'price': np.expm1(subm)})
subm.to_csv("baseline5_2_cutoff_2lgbm.csv", index=False)
