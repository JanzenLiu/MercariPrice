import pandas as pd
import numpy as np
import pickle
import os


# ----- prepare data -----
df_train = pd.read_table('../../data/raw/train.tsv')
df_test = pd.read_table('../../data/raw/test.tsv')

df_train['target'] = np.log1p(df_train['price'])


# ----- fill missing -----
def fill_missing(df):
    df['item_description'].fillna(value='missing', inplace=True)
    df['brand_name'].fillna(value='missing', inplace=True)
    df['category_name'].fillna(value='missing', inplace=True)


fill_missing(df_train)
fill_missing(df_test)


# ----- replace some category manually -----
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


replace_cat(df_train)
replace_cat(df_test)


# ----- split category_name into levels -----
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
    df['num_cat_levles'], df['cat_level_0'], df['cat_level_1'], df['cat_level_2'] =  \
        zip(*df['category_name'].apply(get_levels_helper))
    return df


df_train = get_levels(df_train)
df_test = get_levels(df_test)
print("Number of unique level 0 category in train: ", df_train['cat_level_0'].nunique())
print("Number of unique level 1 category in train: ", df_train['cat_level_1'].nunique())
print("Number of unique level 2 category in train: ", df_train['cat_level_2'].nunique())
print("Number of unique category in train: ", df_train['category_name'].nunique())


# ----- cutoff rare brand and sub-category -----
BRAND_CUTOFF = 150
CAT_1_CUTOFF = 300  # modify to 300?
CAT_2_CUTOFF = 10


def get_reserved_list(df, col_, cutoff):
    counts = df[col_].value_counts()
    return list(counts[counts >= cutoff].index)


def get_reserved_brand_list(df, cutoff=BRAND_CUTOFF):
    return get_reserved_list(df, col_='brand_name', cutoff=cutoff)


def get_reserved_cat1_list(df, cutoff=CAT_1_CUTOFF):
    return get_reserved_list(df, col_='cat_level_1', cutoff=cutoff)
    
    
def get_reserved_cat2_list(df, cutoff=CAT_2_CUTOFF):
    return get_reserved_list(df, col_='cat_level_2', cutoff=cutoff)


brand_list = get_reserved_brand_list(df_train)
cat_1_list = get_reserved_cat1_list(df_train)
cat_2_list = get_reserved_cat2_list(df_train)

df_train['brand_name'] = df_train['brand_name'].apply(lambda x: x if x in brand_list else "other")
df_train['cat_level_1'] = df_train['cat_level_1'].apply(lambda x: x if x in cat_1_list else "other")
df_train['cat_level_2'] = df_train['cat_level_2'].apply(lambda x: x if x in cat_2_list else "other")
df_test['brand_name'] = df_test['brand_name'].apply(lambda x: x if x in brand_list else "other")
df_test['cat_level_1'] = df_test['cat_level_1'].apply(lambda x: x if x in cat_1_list else "other")
df_test['cat_level_2'] = df_test['cat_level_2'].apply(lambda x: x if x in cat_2_list else "other")


# ----- grouping -----
group_by = ['brand_name', 'cat_level_0', 'cat_level_1', 'cat_level_2', 'item_condition_id', 'shipping']
group_train = df_train.groupby(group_by)

group_means = dict(group_train['target'].mean())
group_std = dict(group_train['target'].std())

# --- mean and standard deviation for whole training set ---
default_m = df_train['target'].mean()
default_s = df_train['target'].std()


# --- auxiliary functions to get group mean/std ---
def get_group_info_helper(row_):  # to mute IDE warning
    # get key for group dictionaries
    tup_ = (  # to mute IDE warning
        row_['brand_name'],
        row_['cat_level_0'],
        row_['cat_level_1'],
        row_['cat_level_2'],
        row_['item_condition_id'],
        row_['shipping']
    )
    # return mean/std calculated from the whole training set for invalid key or NaN std
    try:
        m = group_means[tup_]
        s = group_std[tup_]
    except KeyError:
        m = default_m
        s = default_s
    if np.isnan(s):
        m = default_m
        s = default_s
    return m, s


def get_group_info(df):
    df['group_mean'], df['group_std'] = zip(*df.apply(get_group_info_helper, axis=1))
    return df


df_train = get_group_info(df_train)
df_test = get_group_info(df_test)

# --- re-calculate target for training ---
y_train = (df_train['target']-df_train['group_mean'])/df_train['group_std']

# --- store grouping information ---
feat_folder = 'feat/Dec_12/'
if not os.path.exists(feat_folder):
    os.makedirs(feat_folder)
with open('feat/Dec_12/recalculed_target.p', 'wb') as f:
    pickle.dump(y_train, f)
with open('feat/Dec_12/train_group_mean.p', 'wb') as f:
    pickle.dump(df_train['group_mean'], f)
with open('feat/Dec_12/train_group_std.p', 'wb') as f:
    pickle.dump(df_train['group_std'], f)
with open('feat/Dec_12/test_group_mean.p', 'wb') as f:
    pickle.dump(df_test['group_mean'], f)
with open('feat/Dec_12/test_group_std.p', 'wb') as f:
    pickle.dump(df_test['group_std'], f)


# --- view some abnormal data ---
row = df_train.iloc[76454]
tup = (row['brand_name'], row['cat_level_0'], row['cat_level_1'], row['cat_level_2'],
       row['item_condition_id'], row['shipping'])
print(type)
print(group_train.get_group(tup))

row = df_train.iloc[16922]
tup = (row['brand_name'], row['cat_level_0'], row['cat_level_1'], row['cat_level_2'],
       row['item_condition_id'], row['shipping'])
print(group_train.get_group(tup))

# TODO: deal with 0 standard deviation
# TODO: deal with infinity (price-mean)/std
