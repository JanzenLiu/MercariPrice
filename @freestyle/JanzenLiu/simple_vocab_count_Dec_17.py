from collections import Counter  # to count words
from multiprocessing import Pool, cpu_count  # for multiprocessing
import pandas as pd
import numpy as np
import nltk
import os


TRAIN_DATA_PATH = '../../data/raw/train.tsv'
TEST_DATA_PATH = '../../data/raw/test.tsv'

df_train = pd.read_table(TRAIN_DATA_PATH)
df_test = pd.read_table(TEST_DATA_PATH)


# docs is a list of documents, e.g. df_train['name']
def get_vocab_helper(docs, verbose=True):
    # DO NOT use for loop to count words over each doc, it's EXTREMELY SLOWLY, this way is INFINITY TIMES FASTER
    # i.e. join all docs first and then tokenize and count
    doc = " ".join([d for d in docs])
    counter = Counter(nltk.word_tokenize(doc))
    if verbose:
        print("finish getting vocabulary from {} documents".format(len(docs)))
    return counter


# docs is a list of documents, e.g. df_train['name']
def get_doc_vocab_helper(doc):
    counter = Counter(nltk.word_tokenize(str(doc)))
    return counter


def get_vocab_helper_by_multiprocessing(docs):
    workers = cpu_count()
    pool = Pool(processes=workers)
    result = pool.map(get_vocab_helper, [d for d in np.array_split(docs, workers)])
    pool.close()
    return result


def counter_to_df(counter, precision=6):
    num_total = sum([value for value in counter.values()])
    df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
    df = df.sort_values(by=0, axis=0, ascending=False).reset_index(drop=True)  # sort row by value(frequency)
    df = df.rename(columns={'index': 'item', 0: 'count'})  # rename columns
    df['rank'] = df.index.values
    df['percentage'] = df['count'].apply(lambda x: round(x/num_total, precision))
    return df


def get_vocab_by_multiprocessing(docs):
    vocabs = get_vocab_helper_by_multiprocessing(docs)
    ret_vocab = Counter()
    for vocab in vocabs:
        ret_vocab += vocab
    ret_vocab = counter_to_df(ret_vocab)
    return ret_vocab


# ----- for debugging use -----
# debug_text = [
#     "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's
# standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a
# type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining
# essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum
# passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.",
#     'Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin
# literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College
# in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going
# through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from
# sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in
# 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem
# Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.',
#     "The standard chunk of Lorem Ipsum used since the 1500s is reproduced below for those interested. Sections
# 1.10.32 and 1.10.33 from \"de Finibus Bonorum et Malorum\" by Cicero are also reproduced in their exact original
# form, accompanied by English versions from the 1914 translation by H. Rackham.",
#     "It is a long established fact that a reader will be distracted by the readable content of a page when looking at
# its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed
# to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and web
# page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web
# sites still in their infancy. Various versions have evolved over the years, sometimes by accident, sometimes on
# purpose (injected humour and the like).",
#     "There are many variations of passages of Lorem Ipsum available, but the majority have suffered alteration in
# some form, by injected humour, or randomised words which don't look even slightly believable. If you are going to use
# a passage of Lorem Ipsum, you need to be sure there isn't anything embarrassing hidden in the middle of text. All the
# Lorem Ipsum generators on the Internet tend to repeat predefined chunks as necessary, making this the first true
# generator on the Internet. It uses a dictionary of over 200 Latin words, combined with a handful of model sentence
# structures, to generate Lorem Ipsum which looks reasonable. The generated Lorem Ipsum is therefore always free from
# repetition, injected humour, or non-characteristic words etc."
# ] * 100
# counter_debug = get_vocab_by_multiprocessing(debug_text)


# ----- preparation for file saving -----
vocab_folder = 'vocab/Dec_17/'
if not os.path.exists(vocab_folder):
    os.makedirs(vocab_folder)


# ----- checking for NaN values -----
df_train.fillna('', inplace=True)
df_test.fillna('', inplace=True)
print("training set 'name' cleaned:", df_train['name'].isnull().sum() == 0)
print("training set 'item_description' cleaned:", df_train['item_description'].isnull().sum() == 0)
print("testing set 'name' cleaned:", df_test['name'].isnull().sum() == 0)
print("testing set 'item_description' cleaned:", df_test['item_description'].isnull().sum() == 0)


# ----- getting vocabulary/word counts for 'name' -----
print("counting words for 'name' in training set")
counter_name_train = get_vocab_by_multiprocessing(df_train['name'])
counter_name_train.to_csv('vocab/Dec_17/name_train.tsv', columns=['rank', 'item', 'count', 'percentage'],
                          sep='\t', index=False)
print("counting words for 'name' in testing set")
counter_name_test = get_vocab_by_multiprocessing(df_test['name'])
counter_name_test.to_csv('vocab/Dec_17/name_test.tsv', columns=['rank', 'item', 'count', 'percentage'],
                         sep='\t', index=False)


# ----- getting vocabulary/word counts for 'item_description' -----
print("counting words for 'item_description' in training set")
counter_desc_train = get_vocab_by_multiprocessing(df_train['item_description'])
counter_desc_train.to_csv('vocab/Dec_17/desc_train.tsv', columns=['rank', 'item', 'count', 'percentage'],
                          sep='\t', index=False)
print("counting words for 'item_description' in testing set")
counter_desc_test = get_vocab_by_multiprocessing(df_test['item_description'])
counter_desc_test.to_csv('vocab/Dec_17/desc_test.tsv', columns=['rank', 'item', 'count', 'percentage'],
                         sep='\t', index=False)


# ----- getting vocabulary/word counts for 'category_name' -----
print("counting words for 'category_name' in training set")
counter_cat_train = get_vocab_by_multiprocessing(df_train['category_name'])
counter_cat_train.to_csv('vocab/Dec_17/cat_train.tsv', columns=['rank', 'item', 'count', 'percentage'],
                         sep='\t', index=False)
print("counting words for 'category_name' in testing set")
counter_cat_test = get_vocab_by_multiprocessing(df_test['category_name'])
counter_cat_test.to_csv('vocab/Dec_17/cat_test.tsv', columns=['rank', 'item', 'count', 'percentage'],
                        sep='\t', index=False)


# ----- getting vocabulary/word counts for 'brand_name' -----
print("counting words for 'brand_name' in training set")
counter_brand_train = get_vocab_by_multiprocessing(df_train['brand_name'])
counter_brand_train.to_csv('vocab/Dec_17/brand_train.tsv', columns=['rank', 'item', 'count', 'percentage'],
                           sep='\t', index=False)
print("counting words for 'brand_name' in testing set")
counter_brand_test = get_vocab_by_multiprocessing(df_test['brand_name'])
counter_brand_test.to_csv('vocab/Dec_17/brand_test.tsv', columns=['rank', 'item', 'count', 'percentage'],
                          sep='\t', index=False)
