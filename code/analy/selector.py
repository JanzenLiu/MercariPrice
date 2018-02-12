from ..utils.perf_utils import *
from nltk.stem.porter import PorterStemmer


stemmer = PorterStemmer()


def df_selector(func):
    def func_wrapper(*args, **kw):
        try:
            return func(args, kw)
        except Exception as e:
            print(e)
            return None

    return func_wrapper


@df_selector
def df_with_brand(df, val):
    try:
        return df[df["brand_name"] == val]
    except Exception as e:
        print(e)
        return None


@df_selector
def df_with_category(df, val):
    return df[df["category_name"] == val]


@task("find samples with brand in name")
@df_selector
def df_with_brand_in_name(df, val, miss=True, strict=False, stem=False):
    if strict:
        idx = df[df["name"].map(lambda x: val.lower() in x.lower().split())].index
        if stem:
            df_copy = df.iloc[idx].copy()
            val = stemmer.stem(val.lower())
            ret = df_copy[df_copy["name"].map(lambda x: val in [stemmer.stem(w) for w in x.lower().split()])]
        else:
            ret = df.iloc[idx]
    else:
        ret = df[df["name"].map(lambda x: val.lower() in x.lower())]
    return ret[ret["brand_name"].isnull()] if miss else ret
