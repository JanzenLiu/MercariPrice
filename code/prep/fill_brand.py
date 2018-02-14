import pandas as pd
import numpy as np
try:
    from ..utils.perf_utils import task
except SystemError as e:
    import sys
    sys.path.insert(0, "../utils/")
    from perf_utils import task


def _get_multiword_dict(s_lst):
    d = {}
    for s in s_lst:
        w_list = s.split(" ")  # split string into word sequence/list
        if len(w_list) < 2:
            continue  # skip word with single word
        w = w_list[0].lower()
        if w in d.keys():
            d[w].append(s)
        else:
            d[w] = [s]
    return d


def _get_multiword_dict_simple(s_lst):
    d = {}
    for s in s_lst:
        w_list = s.split(" ")  # split string into word sequence/list
        w = w_list[0].lower()
        if w in d.keys():
            d[w].append(s)
        else:
            d[w] = [s]
    return d


def _get_row_brand_suggestion_with_text_simple(text, singleword_keys, singleword_dict,
                                               multiword_keys, multiword_dict, default_value=None):
    text = " {} ".format(str(text).lower())  # process name
    seq = text.split(" ")  # get word sequence/list

    # check multiple-word candidates first
    for word in seq:
        if word in multiword_keys:
            for s in multiword_dict[word]:
                s2 = " {} ".format(s.lower())
                if s2 in text:
                    return s

    # if not found, check single-word candidates
    for word in seq:
        if word in singleword_keys:
            return singleword_dict[word]

    # if not found, return with default column value
    if isinstance(default_value, str):
        return default_value
    else:
        return np.nan


def get_brand_with_col(row, col):
    ret = "Missing Brand"
    try:
        if isinstance(row[col], str):
            ret = row[col]
    except Exception as e:
        pass
    return ret


def get_brand_default(row):
    return get_brand_with_col(row, "category_name")


def prepare_fill_brand(lst):
    # sort candidates to single-word and multiple-word
    sl = []  # sl for single-word list
    ml = []  # ml for multiple-word list
    for v in lst:
        if len(v.split(" ")) > 1:
            ml.append(v)
        else:
            sl.append(v)

    md = _get_multiword_dict_simple(ml)  # md for multiple-word dictionary
    mk = md.keys()  # mk for multiple-word keys

    sd = {s.lower():s for s in sl}  # sd for single-word dictionary
    sk = sd.keys()  # sk for single-word keys

    return sk, sd, mk, md


@task("fill brand with name")
def fill_brand_with_name(df, lst, default_value=None):
    assert isinstance(df, pd.DataFrame)
    sk, sd, mk, md = prepare_fill_brand(lst)

    col = "brand_name"
    idx = df[df[col].isnull()].index
    df.loc[idx, "brand_name"] = df.loc[idx]["name"].map(
        lambda x: _get_row_brand_suggestion_with_text_simple(x, singleword_keys=sk, singleword_dict=sd,
                                                             multiword_keys=mk, multiword_dict=md,
                                                             default_value=default_value))
    return df


@task("fill brand with description")
def fill_brand_with_description(df, lst, default_value=None):
    assert isinstance(df, pd.DataFrame)
    sk, sd, mk, md = prepare_fill_brand(lst)

    col = "brand_name"
    idx = df[df[col].isnull()].index
    df.loc[idx, "brand_name"] = df.loc[idx]["item_description"].map(
        lambda x: _get_row_brand_suggestion_with_text_simple(x, singleword_keys=sk, singleword_dict=sd,
                                                             multiword_keys=mk, multiword_dict=md,
                                                             default_value=default_value))
    return df
