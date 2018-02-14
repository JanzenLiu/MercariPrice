import pandas as pd
import numpy as np


default_sep = "9qmi02BD"  # random hash string


def get_groups(df, cols, target="log_price", min_count=3):
    """
    Group the DataFrame with given columns

    :param df: pd.DataFrame
    :param cols: [string]
    :param target: string
    :param min_count: integer
    :return: pd.DataFrame
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(cols, list)
    assert isinstance(target, str)
    assert target in df.columns

    ret = df.groupby(cols)[target].apply(list).reset_index()
    if min_count <= 1:
        return ret
    else:
        idx = ret[target].apply(lambda x: len(x) >= min_count)
        return ret.loc[idx]


def get_groups_with_mask(df, cols, mask, target="log_price", min_count=3):
    """
    Group the DataFrame with masked columns, with the columns returned as well, whose values is pre-defined

    :param df: pd.DataFrame
    :param cols: [string]
    :param mask: [boolean]
    :param target: string
    :param min_count: integer
    :return:
    """
    assert len(cols) == len(mask)

    cols_true = []
    cols_false = []
    for i in range(len(mask)):
        k = cols[i]
        v = mask[i]
        if v:
            cols_true.append(k)
        else:
            cols_false.append(k)
    ret = get_groups(df, cols_true, target, min_count)

    for col in cols_false:
        ret[col] = "###IGNORE###"

    return ret[cols + [target]]  # adjust the order of columns


def _get_row_stats(lst):
    """
    Calculate element counts, standard deviation and mean of a given list of number

    :param lst: [float/double]
    :return: float, float, float
    """
    return len(lst), np.std(lst), np.mean(lst)


def extract_group_stats(df, col="log_price"):
    """
    Add columns standing for counts, standard deviation and mean calculated from a specified column to the DataFrame

    :param df: pd.DataFrame
    :param col: string
    :return: pd.DataFrame
    """
    df['count'], df['std'], df['mean'] = \
        zip(*df[col].apply(_get_row_stats))
    return df


bool_int_dict = {True: 1, False: 0}


def _bool_list_to_idx(*args):
    """
    Convert a sequqnce of boolean to a integer, whose bits each represent a True/False in the arguments

    :param args: [boolean]
    :return: integer
    """
    l = len(args)
    idx = 0
    for i in range(l):
        idx *= 2
        idx += bool_int_dict[args[i]]
    return idx


def get_all_masks(size, skip_all_false=True, skip_all_true=False):
    """
    Generate the list of all possible masks given the length of the mask required

    :param size: integer
    :param skip_all_false: boolean
    :param skip_all_true: boolean
    :return:
    """
    if size == 1:
        ret = [[False], [True]]
    else:
        masks = get_all_masks(size-1)
        ret = [[False] + mask for mask in masks] + [[True] + mask for mask in masks]

    if skip_all_false:
        ret = ret[1:]

    if skip_all_true:
        ret = ret[:-1]

    return ret


class GroupHelper:
    def __init__(self, df, cols, target="log_price", min_count=3, jointer=default_sep):
        assert isinstance(df, pd.DataFrame)
        assert isinstance(cols, list) and len(cols) > 0
        assert isinstance(target, str)
        assert df[cols].isnull().sum().sum() == 0

        self.cols = cols
        self.num_cols = len(cols)
        self.target = target
        self.min_count = min_count
        self.jointer = jointer
        self.raw_mean = np.mean(df[self.target])
        self.raw_std = np.std(df[self.target])
        self.masks = get_all_masks(self.num_cols, False, False)
        self.col_unique_vals = {}
        self.group_info = {}
        self.query_info = {}

        for col in cols:
            self.col_unique_vals[col] = df[~df[col].isnull()][col].unique().tolist()

        for mask in self.masks:
            group = get_groups_with_mask(df, self.cols, mask, self.target, self.min_count)
            group = extract_group_stats(group, self.target)
            df_tmp = group[["std", "mean"]]
            df_tmp["key"] = group[self.cols[0]]
            for i in range(1, self.num_cols):
                df_tmp["key"] += jointer + group[cols[i]]
            for i, row in df_tmp.iterrows():
                self.group_info[row["key"]] = (row["std"], row["mean"])

        self.group_keys = self.group_info.keys()

    def update_query(self, vals):
        assert isinstance(vals, list) and len(vals) == self.num_cols

        real_key = self.jointer.join(vals)
        best_std = self.raw_std
        best_mean = self.raw_mean

        for mask in self.masks:
            vals_copy = vals.copy()
            for i in range(len(mask)):
                m = mask[i]
                if not m:
                    vals_copy[i] = "###IGNORE###"
            key = self.jointer.join(vals)
            if key in self.group_keys:
                std, mean = self.group_info[key]
                if std < best_std:
                    best_std = std
                    best_mean = mean

        self.query_info[real_key] = (best_std, best_mean)

    def update_query_from_df(self, df):
        df_copy = df.copy()
        df_copy["###"] = 1
        group = df_copy.groupby(self.cols)["###"].count().reset_index()
        for i, row in group.iterrows():
            self.update_query(row[self.cols].tolist())

    def query_tuple(self, vals):
        key = self.jointer.join(vals)
        if key in self.query_info:
            return self.query_info[key]
        else:
            return self.raw_std, self.raw_mean

    def query_row(self, row):
        return self.query_tuple(row[self.cols])
