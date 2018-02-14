import pandas as pd


def replace_brand_simple(df, dic):
    assert isinstance(df, pd.DataFrame)
    assert isinstance(dic, dict)

    col = "brand_name"
    keys = dic.keys
    idx = df[df[col].isin(keys)].index
    df.loc[idx, col] = df.loc[idx, col].map(lambda x: dic[x])
    return df
