import pandas as pd


def fill_description(df):
    assert isinstance(df, pd.DataFrame)

    df["item_description"].fillna("no description yet", inplace=True)

    return df
