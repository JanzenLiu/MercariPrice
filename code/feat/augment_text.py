import pandas as pd


def augment_text(df, col="item_description", add_cols=None, joiner=" "):
    assert isinstance(df, pd.DataFrame)
    assert isinstance(joiner, str)

    if add_cols is None or (not isinstance(add_cols, list)) or \
            len(add_cols) == 0 or (not isinstance(add_cols[0], str)):
        add_cols = ['name', 'category_name', 'brand_name']

    for add_col in add_cols:
        df[col] = df[add_col] + " " + df[col]  # this way will be faster than simply call pd.DataFrame.apply()

    return df
