import pandas as pd
try:
    from ..utils.perf_utils import task
except SystemError as e:
    import sys
    sys.path.insert(0, "../utils/")
    from perf_utils import task


task("augment text")
def augment_text(df, col="item_description", add_cols=None, jointer=" "):
    assert isinstance(df, pd.DataFrame)
    assert isinstance(jointer, str)

    if add_cols is None or (not isinstance(add_cols, list)) or \
            len(add_cols) == 0 or (not isinstance(add_cols[0], str)):
        add_cols = ['name', 'category_name', 'brand_name']

    for add_col in add_cols:
        df[col] = df[add_col] + jointer + df[col]  # this way will be faster than simply call pd.DataFrame.apply()

    return df
