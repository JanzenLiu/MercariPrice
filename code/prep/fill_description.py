import pandas as pd
try:
    from ..utils.perf_utils import task
except SystemError as e:
    import sys
    sys.path.insert(0, "../utils/")
    from perf_utils import task


@task("fill description")
def fill_description(df):
    assert isinstance(df, pd.DataFrame)

    df["item_description"].fillna("no description yet", inplace=True)

    return df
