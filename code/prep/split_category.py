def _get_row_category_levels(text, fill_str):
    try:
        lvls = text.split("/")

        l = len(lvls)
        if l > 3:
            lvls = lvls[:3]
        if l < 3:
            lvls += [fill_str] * (3-l)

        return lvls
    except Exception as e:
        return [fill_str] * 3


def split_category(df, fill_str="Category Missing"):
    df['cat_0'], df['cat_1'], df['cat_2'] = \
        zip(*df['category_name'].apply(lambda x: _get_row_category_levels(x, fill_str)))
