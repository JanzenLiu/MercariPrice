import pandas as pd


default_cat_map = {
    'Electronics/Computers & Tablets/iPad/Tablet/eBook Readers':
        'Electronics/Computers & Tablets/iPad or Tablet or eBook Readers',
    'Electronics/Computers & Tablets/iPad/Tablet/eBook Access':
        'Electronics/Computers & Tablets/iPad or Tablet or eBook Access',
    'Sports & Outdoors/Exercise/Dance/Ballet':
        'Sports & Outdoors/Exercise/Dance or Ballet',
    'Sports & Outdoors/Outdoors/Indoor/Outdoor Games':
        'Sports & Outdoors/Outdoors/Indoor or Outdoor Games',
    'Men/Coats & Jackets/Flight/Bomber':
        'Men/Coats & Jackets/Flight or Bomber',
    'Men/Coats & Jackets/Varsity/Baseball':
        'Men/Coats & Jackets/Varsity or Baseball',
    'Handmade/Housewares/Entertaining/Serving':
        'Handmade/Housewares/Entertaining or Serving'
}


def clean_category(df, cat_map=None):
    assert isinstance(df, pd.DataFrame)

    if cat_map is None or (not isinstance(cat_map, dict)):
        cat_map = default_cat_map

    df['category_name'] = df['category_name'].apply(lambda x: cat_map[x] if x in cat_map.keys() else x)
