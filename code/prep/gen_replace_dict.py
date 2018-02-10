# coding: utf-8
import pandas as pd
from nltk.stem.porter import PorterStemmer
import json  # to dump python dict
from ..utils.perf_utils import *


# --- load manually checked replacement schemes for equivalent brand names ---
folder = './raw/'
fname = 'brand_dist_1_v1_janzen.csv'
df_d1_v1 = pd.read_csv(os.path.join(folder, fname))
fname = 'brand_dist_1_v1_glassy.csv'
df_d1_v2 = pd.read_csv(os.path.join(folder, fname), encoding='ISO-8859-1')  # glassy's can't be opened with utf-8


# --- get indices for different and same pairs ---
diff_idx = df_d1_v1[df_d1_v2['replacable'] != df_d1_v1['replacable']].index
same_idx = df_d1_v1[df_d1_v2['replacable'] == df_d1_v1['replacable']].index
print("num diff between two replace scheme:", len(diff_idx))


# --- get replaceable pairs agreed by both schemes ---
df_same = df_d1_v1.iloc[same_idx]
df_p1 = df_same[df_same['replaceable'] == 1]
dict_p1 = {row['brand_B']: row['brand_A'] for i, row in df_p1.iterrows()}  # generate dict
print("num tuples in v1 replace dict:", len(dict_p1))


# --- save part 1 replacement dictionary ---
folder = './checked/'
fname = 'brand_d1_p1.dict'
if not os.path.exists(folder):
    os.makedirs(folder)
with open(os.path.join(folder, fname), 'w') as f:
    json.dump(dict_p1, f, indent=2)


# --- view diff pairs ---
print(df_d1_v1.iloc[diff_idx])
print(df_d1_v2.iloc[diff_idx])


# --- load train data to inspect ---
folder = '../../data/raw/'
fname = 'train.tsv'
df_train = pd.read_table(os.path.join(folder, fname))
fname = 'test.tsv'
df_test = pd.read_table(os.path.join(folder, fname))


# --- helper function to find subset of DataFrame with certain brand ---
@task("find samples with given brand_name")
def df_with_brand(df, bname):
    return df[df['brand_name'] == bname]


# --- view brands ---
df_with_brand(df_train, 'Camilla')  # only one is left here because I did the inspection in Jupyter Notebook
df_with_brand(df_test, 'Camilla')  # only one is left here because I did the inspection in Jupyter Notebook


# --- confirmed replacable pairs ---
p2_idx = [68, 86, 87, 95, 106, 156, 267, 277, 366, 378]
special_brand_list = ["Athelete", "MATRIX", "Elements", "Curve"]  # mislabelled brands found here
df_p2 = df_d1_v1.iloc[p2_idx]
dict_p2 = {row['brand_B']: row['brand_A'] for _, row in df_p2.iterrows()}  # generate dict


# --- save part 2 replacement dictionary ---
folder = './checked/'
fname = 'brand_d1_p2.dict'
with open(os.path.join(folder, fname), 'w') as f:
    json.dump(dict_p2, f, indent=2)


# --- save mislabelled brands list ---
fname = 'brand_mislabeled_p1.lst'
with open(os.path.join(folder, fname), 'w') as f:
    for val in special_brand_list:
        f.write(val + '\n')


# --- load manually checked replacement schemes for special characters ---
folder = './raw/'
fname = 'char_v1.csv'
df_char_v1 = pd.read_csv(os.path.join(folder, fname))
df_char_v1['to_replace_with'] = df_char_v1['to_replace_with'].map(
    lambda x: x if len(x) == 1 else " {} ".format(x.strip())  # if the string used to replace is a word, add spaces
)
dict_char = {row['character']: row['to_replace_with'] for _, row in df_char_v1.iterrows()}  # generate dict


# --- save replacement dictionary ---
folder = './checked'
fname = 'char_v1.dict'
with open(os.path.join(folder, fname), 'w') as f:
    json.dump(dict_char, f, indent=2)


# --- load manually checked fillable brand_name(A-K) from name ---
folder = './raw/'
fname = 'brand_in_name_A_K.xlsx'
ws = pd.read_excel(os.path.join(folder, fname),
                   sheet_name="results", usecols=["index", "name", "replace"],
                   dtype={"index": np.int32, "name": str, "replace": np.int8})  # ws for worksheet
ws['name'] = ws['name'].map(lambda x: x.replace('.test.tsv', '').replace('@@slash@@', '/'))  # correct error and decode


# --- get replaceable, not replaceable and to-be-determined brands ---
bin_no_repl = {}  # to store brands; bin for "brand in name", repl for replaceable
bin_repl = {}
bin_empty = {}
bin_to_determine = {}
bin_no_repl["A_K"] = ws[ws["replace"] == 0]["name"].tolist()
bin_repl["A_K"] = ws[ws["replace"] == 1]["name"].tolist()
bin_empty["A_K"] = ws[ws["replace"] == 2]["name"].tolist()
bin_to_determine["A_K"] = []


# ===== helper class =====
class FileSaver:
    def __init__(self, folder):
        self.folder = folder
        if not os.path.exists(folder):
            os.makedirs(folder)

    def save_list(self, lst, fname):
        assert isinstance(lst, list), "\"lst\" must be an instance of list"
        with open(os.path.join(self.folder, fname + ".lst"), 'w') as f:
            for v in lst:
                f.write(str(v) + '\n')

    def save_dict(self, dic, fname, **kw):
        assert isinstance(dic, dict), "\"dic\" must be an instance of dict"
        kw.setdefault("indent", 2)
        with open(os.path.join(folder, fname + ".dict"), 'w') as f:
            json.dump(dic, f, **kw)


fs = FileSaver('./checked/')
fs.save_list(bin_no_repl["A_K"], "brand_in_name_norepl_A_K")
fs.save_list(bin_repl["A_K"], "brand_in_name_repl_A_K")
fs.save_list(bin_empty["A_K"], "brand_in_name_empty_A_K")


special_brand_list2 = [
    "% Pure",
    "KangaRoos",  # should belongs to "LuLaRoe"
    "Karl Lagerfeld",  # should belongs to "Chanel" possibly
    "ULTra",  # only 1 sample, might be mislabelled
    "Ultra Pro",  # only 3 samples and they looks unrelated
    "Umbra",  # the missing samples looks unrelated to the labelled samples; but themselves are similar
    "Unik",  # only one sample; this is a real brand, the missing sample seems unrelated
    "USAF",  # the missing samples looks unrelated to the labelled samples; but themselves are similar
    "Vandor",  # the missing samples looks unrelated to the labelled sample
    # "Vanilla Sugar",  # the missing samples looks unrelated to the labelled samples; but themselves are similar
    # "Vantage", # the missing samples looks unrelated to the labelled samples; but themselves are similar
    # "Veet", # the missing samples looks unrelated to the labelled samples; but themselves are similar
    "Velocity",  # seems like mislabelled
    "Velvet",  # seems mislabelled
    "Vilebrequin", # the labelled seems non-sense
    "Viper",  # seems mislabelled
    "ViX",  # might be confused with "Matteify vix"
    "Wanted",  # seems mislabelled
    "Warrior",  # some missing are related and some are not
    "Wendy Bellissimo",  # the missing samples looks unrelated to the labelled samples; but themselves are similar
    "Wet",
    "Winchester",  # some missing samples are related and some are not
    "Wish",  # some missing samples are related and some are not
    "World's Best",  # this label is bullshit
    "Worth",  # some missing samples are related and some are not
]

fs.save_list(special_brand_list2, "brand_mislabeled_p2.lst")


stemmer = PorterStemmer()  # get NLTK stemmer


@task("find samples with brand in name")
def df_with_bin(df, bname, miss=True, strict=False, stem=False):
    if strict:
        idx = df[df["name"].map(lambda x: bname.lower() in x.lower().split())].index
        if stem:
            df_copy = df.iloc[idx].copy()
            bname = stemmer.stem(bname.lower())
            ret = df_copy[df_copy["name"].map(lambda x: bname in [stemmer.stem(w) for w in x.lower().split()])]
        else:
            ret = df.iloc[idx]
    else:
        ret = df[df["name"].map(lambda x: bname.lower() in x.lower())]
    return ret[ret["brand_name"].isnull()] if miss else ret
