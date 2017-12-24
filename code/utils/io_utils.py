import pandas as pd
import pickle
import os
from scipy.sparse import hstack


def load_file(filepath, **params):
    # TODO: add exception handling
    # TODO: add log
    # check existence
    if not os.path.exists(filepath):
        print("error: file not exists")
        return False
    # load according to extension
    ret = None
    ext = os.path.splitext(filepath)[1][1:]
    if ext == 'tsv':
        ret = pd.read_table(filepath, **params)
    elif ext == 'csv':
        ret = pd.read_csv(filepath, **params)
    elif ext == 'p':
        with open(filepath, 'rb') as f:
            ret = pickle.load(f, **params)
    # to be complemented... (json, xgboost model, txt, etc.)
    else:
        print("error: unsupported file format {}".format(ext))

    return ret


def save_file(obj, filepath, **params):
    # TODO: add exception handling
    # TODO: add log
    # prepare folder
    folder = os.path.dirname(filepath)
    if (not os.path.exists(folder)) or (not os.path.isdir(folder)):
        os.makedirs(folder)
    # save according to extension
    ext = os.path.splitext(filepath)[1][1:]
    if ext == 'tsv':
        # assume obj is a pandas DataFrame
        # TODO: add type handling to accept type other than pandas DataFrame or reject invalid type
        assert isinstance(obj, pd.DataFrame)
        obj.to_csv(filepath, sep='/t', **params)
    elif ext == 'csv':
        # assume obj is a pandas DataFrame
        # TODO: add type handling to accept type other than pandas DataFrame or reject invalid type
        assert isinstance(obj, pd.DataFrame)
        obj.to_csv(filepath, **params)
    elif ext == 'p':
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    # to be complemented... (json, xgboost model, txt, etc.)
    else:
        print("error: unsupported file format {}".format(ext))


def load_data(file_list, param_dict=None, sparse=True):
    mat = None  # to mute IDE warning
    for i, filepath in enumerate(file_list):
        if not param_dict:
            param_dict = {}
        params = param_dict.get(filepath, {})
        new_mat = load_file(filepath, **params)
        mat = new_mat if i == 0 else hstack((mat, new_mat)).tocsr() if sparse else hstack((mat, new_mat))
    return mat
