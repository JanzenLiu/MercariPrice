import os
import json
import pandas as pd
import pickle


class FileSaver:
    def __init__(self, folder):
        self.folder = folder
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _get_path(self, fname, ext):
        return os.path.join(self.folder, "{}.{}".format(fname, ext))

    def save_list(self, lst, fname):
        assert isinstance(lst, list), "\"lst\" must be an instance of list"  # TODO(Janzen): to verify
        path = self._get_path(fname, "lst")
        with open(path, 'w') as f:
            for v in lst:
                f.write(str(v) + '\n')

    def save_dict(self, dic, fname, **kw):
        assert isinstance(dic, dict), "\"dic\" must be an instance of dict"
        kw.setdefault("indent", 2)
        path = self._get_path(fname, "dict")
        with open(path, 'w') as f:
            json.dump(dic, f, **kw)

    def save_csv(self, df, fname, **kw):
        assert isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)  # TODO(Janzen): to verify
        path = self._get_path(fname, "csv")
        df.to_csv(path, **kw)

    def save_tsv(self, df, fname, **kw):
        assert isinstance(df, pd.DataFrame)
        kw.pop("sep", None)  # pop item with key "sep" out
        path = self._get_path(fname, "tsv")
        df.to_csv(path, sep="\t", **kw)

    def save_pickle(self, obj, fname, **kw):
        path = self._get_path(fname, "pkl")
        pickle.dump(obj, path, **kw)


class FileLoader:
    def __init__(self, folder):
        self.folder = folder

    def _get_path(self, fname):
        return os.path.join(self.folder, fname)

    def load_list(self, fname, fast=True):
        path = self._get_path(fname)
        try:
            with open(path, "r") as f:
                if fast:
                    return [line.strip() for line in f.readlines()]
                else:
                    lines = []
                    while True:
                        line = f.readline()
                        if line:
                            lines.append(line)
                        else:
                            break
                    return lines
        except Exception as e:
            return None
