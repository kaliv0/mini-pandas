import numpy as np

from mini_pandas.constants import DFType, Tokens
from mini_pandas.dataframe import DataFrame


def read_csv(fn):
    """
    Read a comma-separated values (csv) file into DataFrame
    """
    values = _read_data_from_file(fn)
    return _create_data_frame(values)


def _create_data_frame(values):
    new_data = {}
    for col, vals in values.items():
        try:
            new_data[col] = np.array(vals, dtype=DFType.INT)
        except ValueError:
            try:
                new_data[col] = np.array(vals, dtype=DFType.FLOAT)
            except ValueError:
                try:
                    new_data[col] = np.array(vals, dtype=DFType.BOOL)
                except ValueError:
                    new_data[col] = np.array(vals, dtype=DFType.STRING)
    return DataFrame(new_data)


def _read_data_from_file(fn):
    from collections import defaultdict

    values = defaultdict(list)
    with open(fn) as f:
        header = f.readline()
        column_names = header.strip(Tokens.NEW_LINE).split(Tokens.DELIMITER)
        for line in f:
            vals = line.strip(Tokens.NEW_LINE).split(Tokens.DELIMITER)
            for val, name in zip(vals, column_names):
                values[name].append(val)
    return values
