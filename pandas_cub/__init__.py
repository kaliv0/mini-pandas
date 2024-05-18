import numpy as np

__version__ = "0.0.1"

"""
TODO:
- add docstrings
- add error msg
- refactor df_repr (html build)
- extract as method 'next(iter(item._data.values()))'
- extract error throwing checks?
- enable users to do row-wise aggregations (!?)
- add property for self._data.items()
"""


class DataFrame:
    DTYPE_NAME = {"O": "string", "i": "int", "f": "float", "b": "bool"}

    def __init__(self, data):
        self._check_input_types(data)
        self._check_array_lengths(data)
        self._data = self._convert_unicode_to_object(data)

        # Allow for special methods for strings
        self.str = StringMethods(self)
        self._add_docs()

    @staticmethod
    def _check_input_types(data):
        if not isinstance(data, dict):
            raise TypeError
        for col, vals in data.items():
            if not isinstance(col, str):
                raise TypeError
            if not isinstance(vals, np.ndarray):
                raise TypeError
            if vals.ndim != 1:  # use check_ndim method
                raise ValueError

    @staticmethod
    def _check_array_lengths(data: dict):
        vals = iter(data.values())
        val_len = len(next(vals))
        if any(len(val) != val_len for val in vals):
            raise ValueError

    @staticmethod
    def _convert_unicode_to_object(data):
        return {
            col: (
                vals.astype("O")
                if vals.dtype.kind == "U"  # extract method and use as lambda?
                else vals
            )
            for col, vals in data.items()
        }

    def __len__(self):
        return len(next(iter(self.rows)))

    @property
    def columns(self):
        return list(self._data)

    @columns.setter
    def columns(self, columns):
        if not isinstance(columns, list):
            raise TypeError
        if len(columns) != len(self.columns):
            raise ValueError
        if not all(isinstance(col, str) for col in columns):
            raise TypeError
        if len(columns) != len(set(columns)):
            raise ValueError

        new_data = dict(zip(columns, self.rows))
        self._data = new_data

    @property
    def rows(self):
        return list(self._data.values())

    @property
    def shape(self):
        return len(self), len(self.columns)

    @property
    def values(self):
        return np.column_stack(tuple(self.rows))

    @property
    def dtypes(self):
        data_types = [self.DTYPE_NAME[row.dtype.kind] for row in self.rows]
        return DataFrame(
            {
                "Column Name": np.array(self.columns),
                "Data Type": np.array(data_types),
            }
        )

    def __getitem__(self, item):
        """
        Row and column selection simultaneously -> df[rs, cs]
            where cs and rs can be integers, slices, or a list of integers
            rs can also be a one-column boolean DataFrame
        """

        match item:
            case str():
                # TODO: what should we do if there is no such key?
                return DataFrame({item: self._data[item]})
            case list():
                return DataFrame({col: self._data[col] for col in item})
            case DataFrame():
                return self._getitem_bool(item)
            case tuple():
                return self._getitem_tuple(item)
            case _:
                # if we got so far, apparently something is wrong
                raise TypeError

    def _getitem_bool(self, item):
        bools = self._get_df_selection(item)
        new_data = {col: vals[bools] for col, vals in self._data.items()}
        return DataFrame(new_data)

    def _getitem_tuple(self, item):
        # simultaneous selection of rows and cols -> df[rs, cs]
        if len(item) != 2:
            raise ValueError

        row_selection = self._get_row_selection(item[0])
        col_selection = self._get_col_selection(item[1])
        new_data = {col: self._data[col][row_selection] for col in col_selection}
        return DataFrame(new_data)

    def _get_row_selection(self, selection):
        match selection:
            case int():
                return [selection]
            case DataFrame():
                return self._get_df_selection(selection)
            case list() | slice():
                return selection
            case _:
                raise TypeError

    def _get_col_selection(self, selection):
        match selection:
            case int():
                return [self.columns[selection]]
            case str():
                return [selection]
            case list():  # extract methods?
                return [
                    self.columns[col] if isinstance(col, int) else col
                    for col in selection
                ]
            case slice():  # extract methods?
                start = (
                    self.columns.index(selection.start)
                    if isinstance(selection.start, str)
                    else selection.start
                )
                stop = (
                    self.columns.index(selection.stop) + 1
                    if isinstance(selection.stop, str)
                    else selection.stop
                )
                step = selection.step
                return self.columns[start:stop:step]
            # Column selection must be either an int, string, list, or slice
            case _:
                raise TypeError

    @staticmethod
    def _get_df_selection(item):
        if item.shape[1] != 1:
            raise ValueError
        df_selection = next(iter(item.rows))
        if df_selection.dtype.kind != "b":
            raise TypeError
        return df_selection

    def _ipython_key_completions_(self):
        # allows for tab completion when doing df['c
        return self.columns

    def __setitem__(self, key, value):
        # adds a new column or overwrites an old column
        if not isinstance(key, str):
            raise NotImplementedError

        match value:
            case np.ndarray():
                # TODO: extract checks as methods -> pass error messages
                if value.ndim != 1:  # use check_ndim method
                    raise ValueError
                if len(value) != len(self):
                    raise ValueError  # TODO: different message
            case DataFrame():
                if value.shape[1] != 1:
                    raise ValueError
                if len(value) != len(self):
                    raise ValueError
                value = next(iter(value.rows))
            case str() | int() | float() | bool():
                value = np.repeat(value, len(self))
            case _:
                raise TypeError

        if value.dtype.kind == "U":
            value = value.astype("O")
        self._data[key] = value

    def head(self, n=5):
        # first n rows and all columns
        return self[:n, :]

    def tail(self, n=5):
        # last n rows
        return self[-n:, :]

    # ### Aggregation Methods ### #

    def min(self):
        return self._agg(np.min)

    def max(self):
        return self._agg(np.max)

    def mean(self):
        return self._agg(np.mean)

    def median(self):
        return self._agg(np.median)

    def sum(self):
        return self._agg(np.sum)

    def var(self):
        return self._agg(np.var)

    def std(self):
        return self._agg(np.std)

    def all(self):
        return self._agg(np.all)

    def any(self):
        return self._agg(np.any)

    def argmax(self):
        return self._agg(np.argmax)

    def argmin(self):
        return self._agg(np.argmin)

    def _agg(self, func):
        new_data = {}
        for col, vals in self._data.items():
            try:
                agg_val = func(vals)
            except TypeError:
                continue
            new_data[col] = np.array([agg_val])
        return DataFrame(new_data)

    def isna(self):
        return DataFrame(
            {
                col: (
                    vals == None  # noqa -> element-wise comparison
                    if vals.dtype.kind == "O"
                    else np.isnan(vals)
                )
                for col, vals in self._data.items()
            }
        )

    def count(self):
        return DataFrame(
            {
                col: np.array([len(self) - vals.sum()])
                for col, vals in self.isna()._data.items()
            }
        )

    def unique(self):
        dfs = [DataFrame({col: np.unique(vals)}) for col, vals in self._data.items()]
        if len(dfs) == 1:
            return dfs[0]
        return dfs

    def nunique(self):
        return DataFrame(
            {col: np.array([len(np.unique(vals))]) for col, vals in self._data.items()}
        )

    def value_counts(self, normalize=False):
        dfs = []
        for col, values in self._data.items():
            keys, raw_counts = self._get_keys_row_counts(values, normalize)
            df = DataFrame(
                {
                    col: keys,
                    "count": raw_counts,
                }
            )
            dfs.append(df)
        if len(dfs) == 1:
            return dfs[0]
        return dfs

    def _get_keys_row_counts(self, values, normalize):
        keys, raw_counts = np.unique(values, return_counts=True)
        order = np.argsort(-raw_counts)
        keys = keys[order]
        raw_counts = raw_counts[order]
        if normalize:
            raw_counts = raw_counts / raw_counts.sum()
        return keys, raw_counts

    # TODO: we should probably update self._data ?
    def rename(self, columns):
        if not isinstance(columns, dict):
            raise TypeError
        return DataFrame(
            {columns.get(col, col): vals for col, vals in self._data.items()}
        )

    # TODO: we should probably update self._data ?
    def drop(self, columns):
        if isinstance(columns, str):
            columns = [columns]
        elif not isinstance(columns, list):
            raise TypeError
        return DataFrame(
            {col: vals for col, vals in self._data.items() if col not in columns}
        )

    # ### Non-Aggregation Methods ### #

    def abs(self):
        return self._non_agg(np.abs)

    def cummin(self):
        return self._non_agg(np.minimum.accumulate)

    def cummax(self):
        return self._non_agg(np.maximum.accumulate)

    def cumsum(self):
        return self._non_agg(np.cumsum)

    def clip(self, lower=None, upper=None):
        return self._non_agg(np.clip, a_min=lower, a_max=upper)

    # NB: The `round` method should ignore boolean columns. -> for all others kinds="bif"
    def round(self, n):
        return self._non_agg(np.round, kinds="if", decimals=n)

    def copy(self):
        return self._non_agg(np.copy)

    def _non_agg(self, func, kinds="bif", **kwargs):
        return DataFrame(
            {
                col: func(vals, **kwargs) if vals.dtype.kind in kinds else vals
                # else vals.copy() # TODO: do we need copy the data -> here and not in other occasions?
                for col, vals in self._data.items()
            }
        )

    def diff(self, n=1):
        def func():
            pass

        return self._non_agg(func)

    def pct_change(self, n=1):
        def func():
            pass

        return self._non_agg(func)

    # ### Arithmetic and Comparison Operators ### #

    def __add__(self, other):
        return self._oper("__add__", other)

    def __radd__(self, other):
        return self._oper("__radd__", other)

    def __sub__(self, other):
        return self._oper("__sub__", other)

    def __rsub__(self, other):
        return self._oper("__rsub__", other)

    def __mul__(self, other):
        return self._oper("__mul__", other)

    def __rmul__(self, other):
        return self._oper("__rmul__", other)

    def __truediv__(self, other):
        return self._oper("__truediv__", other)

    def __rtruediv__(self, other):
        return self._oper("__rtruediv__", other)

    def __floordiv__(self, other):
        return self._oper("__floordiv__", other)

    def __rfloordiv__(self, other):
        return self._oper("__rfloordiv__", other)

    def __pow__(self, other):
        return self._oper("__pow__", other)

    def __rpow__(self, other):
        return self._oper("__rpow__", other)

    def __gt__(self, other):
        return self._oper("__gt__", other)

    def __lt__(self, other):
        return self._oper("__lt__", other)

    def __ge__(self, other):
        return self._oper("__ge__", other)

    def __le__(self, other):
        return self._oper("__le__", other)

    def __ne__(self, other):
        return self._oper("__ne__", other)

    def __eq__(self, other):
        return self._oper("__eq__", other)

    def _oper(self, op, other):
        pass

    def sort_values(self, by, asc=True):
        pass

    def sample(self, n=None, frac=None, replace=False, seed=None):
        pass

    def pivot_table(self, rows=None, columns=None, values=None, aggfunc=None):
        pass

    @staticmethod
    def _add_docs():
        agg_names = [
            "min",
            "max",
            "mean",
            "median",
            "sum",
            "var",
            "std",
            "any",
            "all",
            "argmax",
            "argmin",
        ]
        agg_doc = """
        Find the {} of each column
        
        Returns
        -------
        DataFrame
        """
        for name in agg_names:
            getattr(DataFrame, name).__doc__ = agg_doc.format(name)

    # helpers
    @staticmethod
    def _check_ndim(value):
        if value.ndim != 1:
            raise ValueError

    def _repr_html_(self):
        html = "<table><thead><tr><th></th>"
        for col in self.columns:
            html += f"<th>{col:10}</th>"

        html += "</tr></thead>"
        html += "<tbody>"

        only_head = False
        num_head = 10
        num_tail = 10
        if len(self) <= 20:
            only_head = True
            num_head = len(self)

        for i in range(num_head):
            html += f"<tr><td><strong>{i}</strong></td>"
            for col, vals in self._data.items():
                kind = vals.dtype.kind
                if kind == "f":
                    html += f"<td>{vals[i]:10.3f}</td>"
                elif kind == "b":
                    html += f"<td>{vals[i]}</td>"
                elif kind == "O":
                    v = vals[i]
                    if v is None:
                        v = "None"
                    html += f"<td>{v:10}</td>"
                else:
                    html += f"<td>{vals[i]:10}</td>"
            html += "</tr>"

        if not only_head:
            html += "<tr><strong><td>...</td></strong>"
            for i in range(len(self.columns)):
                html += "<td>...</td>"
            html += "</tr>"
            for i in range(-num_tail, 0):
                html += f"<tr><td><strong>{len(self) + i}</strong></td>"
                for col, vals in self._data.items():
                    kind = vals.dtype.kind
                    if kind == "f":
                        html += f"<td>{vals[i]:10.3f}</td>"
                    elif kind == "b":
                        html += f"<td>{vals[i]}</td>"
                    elif kind == "O":
                        v = vals[i]
                        if v is None:
                            v = "None"
                        html += f"<td>{v:10}</td>"
                    else:
                        html += f"<td>{vals[i]:10}</td>"
                html += "</tr>"

        html += "</tbody></table>"
        return html


class StringMethods:
    def __init__(self, df):
        self._df = df

    def capitalize(self, col):
        return self._str_method(str.capitalize, col)

    def center(self, col, width, fillchar=None):
        if fillchar is None:
            fillchar = " "
        return self._str_method(str.center, col, width, fillchar)

    def count(self, col, sub, start=None, stop=None):
        return self._str_method(str.count, col, sub, start, stop)

    def endswith(self, col, suffix, start=None, stop=None):
        return self._str_method(str.endswith, col, suffix, start, stop)

    def startswith(self, col, suffix, start=None, stop=None):
        return self._str_method(str.startswith, col, suffix, start, stop)

    def find(self, col, sub, start=None, stop=None):
        return self._str_method(str.find, col, sub, start, stop)

    def len(self, col):
        return self._str_method(str.__len__, col)

    def get(self, col, item):
        return self._str_method(str.__getitem__, col, item)

    def index(self, col, sub, start=None, stop=None):
        return self._str_method(str.index, col, sub, start, stop)

    def isalnum(self, col):
        return self._str_method(str.isalnum, col)

    def isalpha(self, col):
        return self._str_method(str.isalpha, col)

    def isdecimal(self, col):
        return self._str_method(str.isdecimal, col)

    def islower(self, col):
        return self._str_method(str.islower, col)

    def isnumeric(self, col):
        return self._str_method(str.isnumeric, col)

    def isspace(self, col):
        return self._str_method(str.isspace, col)

    def istitle(self, col):
        return self._str_method(str.istitle, col)

    def isupper(self, col):
        return self._str_method(str.isupper, col)

    def lstrip(self, col, chars):
        return self._str_method(str.lstrip, col, chars)

    def rstrip(self, col, chars):
        return self._str_method(str.rstrip, col, chars)

    def strip(self, col, chars):
        return self._str_method(str.strip, col, chars)

    def replace(self, col, old, new, count=None):
        if count is None:
            count = -1
        return self._str_method(str.replace, col, old, new, count)

    def swapcase(self, col):
        return self._str_method(str.swapcase, col)

    def title(self, col):
        return self._str_method(str.title, col)

    def lower(self, col):
        return self._str_method(str.lower, col)

    def upper(self, col):
        return self._str_method(str.upper, col)

    def zfill(self, col, width):
        return self._str_method(str.zfill, col, width)

    def encode(self, col, encoding="utf-8", errors="strict"):
        return self._str_method(str.encode, col, encoding, errors)

    def _str_method(self, method, col, *args):
        pass


def read_csv(fn):
    pass
