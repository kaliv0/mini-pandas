from collections import defaultdict
from enum import StrEnum

import numpy as np

__version__ = "0.0.1"

"""
TODO:
- extract as method 'next(iter(item._data.values()))'
- - or extract generic method get_first => next(iter())?
- extract error throwing checks?
- enable users to do row-wise aggregations (!?)
- add property for self._data.items()
- refactor _non_agg method -> extract kinds const (as enum ?) and adjust round method
"""


class PivotType(StrEnum):
    ROWS = "rows"
    COLUMNS = "columns"
    ALL = "all"


class DFType(StrEnum):  # TODO: rename e.g. BOOL="b"
    U = "U"
    O = "O"
    B = "b"
    F = "f"


class DataFrame:
    """
    Two-dimensional, size-mutable, potentially heterogeneous tabular data
    """

    DTYPE_NAME = {
        "O": "string",
        "i": "int",
        "f": "float",
        "b": "bool",
    }  # TODO: extract and combine with DFType

    def __init__(self, data):
        self._check_input_types(data)
        self._check_array_lengths(data)
        self._data = self._convert_unicode_to_object(data)
        self.str = StringMethods(self)
        self._add_docs()

    @staticmethod
    def _check_input_types(data):
        if not isinstance(data, dict):
            raise TypeError("`data` must be a dictionary of 1-D NumPy arrays")
        for col, vals in data.items():
            if not isinstance(col, str):
                raise TypeError("All column names must be strings")
            if not isinstance(vals, np.ndarray):
                raise TypeError("All values must be a 1-D NumPy array")
            if vals.ndim != 1:  # use check_ndim method
                raise ValueError("Each value must be a 1-D NumPy array")

    @staticmethod
    def _check_array_lengths(data: dict):
        vals = iter(data.values())
        val_len = len(next(vals))
        if any(len(val) != val_len for val in vals):
            raise ValueError("All values must be the same length")

    @staticmethod
    def _convert_unicode_to_object(data):
        return {
            col: (
                vals.astype(DFType.O)
                if vals.dtype.kind == DFType.U  # TODO extract method and use as lambda?
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
            raise TypeError("New columns must be a list")
        if len(columns) != len(self.columns):
            raise ValueError(f"New column length must be {len(self._data)}")
        if not all(isinstance(col, str) for col in columns):
            raise TypeError("New column names must be strings")
        if len(columns) != len(set(columns)):
            raise ValueError("Column names must be unique")
        self._data = dict(zip(columns, self.rows))

    # FIXME -> remove
    @property
    def rows(self):
        return list(self._data.values())

    @property
    def shape(self):
        """
        Return a tuple representing the dimensionality of the DataFrame
        """
        return len(self), len(self.columns)

    @property
    def values(self):
        """
        Return a Numpy representation of the DataFrame
        """
        return np.column_stack(tuple(self.rows))

    @property
    def dtypes(self):
        """
        Return the dtypes in the DataFrame.
        """
        data_types = [self.DTYPE_NAME[row.dtype.kind] for row in self.rows]
        return DataFrame(
            {
                "Column Name": np.array(self.columns),
                "Data Type": np.array(data_types),
            }
        )

    def __getitem__(self, item):
        match item:
            case str():
                return DataFrame({item: self._data[item]})
            case list():
                return DataFrame({col: self._data[col] for col in item})
            case DataFrame():
                return self._getitem_bool(item)
            case tuple():
                return self._getitem_tuple(item)
            case _:
                # if we got so far, apparently something is wrong
                raise TypeError(
                    "Select with either a string, a list, or a row and column"
                )

    def _getitem_bool(self, item):
        bools = self._get_df_selection(item)
        new_data = {col: vals[bools] for col, vals in self._data.items()}
        return DataFrame(new_data)

    def _getitem_tuple(self, item):
        # simultaneous selection of rows and cols -> df[rs, cs]
        if len(item) != 2:
            raise ValueError(
                "Pass either a single string or a two-item tuple inside the selection operator."
            )
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
                raise TypeError(
                    "Row selection must be either an int, slice, list, or DataFrame"
                )

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
                raise TypeError(
                    "Column selection must be either an int, string, list, or slice"
                )

    @staticmethod
    def _get_df_selection(item):
        if item.shape[1] != 1:
            raise ValueError("Can only pass a one column DataFrame for selection")
        df_selection = next(iter(item.rows))
        if df_selection.dtype.kind != DFType.B:
            raise TypeError("DataFrame must be a boolean")
        return df_selection

    def __setitem__(self, key, value):
        # adds a new column or overwrites an old one
        if not isinstance(key, str):
            raise NotImplementedError("Only able to set a single column")

        match value:
            case np.ndarray():
                # TODO: extract checks as methods -> pass error messages
                if value.ndim != 1:  # use check_ndim method
                    raise ValueError("Setting array must be 1D")
                if len(value) != len(self):
                    raise ValueError("Setting array must be same length as DataFrame")
            case DataFrame():
                if value.shape[1] != 1:
                    raise ValueError("Setting DataFrame must be one column")
                if len(value) != len(self):
                    raise ValueError(
                        "Setting and Calling DataFrames must be the same length"
                    )
                value = next(iter(value.rows))
            case str() | int() | float() | bool():
                value = np.repeat(value, len(self))
            case _:
                raise TypeError(
                    "Setting value must be a NParray, DataFrame, integer, string, float, or boolean"
                )

        if value.dtype.kind == DFType.U:
            value = value.astype(DFType.O)
        self._data[key] = value

    def head(self, n=5):
        """
        Return the first n rows
        """
        return self[:n, :]

    def tail(self, n=5):
        """
        Return the last n rows
        """
        return self[-n:, :]

    # ### Aggregation methods ### #

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

    def _agg(self, agg_func):
        new_data = {}
        for col, vals in self._data.items():
            try:
                agg_val = agg_func(vals)
            except TypeError:
                continue
            new_data[col] = np.array([agg_val])
        return DataFrame(new_data)

    def isna(self):
        """
        Detect missing values.
        Return a boolean same-sized object indicating if the values are NA.
        """
        return DataFrame(
            {
                col: (
                    vals == None  # noqa -> element-wise comparison
                    if vals.dtype.kind == DFType.O
                    else np.isnan(vals)
                )
                for col, vals in self._data.items()
            }
        )

    def count(self):
        """
        Count non-NA cells for each column or row
        """
        return DataFrame(
            {
                col: np.array([len(self) - vals.sum()])
                for col, vals in self.isna()._data.items()
            }
        )

    def unique(self):
        """
        Return distinct elements in specified axis
        """
        dfs = [DataFrame({col: np.unique(vals)}) for col, vals in self._data.items()]
        if len(dfs) == 1:
            return dfs[0]
        return dfs

    def nunique(self):
        """
        Count number of distinct elements in specified axis
        """
        return DataFrame(
            {col: np.array([len(np.unique(vals))]) for col, vals in self._data.items()}
        )

    def value_counts(self, normalize=False):
        """
        Return a Series containing the frequency of each distinct row in the Dataframe
        """
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

    @staticmethod
    def _get_keys_row_counts(values, normalize):
        keys, raw_counts = np.unique(values, return_counts=True)
        order = np.argsort(-raw_counts)
        keys = keys[order]
        raw_counts = raw_counts[order]
        if normalize:
            raw_counts = raw_counts / raw_counts.sum()
        return keys, raw_counts

    # TODO: we should probably update self._data ?
    def rename(self, columns):
        """
        Rename columns
        """
        if not isinstance(columns, dict):
            raise TypeError("`columns` must be a dictionary")
        return DataFrame(
            {columns.get(col, col): vals for col, vals in self._data.items()}
        )

    # TODO: we should probably update self._data ?
    def drop(self, columns):
        """
        Drop specified labels from rows or columns
        """
        if isinstance(columns, str):
            columns = [columns]
        elif not isinstance(columns, list):
            raise TypeError("`columns` must be either a string or a list")
        return DataFrame(
            {col: vals for col, vals in self._data.items() if col not in columns}
        )

    # ### Non-aggregation methods ### #

    def abs(self):
        """
        Return a DataFrame with absolute numeric value of each element
        """
        return self._non_agg(np.abs)

    def cummin(self):
        """
        Return cumulative minimum over a DataFrame
        """
        return self._non_agg(np.minimum.accumulate)

    def cummax(self):
        """
        Return cumulative maximum over a DataFrame
        """
        return self._non_agg(np.maximum.accumulate)

    def cumsum(self):
        """
        Return cumulative sum over a DataFrame
        """
        return self._non_agg(np.cumsum)

    def clip(self, lower=None, upper=None):
        """
        Trim values at input threshold(s)

        Assigns values outside boundary to boundary values.
        Thresholds can be singular values or array like,
        and in the latter case the clipping is performed element-wise in the specified axis
        """
        return self._non_agg(np.clip, a_min=lower, a_max=upper)

    # NB: The `round` method should ignore boolean columns. -> for all others kinds="bif"
    # TODO: use DTYPE const for kinds?
    def round(self, n):
        """
        Round a DataFrame to a variable number of decimal places
        """
        return self._non_agg(np.round, kinds="if", decimals=n)

    def copy(self):
        """
        Make a copy of this object's indices and data
        """
        return self._non_agg(np.copy)

    def diff(self, n=1):
        """
        Calculate the difference of a DataFrame element
        compared with another element in the DataFrame
        (default is element in previous row)
        """

        def non_agg_func(values):
            values = values.astype("float")  # TODO: extract const
            values_shifted = np.roll(values, n)
            values = values - values_shifted
            if n >= 0:
                values[:n] = np.NAN
            else:
                values[n:] = np.NAN
            return values

        return self._non_agg(non_agg_func)

    def pct_change(self, n=1):
        """
        Fractional change between the current and a prior element
        """

        def non_agg_func(values):
            # TODO: extract const
            values = values.astype("float")
            values_shifted = np.roll(values, n)
            values = values - values_shifted
            if n >= 0:
                values[:n] = np.NAN
            else:
                values[n:] = np.NAN
            return values / values_shifted

        return self._non_agg(non_agg_func)

    def _non_agg(self, func, kinds="bif", **kwargs):  # TODO: use DTYPE const or similar
        return DataFrame(
            {
                col: func(vals, **kwargs) if vals.dtype.kind in kinds else vals
                for col, vals in self._data.items()
            }
        )

    # ### Arithmetic and comparison operators ### #

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
        if isinstance(other, DataFrame):
            if other.shape[1] != 1:
                raise ValueError("`other` must be a one-column DataFrame")
            other = next(iter(other._data.values()))
        return DataFrame(
            {
                # retrieve the underlying numpy array method and call it directly
                col: getattr(vals, op)(other)
                for col, vals in self._data.items()
            }
        )

    def sort_values(self, by, asc=True):
        match by:
            case str():
                order_indices = np.argsort(self._data[by])
            case list():
                order_indices = np.lexsort(
                    [self._data[col] for col in by[::-1]]
                )  # why?
            case _:
                raise TypeError("`by` must be a str or a list")
        if asc is False:
            order_indices = order_indices[::-1]
        return self[order_indices.tolist(), :]

    def sample(self, n=None, frac=None, replace=False, seed=None):
        if frac is None and n is None:
            raise ValueError("Pass either `frac` or `n` value")
        if frac:
            if frac <= 0:
                raise ValueError("`frac` must be positive")
            n = int(frac * len(self))
        if not isinstance(n, int):
            raise TypeError("`n` must be an int")
        if seed:
            np.random.seed(seed)
        rows = np.random.choice(np.arange(len(self)), size=n, replace=replace).tolist()
        return self[rows, :]

    # ### Pivot ### #
    def pivot_table(self, rows=None, columns=None, values=None, agg_func=None):
        """
        Create a spreadsheet-style pivot table as a DataFrame
        """
        col_data, row_data, pivot_type = self._get_grouping_data(columns, rows)
        val_data, agg_func = self._get_agg_data(values, agg_func)
        agg_dict = self._get_agg_dict(
            col_data, row_data, pivot_type, val_data, agg_func
        )
        return self._create_pivoted_data_frame(rows, pivot_type, agg_dict, agg_func)

    def _get_grouping_data(self, columns, rows):
        if rows is None and columns is None:
            raise ValueError("`rows` or `columns` cannot both be `None`")
        pivot_type = PivotType.ALL
        row_data = []
        col_data = []
        if rows:
            row_data = self._data[rows]
        else:
            pivot_type = PivotType.COLUMNS

        if columns:
            col_data = self._data[columns]
        else:
            pivot_type = PivotType.ROWS
        return col_data, row_data, pivot_type

    def _get_agg_data(self, values, agg_func):
        # FIXME
        if values:
            val_data = self._data[values]
            if agg_func is None:
                raise ValueError(
                    "You must provide `aggfunc` when `values` is provided."
                )
        else:
            if agg_func is None:
                agg_func = "size"  # TODO: extract const?
                val_data = np.empty(len(self))
            else:
                raise ValueError("You cannot provide `aggfunc` when `values` is None")
        return val_data, agg_func

    def _get_agg_dict(self, col_data, row_data, pivot_type, val_data, agg_func):
        group_dict = self._get_group_dict(col_data, row_data, pivot_type, val_data)
        return {
            # NB: Since `agg_func` is a string, you will need to use the builtin `getattr` function
            # to get the correct numpy function.
            group: getattr(np, agg_func)(np.array(vals))
            for group, vals in group_dict.items()
        }

    @staticmethod
    def _get_group_dict(col_data, row_data, pivot_type, val_data):
        group_dict = defaultdict(list)
        match pivot_type:
            case PivotType.COLUMNS:
                for group, val in zip(col_data, val_data):
                    group_dict[group].append(val)
            case PivotType.ROWS:
                for group, val in zip(row_data, val_data):
                    group_dict[group].append(val)
            case PivotType.ALL:
                for group1, group2, val in zip(row_data, col_data, val_data):
                    group_dict[(group1, group2)].append(val)
        return group_dict

    def _create_pivoted_data_frame(self, rows, pivot_type, agg_dict, agg_func):
        match pivot_type:
            case PivotType.COLUMNS:
                return self._create_df_for_pt_columns(agg_dict)
            case PivotType.ROWS:
                return self._create_df_for_pt_rows(agg_dict, agg_func, rows)
            case PivotType.ALL:
                return self._create_df_for_pt_all(agg_dict, rows)

    @staticmethod
    def _create_df_for_pt_columns(agg_dict):
        return DataFrame(
            {col_name: np.array([agg_dict[col_name]]) for col_name in sorted(agg_dict)}
        )

    @staticmethod
    def _create_df_for_pt_rows(agg_dict, agg_func, rows):
        row_arr = np.array(list(agg_dict.keys()))
        val_arr = np.array(list(agg_dict.values()))
        order = np.argsort(row_arr)
        return DataFrame({rows: row_arr[order], agg_func: val_arr[order]})

    @staticmethod
    def _create_df_for_pt_all(agg_dict, rows):
        row_set = set()
        col_set = set()
        for group in agg_dict:
            row_set.add(group[0])
            col_set.add(group[1])

        row_list = sorted(row_set)
        col_list = sorted(col_set)
        new_data = {rows: np.array(row_list)}
        for col in col_list:
            new_data[col] = np.array(
                [agg_dict.get((row, col), np.nan) for row in row_list]
            )
        return DataFrame(new_data)

    # ### Helpers ### #
    @staticmethod
    def _add_docs():
        # TODO: extract const
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
        """
        for name in agg_names:
            getattr(DataFrame, name).__doc__ = agg_doc.format(name)

    # ### jupyter helper methods ### #
    def _ipython_key_completions_(self):
        return self.columns

    def _repr_html_(self):
        num_head, num_tail, only_head = self._get_building_params()
        return (
            self._prepare_html_table()
            + self._build_html_head(num_head)
            + self._build_html_body(only_head, num_tail)
            + self._prepare_html_closing()
        )

    def _get_building_params(self):
        # returns num_head, num_tail, only_head
        if len(self) <= 20:
            return len(self), 10, True
        return 10, 10, False

    def _prepare_html_table(self):
        html = "<table><thead><tr><th></th>"
        for col in self.columns:
            html += f"<th>{col:10}</th>"
        html += "</tr></thead>"
        html += "<tbody>"
        return html

    def _build_html_head(self, num_head):
        html = ""
        for i in range(num_head):
            html = f"<tr><td><strong>{i}</strong></td>"
            for col, vals in self._data.items():
                kind = vals.dtype.kind
                match kind:
                    case DFType.F:
                        html += f"<td>{vals[i]:10.3f}</td>"
                    case DFType.B:
                        html += f"<td>{vals[i]}</td>"
                    case DFType.O:
                        v = vals[i]
                        if v is None:
                            v = "None"
                        html += f"<td>{v:10}</td>"
                    case _:
                        html += f"<td>{vals[i]:10}</td>"
            html += "</tr>"
        return html

    def _build_html_body(self, only_head, num_tail):
        if only_head:
            return ""

        html = "<tr><strong><td>...</td></strong>"
        for i in range(len(self.columns)):
            html += "<td>...</td>"
        html += "</tr>"
        for i in range(-num_tail, 0):
            html += f"<tr><td><strong>{len(self) + i}</strong></td>"
            for col, vals in self._data.items():
                kind = vals.dtype.kind
                match kind:
                    case DFType.F:
                        html += f"<td>{vals[i]:10.3f}</td>"
                    case DFType.B:
                        html += f"<td>{vals[i]}</td>"
                    case DFType.O:
                        v = vals[i]
                        if v is None:
                            v = "None"
                        html += f"<td>{v:10}</td>"
                    case _:
                        html += f"<td>{vals[i]:10}</td>"
            html += "</tr>"
        return html

    @staticmethod
    def _prepare_html_closing():
        return "</tbody></table>"


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
        old_values = self._df._data[col]  # FIXME
        # TODO: use DTYPE
        if old_values.dtype.kind != "O":
            raise TypeError("The `str` accessor only works with string columns")
        return DataFrame(
            {col: np.array([method(val, *args) if val else val for val in old_values])}
        )


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
            # TODO: use DTYPE, add parsing ass dtype="boolean"?
            new_data[col] = np.array(vals, dtype="int")
        except ValueError:
            try:
                new_data[col] = np.array(vals, dtype="float")
            except ValueError:
                new_data[col] = np.array(vals, dtype="O")
    return DataFrame(new_data)


def _read_data_from_file(fn):
    values = defaultdict(list)
    with open(fn) as f:
        header = f.readline()
        column_names = header.strip("\n").split(",")  # TODO: extract consts
        for line in f:
            vals = line.strip("\n").split(",")
            for val, name in zip(vals, column_names):
                values[name].append(val)
    return values
