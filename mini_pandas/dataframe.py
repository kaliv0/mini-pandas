import numpy as np

from mini_pandas.constants import (
    DFType,
    PivotType,
    DEFAULT_AGG_FUNC,
    AGG_METHOD_NAMES,
    BuildingParams,
)
from mini_pandas.str_utils import StringMethods


class DataFrame:
    """
    Two-dimensional, size-mutable, potentially heterogeneous tabular data
    """

    def __init__(self, data):
        self._check_input_types(data)
        self._check_array_lengths(data)
        self._data = self._convert_unicode_to_object(data)

        # compute once since DF isn't modified in place
        self._rows = list(self._data.values())
        self._items = self._data.items()

        self.str = StringMethods(self)
        self._add_docs()

    def _check_input_types(self, data):
        if not isinstance(data, dict):
            raise TypeError("`data` must be a dictionary of 1-D NumPy arrays")
        for col, vals in data.items():
            if not isinstance(col, str):
                raise TypeError("All column names must be strings")
            if not isinstance(vals, np.ndarray):
                raise TypeError("All values must be a 1-D NumPy array")
            if self._validate_ndim(vals):
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
                vals.astype(DFType.STRING)
                if vals.dtype.kind == DFType.U_STRING
                else vals
            )
            for col, vals in data.items()
        }

    ###############################
    def __len__(self):
        return len(self._get_first(self._rows))

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
        self._data = dict(zip(columns, self._rows))

    ###############################
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
        return np.column_stack(tuple(self._rows))

    @property
    def dtypes(self):
        """
        Return the dtypes in the DataFrame.
        """
        data_types = [DFType(row.dtype.kind).name.lower() for row in self._rows]
        return DataFrame(
            {
                "Column Name": np.array(self.columns),
                "Data Type": np.array(data_types),
            }
        )

    ###############################
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

    def _get_df_selection(self, item):
        if self._validate_shape(item):
            raise ValueError("Can only pass a one column DataFrame for selection")
        df_selection = self._get_first(item._rows)  # noqa
        if df_selection.dtype.kind != DFType.BOOL:
            raise TypeError("DataFrame must be a boolean")
        return df_selection

    def _get_col_selection(self, selection):
        match selection:
            case int():
                return [self.columns[selection]]
            case str():
                return [selection]
            case list():
                return self._get_col_selection_from_list(selection)
            case slice():
                return self._get_col_selection_from_slice(selection)
            case _:
                raise TypeError(
                    "Column selection must be either an int, string, list, or slice"
                )

    def _get_col_selection_from_list(self, selection):
        return [self.columns[col] if isinstance(col, int) else col for col in selection]

    def _get_col_selection_from_slice(self, selection):
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

    ###############################
    def __setitem__(self, key, value):
        # adds a new column or overwrites an old one
        if not isinstance(key, str):
            raise NotImplementedError("Only able to set a single column")

        match value:
            case np.ndarray():
                self._validate_ndarray_type(value)
            case DataFrame():
                self._validate_df_type(value)
                value = self._get_first(value._rows)  # noqa
            case str() | int() | float() | bool():
                value = np.repeat(value, len(self))
            case _:
                raise TypeError(
                    "Setting value must be a NParray, DataFrame, integer, string, float, or boolean"
                )

        if value.dtype.kind == DFType.U_STRING:
            value = value.astype(DFType.STRING)
        self._data[key] = value

    def _validate_ndarray_type(self, value):
        if self._validate_ndim(value):
            raise ValueError("Setting array must be 1D")
        if self._validate_len(value):
            raise ValueError("Setting array must be same length as DataFrame")

    def _validate_df_type(self, value):
        if self._validate_shape(value):
            raise ValueError("Setting DataFrame must be one column")
        if self._validate_len(value):
            raise ValueError("Setting and Calling DataFrames must be the same length")

    ###############################
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
        for col, vals in self._items:
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
                    if vals.dtype.kind == DFType.STRING
                    else np.isnan(vals)
                )
                for col, vals in self._items
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
        dfs = [DataFrame({col: np.unique(vals)}) for col, vals in self._items]
        if len(dfs) == 1:
            return dfs[0]
        return dfs

    def nunique(self):
        """
        Count number of distinct elements in specified axis
        """
        return DataFrame(
            {col: np.array([len(np.unique(vals))]) for col, vals in self._items}
        )

    def value_counts(self, normalize=False):
        """
        Return a Series containing the frequency of each distinct row in the Dataframe
        """
        dfs = []
        for col, values in self._items:
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

    def rename(self, columns):
        """
        Rename columns
        """
        if not isinstance(columns, dict):
            raise TypeError("`columns` must be a dictionary")
        return DataFrame({columns.get(col, col): vals for col, vals in self._items})

    def drop(self, columns):
        """
        Drop specified labels from rows or columns
        """
        if isinstance(columns, str):
            columns = [columns]
        elif not isinstance(columns, list):
            raise TypeError("`columns` must be either a string or a list")
        return DataFrame({col: vals for col, vals in self._items if col not in columns})

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

    def round(self, n):
        """
        Round a DataFrame to a variable number of decimal places
        """
        return self._non_agg(np.round, ignore_bool=True, decimals=n)

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
        return self._non_agg(self._diff_pct_agg_func, n=n, compute_fraction=False)

    def pct_change(self, n=1):
        """
        Fractional change between the current and a prior element
        """
        return self._non_agg(self._diff_pct_agg_func, n=n, compute_fraction=True)

    def _diff_pct_agg_func(self, values, n, compute_fraction):
        values = self._cast_to_float(values)
        values_shifted = np.roll(values, n)
        values = values - values_shifted
        if n >= 0:
            values[:n] = np.NAN
        else:
            values[n:] = np.NAN

        if compute_fraction:
            return values / values_shifted
        return values

    @staticmethod
    def _cast_to_float(values):
        return values.astype(DFType.FLOAT.name.lower())

    def _non_agg(self, func, ignore_bool=False, **kwargs):
        kinds = (
            (DFType.INT, DFType.FLOAT)
            if ignore_bool
            else (DFType.BOOL, DFType.INT, DFType.FLOAT)
        )
        return DataFrame(
            {
                col: func(vals, **kwargs) if vals.dtype.kind in kinds else vals
                for col, vals in self._items
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
            if self._validate_shape(other):
                raise ValueError("`other` must be a one-column DataFrame")
            other = self._get_first(other._data.values())
        return DataFrame(
            {
                # retrieve the underlying numpy array method and call it directly
                col: getattr(vals, op)(other)
                for col, vals in self._items
            }
        )

    def sort_values(self, by, asc=True):
        match by:
            case str():
                order_indices = np.argsort(self._data[by])
            case list():
                order_indices = np.lexsort([self._data[col] for col in by[::-1]])
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
        if values:
            val_data = self._data[values]
            if agg_func is None:
                raise ValueError(
                    "You must provide `agg_func` when `values` is provided."
                )
        else:
            if agg_func is None:
                agg_func = DEFAULT_AGG_FUNC
                val_data = np.empty(len(self))
            else:
                raise ValueError("You cannot provide `agg_func` when `values` is None")
        return val_data, agg_func

    def _get_agg_dict(self, col_data, row_data, pivot_type, val_data, agg_func):
        group_dict = self._get_group_dict(col_data, row_data, pivot_type, val_data)
        return {
            # NB: `agg_func` is a string
            group: getattr(np, agg_func)(np.array(vals))
            for group, vals in group_dict.items()
        }

    @staticmethod
    def _get_group_dict(col_data, row_data, pivot_type, val_data):
        from collections import defaultdict

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
    def _validate_shape(item):
        return item.shape[1] != 1

    @staticmethod
    def _validate_ndim(item):
        return item.ndim != 1

    def _validate_len(self, value):
        return len(value) != len(self)

    @staticmethod
    def _get_first(collection):
        return next(iter(collection))

    @staticmethod
    def _add_docs():
        agg_doc = """
        Find the {} of each column
        """
        for name in AGG_METHOD_NAMES:
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
        if len(self) <= BuildingParams.DEFAULT_COL_COUNT:
            return len(self), BuildingParams.DEFAULT_NUM_TAIL, True
        return BuildingParams.DEFAULT_NUM_HEAD, BuildingParams.DEFAULT_NUM_TAIL, False

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
            html += f"<tr><td><strong>{i}</strong></td>"
            for col, vals in self._items:
                kind = vals.dtype.kind
                match kind:
                    case DFType.FLOAT:
                        html += f"<td>{vals[i]:10.3f}</td>"
                    case DFType.BOOL:
                        html += f"<td>{vals[i]}</td>"
                    case DFType.STRING:
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
            for col, vals in self._items:
                kind = vals.dtype.kind
                match kind:
                    case DFType.FLOAT:
                        html += f"<td>{vals[i]:10.3f}</td>"
                    case DFType.BOOL:
                        html += f"<td>{vals[i]}</td>"
                    case DFType.STRING:
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
