import numpy as np
import pytest
from numpy.testing import assert_array_equal

import mini_pandas as mpd
from tests import assert_df_equals

a = np.array(["a", "b", "c"])
b = np.array(["c", "d", None])
c = np.random.rand(3)
d = np.array([True, False, True])
e = np.array([1, 2, 3])
df = mpd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})


class TestSelection:
    def test_one_column(self):
        assert_array_equal(df["a"].values[:, 0], a)
        assert_array_equal(df["c"].values[:, 0], c)

    def test_multiple_columns(self):
        cols = ["a", "c"]
        df_result = df[cols]
        df_answer = mpd.DataFrame({"a": a, "c": c})
        assert_df_equals(df_result, df_answer)

    def test_simple_boolean(self):
        bool_arr = np.array([True, False, False])
        df_bool = mpd.DataFrame({"col": bool_arr})
        df_result = df[df_bool]
        df_answer = mpd.DataFrame(
            {
                "a": a[bool_arr],
                "b": b[bool_arr],
                "c": c[bool_arr],
                "d": d[bool_arr],
                "e": e[bool_arr],
            }
        )
        assert_df_equals(df_result, df_answer)

        with pytest.raises(ValueError):
            df_bool = mpd.DataFrame({"col": bool_arr, "col2": bool_arr})
            df[df_bool]

        with pytest.raises(TypeError):
            df_bool = mpd.DataFrame({"col": np.array[1, 2, 3]})  # type: ignore

    def test_one_column_tuple(self):
        assert_df_equals(df[:, "a"], mpd.DataFrame({"a": a}))

    def test_multiple_columns_tuple(self):
        cols = ["a", "c"]
        df_result = df[:, cols]
        df_answer = mpd.DataFrame({"a": a, "c": c})
        assert_df_equals(df_result, df_answer)

    def test_int_selection(self):
        assert_df_equals(df[:, 3], mpd.DataFrame({"d": d}))

    def test_simultaneous_tuple(self):
        with pytest.raises(TypeError):
            s = set()
            df[s]

        with pytest.raises(ValueError):
            df[1, 2, 3]

    def test_single_element(self):
        df_answer = mpd.DataFrame({"e": np.array([2])})
        assert_df_equals(df[1, "e"], df_answer)

    def test_all_row_selections(self):
        df1 = mpd.DataFrame(
            {"a": np.array([True, False, True]), "b": np.array([1, 3, 5])}
        )
        with pytest.raises(ValueError):
            df[df1, "e"]

        with pytest.raises(TypeError):
            df[df1["b"], "c"]

        df_result = df[df1["a"], "c"]
        df_answer = mpd.DataFrame({"c": c[[True, False, True]]})
        assert_df_equals(df_result, df_answer)

        df_result = df[[1, 2], 0]
        df_answer = mpd.DataFrame({"a": a[[1, 2]]})
        assert_df_equals(df_result, df_answer)

        df_result = df[1:, 0]
        assert_df_equals(df_result, df_answer)

    def test_list_columns(self):
        df_answer = mpd.DataFrame({"c": c, "e": e})
        assert_df_equals(df[:, [2, 4]], df_answer)
        assert_df_equals(df[:, [2, "e"]], df_answer)
        assert_df_equals(df[:, ["c", "e"]], df_answer)

        df_result = df[2, ["a", "e"]]
        df_answer = mpd.DataFrame({"a": a[[2]], "e": e[[2]]})
        assert_df_equals(df_result, df_answer)

        df_answer = mpd.DataFrame({"c": c[[1, 2]], "e": e[[1, 2]]})
        assert_df_equals(df[[1, 2], ["c", "e"]], df_answer)

        df1 = mpd.DataFrame(
            {"a": np.array([True, False, True]), "b": np.array([1, 3, 5])}
        )
        df_answer = mpd.DataFrame({"c": c[[0, 2]], "e": e[[0, 2]]})
        assert_df_equals(df[df1["a"], ["c", "e"]], df_answer)

    def test_col_slice(self):
        df_answer = mpd.DataFrame({"a": a, "b": b, "c": c})
        assert_df_equals(df[:, :3], df_answer)

        df_answer = mpd.DataFrame({"a": a[::2], "b": b[::2], "c": c[::2]})
        assert_df_equals(df[::2, :3], df_answer)

        df_answer = mpd.DataFrame(
            {"a": a[::2], "b": b[::2], "c": c[::2], "d": d[::2], "e": e[::2]}
        )
        assert_df_equals(df[::2, :], df_answer)

        with pytest.raises(TypeError):
            df[:, set()]

    def test_tab_complete(self):
        assert ["a", "b", "c", "d", "e"] == df._ipython_key_completions_()

    def test_new_column(self):
        df_result = mpd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
        f = np.array([1.5, 23, 4.11])
        df_result["f"] = f
        df_answer = mpd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e, "f": f})
        assert_df_equals(df_result, df_answer)

        df_result = mpd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
        df_result["f"] = True
        f = np.repeat(True, 3)
        df_answer = mpd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e, "f": f})
        assert_df_equals(df_result, df_answer)

        df_result = mpd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
        f = np.array([1.5, 23, 4.11])
        df_result["c"] = f
        df_answer = mpd.DataFrame({"a": a, "b": b, "c": f, "d": d, "e": e})
        assert_df_equals(df_result, df_answer)

        with pytest.raises(NotImplementedError):
            df[["a", "b"]] = 5

        with pytest.raises(ValueError):
            df["a"] = np.random.rand(5, 5)

        with pytest.raises(ValueError):
            df["a"] = np.random.rand(5)

        with pytest.raises(ValueError):
            df["a"] = df[["a", "b"]]

        with pytest.raises(ValueError):
            df1 = mpd.DataFrame({"a": np.random.rand(5)})
            df["a"] = df1

        with pytest.raises(TypeError):
            df["a"] = set()

    def test_head_tail(self):
        df_result = df.head(2)
        df_answer = mpd.DataFrame(
            {"a": a[:2], "b": b[:2], "c": c[:2], "d": d[:2], "e": e[:2]}
        )
        assert_df_equals(df_result, df_answer)

        df_result = df.tail(2)
        df_answer = mpd.DataFrame(
            {"a": a[-2:], "b": b[-2:], "c": c[-2:], "d": d[-2:], "e": e[-2:]}
        )
        assert_df_equals(df_result, df_answer)
