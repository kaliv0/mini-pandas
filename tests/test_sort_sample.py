import numpy as np
import pytest

import mini_pandas as mpd
from tests import assert_df_equals

a1 = np.array(["b", "c", "a", "a", "b"])
b1 = np.array([3.4, 5.1, 2, 1, 6])
df1 = mpd.DataFrame({"a": a1, "b": b1})

a2 = np.array(["b", "a", "a", "a", "b"])
b2 = np.array([3.4, 5.1, 2, 1, 6])
df2 = mpd.DataFrame({"a": a2, "b": b2})


class TestSortSample:
    def test_sort_values(self):
        df_result = df1.sort_values("a")
        a = np.array(["a", "a", "b", "b", "c"])
        b = np.array([2, 1, 3.4, 6, 5.1])
        df_answer = mpd.DataFrame({"a": a, "b": b})
        assert_df_equals(df_result, df_answer)

    def test_sort_values_desc(self):
        df_result = df1.sort_values("a", asc=False)
        a = np.array(["c", "b", "b", "a", "a"])
        b = np.array([5.1, 6, 3.4, 1, 2])
        df_answer = mpd.DataFrame({"a": a, "b": b})
        assert_df_equals(df_result, df_answer)

    def test_sort_values_two(self):
        df_result = df2.sort_values(["a", "b"])
        a = np.array(["a", "a", "a", "b", "b"])
        b = np.array([1, 2, 5.1, 3.4, 6])
        df_answer = mpd.DataFrame({"a": a, "b": b})
        assert_df_equals(df_result, df_answer)

    def test_sort_values_two_desc(self):
        df_result = df2.sort_values(["a", "b"], asc=False)
        a = np.array(["a", "a", "a", "b", "b"])
        b = np.array([1, 2, 5.1, 3.4, 6])
        df_answer = mpd.DataFrame({"a": a[::-1], "b": b[::-1]})
        assert_df_equals(df_result, df_answer)

    def test_sample(self):
        df_result = df2.sample(2, seed=1)
        df_answer = mpd.DataFrame(
            {"a": np.array(["a", "a"], dtype=object), "b": np.array([2.0, 5.1])}
        )
        assert_df_equals(df_result, df_answer)

        df_result = df2.sample(frac=0.7, seed=1)
        df_answer = mpd.DataFrame(
            {
                "a": np.array(["a", "a", "b"], dtype=object),
                "b": np.array([2.0, 5.1, 6.0]),
            }
        )
        assert_df_equals(df_result, df_answer)

        with pytest.raises(TypeError):
            df2.sample(2.5)

        with pytest.raises(ValueError):
            df2.sample(frac=-2)
