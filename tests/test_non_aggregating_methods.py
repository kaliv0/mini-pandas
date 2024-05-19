import numpy as np
from numpy.testing import assert_array_equal

import mini_pandas as mpd
from tests import assert_df_equals

a1 = np.array(["a", None, "c"])
b1 = np.array([11, 5, 8])
c1 = np.array([3.4, np.nan, 5.1])
df1 = mpd.DataFrame({"a": a1, "b": b1, "c": c1})

a2 = np.array(["a", "a", "c"], dtype="O")
b2 = np.array([11, 5, 5])
c2 = np.array([3.4, np.nan, 3.4])
df2 = mpd.DataFrame({"a": a2, "b": b2, "c": c2})

df3 = mpd.DataFrame({"a": np.array([10, 8, 9]), "b": np.array([5, 7, 3])})


class TestNonAggregatingMethods:
    def test_isna(self):
        df_result = df1.isna()
        df_answer = mpd.DataFrame(
            {
                "a": np.array([False, True, False]),
                "b": np.array([False, False, False]),
                "c": np.array([False, True, False]),
            }
        )
        assert_df_equals(df_result, df_answer)

    def test_count(self):
        df_result = df1.count()
        df_answer = mpd.DataFrame(
            {"a": np.array([2]), "b": np.array([3]), "c": np.array([2])}
        )
        assert_df_equals(df_result, df_answer)

    def test_unique(self):
        df_result = df2.unique()
        assert_array_equal(df_result[0].values[:, 0], np.unique(a2))
        assert_array_equal(df_result[1].values[:, 0], np.unique(b2))
        assert_array_equal(df_result[2].values[:, 0], np.unique(c2))

    def test_nunique(self):
        df_result = df2.nunique()
        df_answer = mpd.DataFrame(
            {"a": np.array([2]), "b": np.array([2]), "c": np.array([2])}
        )
        assert_df_equals(df_result, df_answer)

    def test_rename(self):
        df_result = df2.rename({"a": "A", "c": "C"})
        df_answer = mpd.DataFrame({"A": a2, "b": b2, "C": c2})
        assert_df_equals(df_result, df_answer)

    def test_drop(self):
        df_result = df2.drop(["a", "b"])
        df_answer = mpd.DataFrame({"c": c2})
        assert_df_equals(df_result, df_answer)

    def test_diff(self):
        df_result = df3.diff(1)
        df_answer = mpd.DataFrame(
            {"a": np.array([np.nan, -2, 1]), "b": np.array([np.nan, 2, -4])}
        )
        assert_df_equals(df_result, df_answer)

    def test_pct_change(self):
        df_result = df3.pct_change(1)
        df_answer = mpd.DataFrame(
            {
                "a": np.array([np.nan, -2 / 10, 1 / 8]),
                "b": np.array([np.nan, 2 / 5, -4 / 7]),
            }
        )
        assert_df_equals(df_result, df_answer)
