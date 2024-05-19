import numpy as np

import mini_pandas as mpd
from tests import assert_df_equals

a1 = np.array(["a", "b", "c"])
b1 = np.array([11, 5, 8])
c1 = np.array([3.4, np.nan, 5.1])
df1 = mpd.DataFrame({"a": a1, "b": b1, "c": c1})

a2 = np.array([True, False])
b2 = np.array([True, True])
c2 = np.array([False, True])
df2 = mpd.DataFrame({"a": a2, "b": b2, "c": c2})


class TestAggregation:
    def test_min(self):
        df_result = df1.min()
        df_answer = mpd.DataFrame(
            {
                "a": np.array(["a"], dtype="O"),
                "b": np.array([5]),
                "c": np.array([np.nan]),
            }
        )
        assert_df_equals(df_result, df_answer)

    def test_max(self):
        df_result = df1.max()
        df_answer = mpd.DataFrame(
            {
                "a": np.array(["c"], dtype="O"),
                "b": np.array([11]),
                "c": np.array([np.nan]),
            }
        )
        assert_df_equals(df_result, df_answer)

    def test_mean(self):
        df_result = df1.mean()
        df_answer = mpd.DataFrame({"b": np.array([8.0]), "c": np.array([np.nan])})
        assert_df_equals(df_result, df_answer)

    def test_median(self):
        df_result = df1.median()
        df_answer = mpd.DataFrame({"b": np.array([8]), "c": np.array([np.nan])})
        assert_df_equals(df_result, df_answer)

    def test_sum(self):
        df_result = df1.sum()
        df_answer = mpd.DataFrame(
            {
                "a": np.array(["abc"], dtype="O"),
                "b": np.array([24]),
                "c": np.array([np.nan]),
            }
        )
        assert_df_equals(df_result, df_answer)

    def test_var(self):
        df_result = df1.var()
        df_answer = mpd.DataFrame({"b": np.array([b1.var()]), "c": np.array([np.nan])})
        assert_df_equals(df_result, df_answer)

    def test_std(self):
        df_result = df1.std()
        df_answer = mpd.DataFrame({"b": np.array([b1.std()]), "c": np.array([np.nan])})
        assert_df_equals(df_result, df_answer)

    def test_all(self):
        df_result = df2.all()
        df_answer = mpd.DataFrame(
            {"a": np.array([False]), "b": np.array([True]), "c": np.array([False])}
        )
        assert_df_equals(df_result, df_answer)

    def test_any(self):
        df_result = df2.any()
        df_answer = mpd.DataFrame(
            {"a": np.array([True]), "b": np.array([True]), "c": np.array([True])}
        )
        assert_df_equals(df_result, df_answer)

    def test_argmax(self):
        df_result = df1.argmax()
        df_answer = mpd.DataFrame(
            {"a": np.array([2]), "b": np.array([0]), "c": np.array([1])}
        )
        assert_df_equals(df_result, df_answer)

    def test_argmin(self):
        df_result = df1.argmin()
        df_answer = mpd.DataFrame(
            {"a": np.array([0]), "b": np.array([1]), "c": np.array([1])}
        )
        assert_df_equals(df_result, df_answer)
