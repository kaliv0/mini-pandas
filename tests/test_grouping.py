import numpy as np

import mini_pandas as mpd
from tests import assert_df_equals

a = np.array(["b", "a", "a", "a", "b", "a", "a", "b"])
b = np.array(["B", "A", "A", "A", "B", "B", "B", "A"])
c = np.array([1, 2, 3, 4, 5, 6, 7, 8])
df = mpd.DataFrame({"a": a, "b": b, "c": c})


class TestGrouping:
    def test_value_counts(self):
        df_temp = mpd.DataFrame(
            {
                "state": np.array(
                    [
                        "texas",
                        "texas",
                        "texas",
                        "florida",
                        "florida",
                        "florida",
                        "florida",
                        "ohio",
                    ]
                ),
                "fruit": np.array(["a", "a", "a", "a", "b", "b", "b", "a"]),
            }
        )
        df_results = df_temp.value_counts()
        df_answer = mpd.DataFrame(
            {
                "state": np.array(["florida", "texas", "ohio"], dtype=object),
                "count": np.array([4, 3, 1]),
            }
        )
        assert_df_equals(df_results[0], df_answer)

        df_answer = mpd.DataFrame(
            {"fruit": np.array(["a", "b"], dtype=object), "count": np.array([5, 3])}
        )
        assert_df_equals(df_results[1], df_answer)

    def test_value_counts_normalize(self):
        df_temp = mpd.DataFrame(
            {
                "state": np.array(
                    [
                        "texas",
                        "texas",
                        "texas",
                        "florida",
                        "florida",
                        "florida",
                        "florida",
                        "ohio",
                    ]
                ),
                "fruit": np.array(["a", "a", "a", "a", "b", "b", "b", "a"]),
            }
        )
        df_results = df_temp.value_counts(normalize=True)
        df_answer = mpd.DataFrame(
            {
                "state": np.array(["florida", "texas", "ohio"], dtype=object),
                "count": np.array([0.5, 0.375, 0.125]),
            }
        )
        assert_df_equals(df_results[0], df_answer)

        df_answer = mpd.DataFrame(
            {
                "fruit": np.array(["a", "b"], dtype=object),
                "count": np.array([0.625, 0.375]),
            }
        )
        assert_df_equals(df_results[1], df_answer)

    def test_pivot_table_rows_or_cols(self):
        df_result = df.pivot_table(rows="a")
        df_answer = mpd.DataFrame(
            {"a": np.array(["a", "b"], dtype=object), "size": np.array([5, 3])}
        )
        assert_df_equals(df_result, df_answer)

        df_result = df.pivot_table(rows="a", values="c", agg_func="sum")
        df_answer = mpd.DataFrame(
            {"a": np.array(["a", "b"], dtype=object), "sum": np.array([22, 14])}
        )
        assert_df_equals(df_result, df_answer)

        df_result = df.pivot_table(columns="b")
        df_answer = mpd.DataFrame({"A": np.array([4]), "B": np.array([4])})
        assert_df_equals(df_result, df_answer)

        df_result = df.pivot_table(columns="a", values="c", agg_func="sum")
        df_answer = mpd.DataFrame({"a": np.array([22]), "b": np.array([14])})
        assert_df_equals(df_result, df_answer)

    def test_pivot_table_both(self):
        df_result = df.pivot_table(rows="a", columns="b", values="c", agg_func="sum")
        df_answer = mpd.DataFrame(
            {
                "a": np.array(["a", "b"], dtype=object),
                "A": np.array([9.0, 8.0]),
                "B": np.array([13.0, 6.0]),
            }
        )
        assert_df_equals(df_result, df_answer)
