import numpy as np

import mini_pandas as mpd
from tests import assert_df_equals

a = np.array([11, 5])
b = np.array([3.4, 5.1])
df = mpd.DataFrame({"a": a, "b": b})


class TestOperators:
    def test_add(self):
        df_result = df + 3
        df_answer = mpd.DataFrame({"a": a + 3, "b": b + 3})
        assert_df_equals(df_result, df_answer)

        df_result = 3 + df
        assert_df_equals(df_result, df_answer)

    def test_sub(self):
        df_result = df - 3
        df_answer = mpd.DataFrame({"a": a - 3, "b": b - 3})
        assert_df_equals(df_result, df_answer)

        df_result = 3 - df
        df_answer = mpd.DataFrame({"a": 3 - a, "b": 3 - b})
        assert_df_equals(df_result, df_answer)

    def test_mul(self):
        df_result = df * 3
        df_answer = mpd.DataFrame({"a": a * 3, "b": b * 3})
        assert_df_equals(df_result, df_answer)

        df_result = 3 * df
        assert_df_equals(df_result, df_answer)

    def test_truediv(self):
        df_result = df / 3
        df_answer = mpd.DataFrame({"a": a / 3, "b": b / 3})
        assert_df_equals(df_result, df_answer)

        df_result = 3 / df
        df_answer = mpd.DataFrame({"a": 3 / a, "b": 3 / b})
        assert_df_equals(df_result, df_answer)

    def test_floordiv(self):
        df_result = df // 3
        df_answer = mpd.DataFrame({"a": a // 3, "b": b // 3})
        assert_df_equals(df_result, df_answer)

        df_result = 3 // df
        df_answer = mpd.DataFrame({"a": 3 // a, "b": 3 // b})
        assert_df_equals(df_result, df_answer)

    def test_pow(self):
        df_result = df**3
        df_answer = mpd.DataFrame({"a": a**3, "b": b**3})
        assert_df_equals(df_result, df_answer)

        df_result = 2**df
        df_answer = mpd.DataFrame({"a": 2**a, "b": 2**b})
        assert_df_equals(df_result, df_answer)

    def test_gt_lt(self):
        df_result = df > 3
        df_answer = mpd.DataFrame({"a": a > 3, "b": b > 3})
        assert_df_equals(df_result, df_answer)

        df_result = df < 2
        df_answer = mpd.DataFrame({"a": a < 2, "b": b < 2})
        assert_df_equals(df_result, df_answer)

    def test_ge_le(self):
        df_result = df >= 3
        df_answer = mpd.DataFrame({"a": a >= 3, "b": b >= 3})
        assert_df_equals(df_result, df_answer)

        df_result = df < 2
        df_answer = mpd.DataFrame({"a": a <= 2, "b": b <= 2})
        assert_df_equals(df_result, df_answer)

    def test_eq_ne(self):
        df_result = df == 3
        df_answer = mpd.DataFrame({"a": a == 3, "b": b == 3})
        assert_df_equals(df_result, df_answer)

        df_result = df != 2
        df_answer = mpd.DataFrame({"a": a != 2, "b": b != 2})
        assert_df_equals(df_result, df_answer)
