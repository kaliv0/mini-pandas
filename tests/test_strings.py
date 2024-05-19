import numpy as np
import pytest

import mini_pandas as mpd
from tests import assert_df_equals

df_string = mpd.DataFrame(
    {
        "movie": np.array(["field of dreams", "star wars"], dtype="O"),
        "num": np.array(["5.1", "6"], dtype="O"),
    }
)


class TestStrings:
    def test_capitalize(self):
        result = df_string.str.capitalize("movie")
        movie = np.array(["Field of dreams", "Star wars"], dtype="O")
        answer = mpd.DataFrame({"movie": movie})
        assert_df_equals(result, answer)

    def test_center(self):
        result = df_string.str.center("movie", 20, "-")
        movie = np.array(["--field of dreams---", "-----star wars------"], dtype="O")
        answer = mpd.DataFrame({"movie": movie})
        assert_df_equals(result, answer)

    def test_count(self):
        result = df_string.str.count("movie", "e")
        movie = np.array([2, 0])
        answer = mpd.DataFrame({"movie": movie})
        assert_df_equals(result, answer)

    def test_startswith(self):
        result = df_string.str.startswith("movie", "field")
        movie = np.array([True, False])
        answer = mpd.DataFrame({"movie": movie})
        assert_df_equals(result, answer)

    def test_endswith(self):
        result = df_string.str.endswith("movie", "s")
        movie = np.array([True, True])
        answer = mpd.DataFrame({"movie": movie})
        assert_df_equals(result, answer)

    def test_find(self):
        result = df_string.str.find("movie", "ar")
        movie = np.array([-1, 2])
        answer = mpd.DataFrame({"movie": movie})
        assert_df_equals(result, answer)

    def test_len(self):
        result = df_string.str.len("movie")
        movie = np.array([15, 9])
        answer = mpd.DataFrame({"movie": movie})
        assert_df_equals(result, answer)

    def test_get(self):
        result = df_string.str.get("movie", 5)
        movie = np.array([" ", "w"], dtype="O")
        answer = mpd.DataFrame({"movie": movie})
        assert_df_equals(result, answer)

    def test_index(self):
        with pytest.raises(ValueError):
            df_string.str.index("movie", "z")

    def test_isalnum(self):
        result = df_string.str.isalnum("num")
        num = np.array([False, True])
        answer = mpd.DataFrame({"num": num})
        assert_df_equals(result, answer)

    def test_isalpha(self):
        result = df_string.str.isalpha("num")
        num = np.array([False, False])
        answer = mpd.DataFrame({"num": num})
        assert_df_equals(result, answer)

    def test_isdecimal(self):
        result = df_string.str.isdecimal("num")
        num = np.array([False, True])
        answer = mpd.DataFrame({"num": num})
        assert_df_equals(result, answer)

    def test_isnumeric(self):
        result = df_string.str.isnumeric("num")
        num = np.array([False, True])
        answer = mpd.DataFrame({"num": num})
        assert_df_equals(result, answer)

    def test_islower(self):
        result = df_string.str.islower("movie")
        movie = np.array([True, True])
        answer = mpd.DataFrame({"movie": movie})
        assert_df_equals(result, answer)

    def test_isupper(self):
        result = df_string.str.isupper("movie")
        movie = np.array([False, False])
        answer = mpd.DataFrame({"movie": movie})
        assert_df_equals(result, answer)

    def test_isspace(self):
        result = df_string.str.isspace("num")
        num = np.array([False, False])
        answer = mpd.DataFrame({"num": num})
        assert_df_equals(result, answer)

    def test_istitle(self):
        result = df_string.str.istitle("num")
        num = np.array([False, False])
        answer = mpd.DataFrame({"num": num})
        assert_df_equals(result, answer)

    def test_lstrip(self):
        result = df_string.str.lstrip("movie", "fies")
        movie = np.array(["ld of dreams", "tar wars"], dtype="O")
        answer = mpd.DataFrame({"movie": movie})
        assert_df_equals(result, answer)

    def test_rstrip(self):
        result = df_string.str.rstrip("movie", "s")
        movie = np.array(["field of dream", "star war"], dtype="O")
        answer = mpd.DataFrame({"movie": movie})
        assert_df_equals(result, answer)

    def test_strip(self):
        result = df_string.str.strip("movie", "fs")
        movie = np.array(["ield of dream", "tar war"], dtype="O")
        answer = mpd.DataFrame({"movie": movie})
        assert_df_equals(result, answer)

    def test_replace(self):
        result = df_string.str.replace("movie", "s", "Z")
        movie = np.array(["field of dreamZ", "Ztar warZ"], dtype="O")
        answer = mpd.DataFrame({"movie": movie})
        assert_df_equals(result, answer)

    def test_swapcase(self):
        result = df_string.str.swapcase("movie")
        movie = np.array(["FIELD OF DREAMS", "STAR WARS"], dtype="O")
        answer = mpd.DataFrame({"movie": movie})
        assert_df_equals(result, answer)

    def test_title(self):
        result = df_string.str.title("movie")
        movie = np.array(["Field Of Dreams", "Star Wars"], dtype="O")
        answer = mpd.DataFrame({"movie": movie})
        assert_df_equals(result, answer)

    def test_upper(self):
        result = df_string.str.upper("movie")
        movie = np.array(["FIELD OF DREAMS", "STAR WARS"], dtype="O")
        answer = mpd.DataFrame({"movie": movie})
        assert_df_equals(result, answer)

    def test_zfill(self):
        result = df_string.str.zfill("movie", 16)
        movie = np.array(["0field of dreams", "0000000star wars"], dtype="O")
        answer = mpd.DataFrame({"movie": movie})
        assert_df_equals(result, answer)
