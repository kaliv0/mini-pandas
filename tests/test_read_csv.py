import numpy as np

import mini_pandas as mpd
from tests import assert_df_equals

df_emp = mpd.read_csv("tests/resources/employee.csv")


class TestReadCSV:
    def test_columns(self):
        result = df_emp.columns
        answer = ["dept", "race", "gender", "salary"]
        assert result == answer

    def test_data_types(self):
        df_result = df_emp.dtypes
        cols = np.array(["dept", "race", "gender", "salary"], dtype="O")
        dtypes = np.array(["string", "string", "string", "int"], dtype="O")
        df_answer = mpd.DataFrame({"Column Name": cols, "Data Type": dtypes})
        assert_df_equals(df_result, df_answer)

    def test_sum(self):
        result = df_emp["salary"].sum()
        answer = 86387875
        assert result == answer

    def test_head(self):
        data = {
            "dept": np.array(
                [
                    "Houston Police Department-HPD",
                    "Houston Fire Department (HFD)",
                    "Houston Police Department-HPD",
                    "Public Works & Engineering-PWE",
                    "Houston Airport System (HAS)",
                ],
                dtype="O",
            ),
            "race": np.array(["White", "White", "Black", "Asian", "White"], dtype="O"),
            "gender": np.array(["Male", "Male", "Male", "Male", "Male"], dtype="O"),
            "salary": np.array([45279, 63166, 66614, 71680, 42390]),
        }
        result = df_emp.head()
        answer = mpd.DataFrame(data)
        assert_df_equals(result, answer)
