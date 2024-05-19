from dataclasses import dataclass
from enum import StrEnum


class PivotType(StrEnum):
    ROWS = "rows"
    COLUMNS = "columns"
    ALL = "all"


class DFType(StrEnum):
    U_STRING = "U"  # Unicode str
    STRING = "O"
    BOOL = "b"
    FLOAT = "f"
    INT = "i"


@dataclass(frozen=True)
class Tokens:
    NEW_LINE = "\n"
    DELIMITER = ","


@dataclass(frozen=True)
class BuildingParams:
    DEFAULT_COL_COUNT = 20
    DEFAULT_NUM_HEAD = 10
    DEFAULT_NUM_TAIL = 10


AGG_METHOD_NAMES = [
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

DEFAULT_AGG_FUNC = "size"
