from typing import Union
from dataclasses import dataclass, field


@dataclass
class TimeRangeSplitterParams:
    """Time range splitter parameters."""

    test_size: str = field(default="30D")
    n_splits: int = field(default=1)
    filter_cold_users: bool = field(default=False)
    filter_cold_items: bool = field(default=False)
    filter_already_seen: bool = field(default=False)


@dataclass
class RandomSplitterParams:
    """Random splitter parameters."""

    test_fold_frac: float = field(default=0.2)
    n_splits: int = field(default=1)
    random_state: int = field(default=23)
    filter_cold_users: bool = field(default=False)
    filter_cold_items: bool = field(default=False)
    filter_already_seen: bool = field(default=False)


@dataclass
class LastNSplitterParams:
    """Last N splitter parameters."""

    n: int = field(default=0.25)
    random_state: int = field(default=23)
    shuffle: bool = field(default=False)


SplitterParams = Union[LastNSplitterParams, RandomSplitterParams, TimeRangeSplitterParams]
