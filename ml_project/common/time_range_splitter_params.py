from dataclasses import dataclass, field

@dataclass
class TimeRangeSplitterParams:
    test_size: str = field(default="30D")
    n_splits: int = field(default=1)
    filter_cold_users: bool = field(default=False)
    filter_cold_items: bool = field(default=False)
    filter_already_seen: bool = field(default=False)