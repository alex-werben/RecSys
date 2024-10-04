from dataclasses import dataclass, field

@dataclass
class RandomSplitterParams:
    test_fold_frac: float = field(default=0.2)
    n_splits: int = field(default=1)
    random_state: int = field(default=23)
    filter_cold_users: bool = field(default=False)
    filter_cold_items: bool = field(default=False)
    filter_already_seen: bool = field(default=False)
