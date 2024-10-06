from dataclasses import dataclass, field

@dataclass
class PredictParams:
    filter_viewed: bool = field(default=False)
    k: int = field(default=10)
    add_rank_col: bool = field(default=True)
