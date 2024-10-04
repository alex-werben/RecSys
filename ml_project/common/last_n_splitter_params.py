from dataclasses import dataclass, field

@dataclass
class LastNSplitterParams:
    n: int = field(default=0.25)
    random_state: int = field(default=23)
    shuffle: bool = field(default=False)