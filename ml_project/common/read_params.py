from dataclasses import dataclass, field

@dataclass
class ReadParams:
    sep: str = field(default=None)
    encoding: str = field(default=None)
