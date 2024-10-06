from dataclasses import dataclass, field

@dataclass
class ReadParams:
    sep: str = field(default=None)
    encoding_escape: str = field(default=None)
