from dataclasses import dataclass, field


@dataclass
class ReadParams:
    """Read parameters."""

    sep: str = field(default=None)
    encoding: str = field(default=None)
