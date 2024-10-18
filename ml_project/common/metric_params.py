from typing import List
from dataclasses import dataclass, field


@dataclass
class Params:
    """Params of top K metrics."""
    k: int = field(default=10)


@dataclass
class MetricParams:
    """Metric params."""

    names: List[str] = field(default_factory="Recall")
    params: List[Params] = field(default_factory=Params())
