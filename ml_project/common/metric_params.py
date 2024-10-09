import typing as tp
from dataclasses import dataclass, field

@dataclass
class MetricParams:
    k: int = field(default=10)

@dataclass
class MetricDictParams:
    name: tp.List[str] = field(default_factory=list)
    parameters: tp.List[]