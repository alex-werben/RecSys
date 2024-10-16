from dataclasses import dataclass, field


@dataclass
class TrainParams:
    """Train parameters."""

    model_type: str = field(default="SVD")
    random_state: int = field(default=23)
