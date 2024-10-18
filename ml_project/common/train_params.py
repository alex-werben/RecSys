from dataclasses import dataclass, field

import implicit
import typing as tp


@dataclass
class SVDParams:
    """SVD parameters."""

    factors: int = field(default=10),
    tol: float = field(default=0),
    maxiter: tp.Optional[int] = field(default=None),
    random_state: tp.Optional[int] = field(default=None),
    verbose: int = field(default=0)


@dataclass
class ALSParams:
    """ALS parameters."""

    factors: int = field(default=100),
    regularization: float = field(default=0.01),
    alpha: float = field(default=1.0),
    use_native: bool = field(default=True),
    use_cg: bool = field(default=True),
    use_gpu: bool = field(default=implicit.gpu.HAS_CUDA),
    iterations: int = field(default=15),
    calculate_training_loss: bool = field(default=False),
    num_threads: int = field(default=0),
    random_state: int = field(default=None)


ModelParams = tp.Union[SVDParams, ALSParams]


@dataclass
class TrainParams:
    """Train params."""

    model_type: str = field(default="SVD")
    model_params: ModelParams = field(default=SVDParams)
