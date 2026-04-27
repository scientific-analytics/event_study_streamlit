from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass, field


@dataclass
class EventStudyConfig:

    model: str = "market_model"
    model_params: Dict[str, Any] = field(default_factory=dict)
    estimation_window: Tuple[int, int] = (-252, -1)
    event_window: Tuple[int, int] = (-5, 5)
    alpha: float = 0.05
    multiple_testing_correction: Optional[str] = None
    correction_params: Dict[str, Any] = field(default_factory=dict)
    min_securities: int = 1

