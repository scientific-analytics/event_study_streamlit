__version__ = "0.1.0"

from .models import EventStudyModel, ConstantMeanModel, MarketModel, ThreeFactorModel, FourFactorModel, get_model
from .config import EventStudyConfig
from .event_study import EventStudy, run_multiple_event_studies
from .validator import validate_events_data, validate_returns_data, validate_factors_data
from .windows import define_windows, validate_window_parameters
from .volatility import cond_vol_gjrgarch, cond_vol_change
from .plot import (
    load_returns, load_factors, load_events,
    get_available_assets, get_available_events, get_available_models,
    run_event_study_for_display, plot_event_study,
    AVAILABLE_MODELS,
)

__all__ = [
    "__version__",
    "EventStudy",
    "run_multiple_event_studies",
    "EventStudyConfig",
    "EventStudyModel",
    "ConstantMeanModel",
    "MarketModel",
    "ThreeFactorModel",
    "FourFactorModel",
    "get_model",
    "validate_events_data",
    "validate_returns_data",
    "validate_factors_data",
    "define_windows",
    "validate_window_parameters",
    "cond_vol_gjrgarch",
    "cond_vol_change",
]