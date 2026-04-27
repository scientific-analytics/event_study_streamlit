from typing import List, Optional, Tuple
import pandas as pd

def validate_events_data(events: pd.DataFrame, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    if required_columns is None:
        required_columns = ['event_date', 'event_id']

    missing_cols = [col for col in required_columns if col not in events.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    if not pd.api.types.is_datetime64_any_dtype(events['event_date']):
        raise ValueError("event_date must be datetime type")

    return events

def validate_returns_data(returns: pd.DataFrame, min_securities: int = 1) -> pd.DataFrame:
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Returns must have DatetimeIndex")

    if returns.shape[1] < min_securities:
        raise ValueError(f"Need at least {min_securities} securities")

    return returns

def validate_factors_data(factors: pd.DataFrame, required_factors: Optional[List[str]] = None) -> pd.DataFrame:
    if not isinstance(factors.index, pd.DatetimeIndex):
        raise ValueError("Factors must have DatetimeIndex")

    if required_factors:
        missing = [f for f in required_factors if f not in factors.columns]
        if missing:
            raise ValueError(f"Missing factors: {missing}")

    return factors

def check_date_alignment(events: pd.DataFrame, returns: pd.DataFrame, factors: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start_date = max(returns.index.min(), factors.index.min())
    end_date = min(returns.index.max(), factors.index.max())
    return start_date, end_date