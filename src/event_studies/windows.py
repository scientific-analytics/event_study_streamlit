from typing import Tuple, Union, Literal
import pandas as pd
from pandas.tseries.offsets import BDay

def define_windows(event_date: pd.Timestamp, estimation_window: Union[Tuple[int, int], Literal["Hurricane season"]],
                   event_window: Tuple[int, int], available_dates: pd.DatetimeIndex) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    available_dates = pd.DatetimeIndex(sorted(available_dates)).tz_localize(None)
    event_date = pd.Timestamp(event_date).tz_localize(None)

    actual_event_dates = available_dates[available_dates >= event_date]
    actual_event_date = actual_event_dates[0]

    if estimation_window == 'Hurricane season':
        year_hurricane = event_date.year
        est_start = pd.Timestamp(year=year_hurricane - 1, month=12, day=31) # end december
        est_end = pd.Timestamp(year=year_hurricane, month=5, day=31) # end may
    else:
        #est_start = actual_event_date + pd.Timedelta(days=estimation_window[0])
        est_start = actual_event_date + BDay(estimation_window[0])
        #est_end = actual_event_date + pd.Timedelta(days=estimation_window[1])
        est_end = actual_event_date + BDay(estimation_window[1])
    #evt_start = actual_event_date + pd.Timedelta(days=event_window[0])
    evt_start = actual_event_date + BDay(event_window[0])
    #evt_end = actual_event_date + pd.Timedelta(days=event_window[1])
    evt_end = actual_event_date + BDay(event_window[1])

    est_dates = available_dates[(available_dates >= est_start) & (available_dates <= est_end)]
    evt_dates = available_dates[(available_dates >= evt_start) & (available_dates <= evt_end)]

    return est_dates, evt_dates

def validate_window_parameters(estimation_window: Tuple[int, int], event_window: Tuple[int, int]) -> None:
    if estimation_window[0] >= estimation_window[1]:
        raise ValueError("Estimation window start must be before end")
    if event_window[0] >= event_window[1]:
        raise ValueError("Event window start must be before end")
    if estimation_window[1] >= event_window[0]:
        raise ValueError("Estimation and event windows must not overlap")

