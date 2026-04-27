from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from .config import EventStudyConfig
from .event_study import EventStudy
from .volatility import cond_vol_gjrgarch, _MODEL_FACTORS

AVAILABLE_MODELS = list(_MODEL_FACTORS.keys())  # exported for UI widgets


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_returns(path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="date", parse_dates=True)


def load_factors(path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    if 'Mkt-RF' in df.columns:
        # Fama-French daily format: YYYYMMDD index, % scale, trailing copyright row
        df.index = pd.to_datetime(df.index, format='%Y%m%d', errors='coerce')
        df = df[df.index.notna()].dropna(subset=['Mkt-RF', 'SMB', 'HML'])
        df = df.rename(columns={'Mkt-RF': 'Mkt'}).drop(columns=['RF'], errors='ignore')
        df[['Mkt', 'SMB', 'HML']] = df[['Mkt', 'SMB', 'HML']] / 100
    else:
        df.index = pd.to_datetime(df.index)
    df.index.name = 'date'
    return df


def _infer_dayfirst(values) -> bool:
    """Return True if any leading date segment exceeds 12, implying DD/MM/YYYY."""
    series = pd.Series(values).dropna().astype(str)
    first_segments = series.str.split(r"[/\-]").str[0]
    try:
        return bool((first_segments.astype(int) > 12).any())
    except (ValueError, TypeError):
        return False


def load_returns(path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="date")
    df.index = pd.to_datetime(df.index, dayfirst=_infer_dayfirst(df.index))
    df.index.name = "date"
    return df


def load_events(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["event_date"] = pd.to_datetime(
        df["event_date"], dayfirst=_infer_dayfirst(df["event_date"])
    )
    return df


# ---------------------------------------------------------------------------
# Selection helpers  (feed directly into a UI widget)
# ---------------------------------------------------------------------------

def get_available_assets(returns: pd.DataFrame) -> list[str]:
    return list(returns.columns)


def get_available_events(events: pd.DataFrame) -> list[dict]:
    """Returns list of {event_id, event_name, event_date} for UI display."""
    return events[["event_id", "event_name", "event_date"]].to_dict("records")


def get_available_models() -> list[str]:
    return AVAILABLE_MODELS


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_event_study_for_display(
    security_id: str,
    event_id,
    event_window: Tuple[int, int],
    model: str,
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    events: pd.DataFrame,
    estimation_window: Tuple[int, int] = (-252, -1),
    alpha: float = 0.05,
) -> Optional[dict]:
    """
    Runs both the OLS event study and the GJR-GARCH conditional volatility
    for a single (security, event) pair.

    Returns a dict with:
        cum_car        — time-series CAR DataFrame (one row per event-window day)
        vol_df         — full-sample conditional vol DataFrame (tot_vol, resid_vol)
        results_row    — scalar summary for this security/event (dict)
        event_name     — str
        event_date     — pd.Timestamp
        config         — EventStudyConfig used
        security_id    — str
    Returns None if the event study produces no results (insufficient data).
    """
    config = EventStudyConfig(
        model=model,
        estimation_window=estimation_window,
        event_window=event_window,
        alpha=alpha,
    )

    single_event = events[events["event_id"] == event_id].copy().dropna()
    if len(single_event) == 0:
        return None

    results_df, cum_car_all = EventStudy(config).run(
        single_event, returns[[security_id]], factors
    )

    if len(results_df) == 0:
        return None

    cum_car = (
        cum_car_all[cum_car_all["security_id"] == security_id]
        .sort_values("effective_int")
        .reset_index(drop=True)
    )

    vol_df = cond_vol_gjrgarch(returns[security_id], factors, config)

    event_row = events[events["event_id"] == event_id].iloc[0]
    results_row = results_df[results_df["security_id"] == security_id].iloc[0].to_dict()

    return {
        "cum_car": cum_car,
        "vol_df": vol_df,
        "results_row": results_row,
        "event_name": str(event_row.get("event_name", f"Event {event_id}")),
        "event_date": pd.Timestamp(event_row["event_date"]),
        "config": config,
        "security_id": security_id,
        "dof": results_row["degrees_of_freedom"],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_event_study(
    data: dict,
    figsize: Tuple[int, int] = (10, 8),
    confidence_level: float = 0.95,
) -> plt.Figure:
    """
    Two-panel figure sharing the x-axis (business days from event):
      Top   — Cumulative Abnormal Return with confidence bands
      Bottom — Total and residual conditional volatility (GJR-GARCH)

    Parameters
    ----------
    data              Output of run_event_study_for_display().
    figsize           Matplotlib figure size.
    confidence_level  Width of the confidence band (default 95%).
    """
    cum_car = data["cum_car"]
    vol_df = data["vol_df"]
    event_name = data["event_name"]
    security_id = data["security_id"]

    x = cum_car["effective_int"].values
    car = cum_car["cum_car"].values.astype(float)
    var = np.maximum(cum_car["cum_car_variances"].values.astype(float), 0)
    se = np.sqrt(var)
    dof = data.get("dof", 100)  # fallback to large dof (≈ normal) if absent
    t_crit = stats.t.ppf(1 - (1 - confidence_level) / 2, df=dof)

    # align conditional vol to the event-window dates; normalize units to avoid
    # reindex mismatches when returns were uploaded with a different datetime precision
    effective_dates = pd.to_datetime(cum_car["effective_date"].values).normalize()
    vol_idx = vol_df.copy()
    vol_idx.index = pd.to_datetime(vol_idx.index).normalize()
    vol_event = vol_idx.reindex(effective_dates).ffill(limit=2)

    tot = vol_event["tot_vol"].values.astype(float)
    resid = vol_event["resid_vol"].values.astype(float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # --- top panel: CAR + confidence band ---
    ax1.plot(x, car*100, color="steelblue", linewidth=1.5, label="CAR (%)")

    lower = (car - t_crit * se) * 100
    upper = (car + t_crit * se) * 100
    # Only mask where car_variances itself is NaN (estimator fix makes this rare).
    # Do NOT use tot's NaN pattern: GARCH can lack estimates for a day even when the
    # return and AR are perfectly valid, which would create spurious holes in the band.
    valid = ~(np.isnan(lower) | np.isnan(upper))
    # draw each contiguous valid segment separately so fill_between never bridges gaps
    _first_fill = True
    for _seg in np.ma.clump_unmasked(np.ma.array(lower, mask=~valid)):
        ax1.fill_between(
            x[_seg], lower[_seg], upper[_seg],
            alpha=0.15, color="steelblue",
            label=(f"{confidence_level:.0%} CI" if _first_fill else "_nolegend_"),
        )
        _first_fill = False
    if _first_fill:  # nothing valid — add a dummy entry to keep the legend consistent
        ax1.fill_between([], [], [], alpha=0.15, color="steelblue", label=f"{confidence_level:.0%} CI")

    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.axvline(0, color="crimson", linestyle="--", linewidth=0.8, alpha=0.7, label="Event")
    ax1.set_ylabel("Cumulative Abnormal Return (%)")
    ax1.set_title(f"{security_id} — {event_name}")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- bottom panel: conditional volatility ---

    ax2.plot(x, tot*np.sqrt(252)*100, color="darkorange", linewidth=1.5, label="Total vol (GJR-GARCH), % Ann.")
    if not np.all(np.isnan(resid)):
        ax2.plot(
            x,
            resid*np.sqrt(252)*100,
            color="purple",
            linewidth=1.5,
            linestyle="--",
            label="Residual vol (GJR-GARCH), % Ann.",
        )
    ax2.axvline(0, color="crimson", linestyle="--", linewidth=0.8, alpha=0.7, label="Event")
    ax2.set_ylabel("Conditional Volatility (% ann.)")
    ax2.set_xlabel("Business days from event")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

