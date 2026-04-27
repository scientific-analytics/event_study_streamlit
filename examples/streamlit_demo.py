"""
Streamlit Event Study Demo
Run from the project root with:
    streamlit run examples/streamlit_demo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from event_studies.plot import (
    get_available_assets,
    get_available_events,
    get_available_models,
    load_events,
    load_factors,
    plot_event_study,
    run_event_study_for_display,
)

DATA_DIR = Path(__file__).resolve().parent / "data_input"

#
_FF_FACTORS_PATH = DATA_DIR / "F-F_Research_Data_Factors_daily.csv"
_DEFAULT_FACTORS_PATH = DATA_DIR / "factors.csv"


@st.cache_data
def _load_factors_from_path(path: str) -> pd.DataFrame:
    return load_factors(path)


@st.cache_data
def _load_events() -> pd.DataFrame:
    return load_events(DATA_DIR / "event_test.csv")


@st.cache_data
def _load_factors_from_bytes(raw_bytes: bytes) -> pd.DataFrame:
    import io
    return load_factors(io.BytesIO(raw_bytes))


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Event Study", page_icon="📈", layout="wide")
st.title("Event Study")

events_db = _load_events()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Inputs")

    # 1 · Returns upload ---------------------------------------------------------
    st.subheader("1 · Returns")
    uploaded_returns = st.file_uploader(
        "CSV file — first column = date index, remaining columns = assets",
        type="csv",
        key="returns_uploader",
    )

    returns     = None
    security_id = None

    if uploaded_returns is not None:
        try:
            returns = pd.read_csv(uploaded_returns, index_col=0, parse_dates=True)
            security_id = st.selectbox(
                "Asset",
                options=get_available_assets(returns),
            )
        except Exception as exc:
            st.error(f"Could not parse returns file: {exc}")
    else:
        st.caption("No file uploaded yet.")

    # 2 · Factors upload (optional) ---------------------------------------------
    st.subheader("2 · Factors")
    uploaded_factors = st.file_uploader(
        "Optional: upload your own factors CSV (standard or Fama-French daily format). "
        "Leave empty to use the bundled Fama-French daily factors.",
        type="csv",
        key="factors_uploader",
    )

    if uploaded_factors is not None:
        try:
            factors = _load_factors_from_bytes(uploaded_factors.read())
            st.caption(
                f"Uploaded factors: {factors.index.min().date()} – {factors.index.max().date()}"
            )
        except Exception as exc:
            st.error(f"Could not parse factors file: {exc}")
            factors = None
    else:
        default_path = _FF_FACTORS_PATH if _FF_FACTORS_PATH.exists() else _DEFAULT_FACTORS_PATH
        factors = _load_factors_from_path(str(default_path))
        st.caption(
            f"Using bundled factors: {factors.index.min().date()} – {factors.index.max().date()}"
        )

    # 3 · Model ------------------------------------------------------------------
    st.subheader("3 · Factor model")
    model = st.selectbox("Model", options=get_available_models())

    # 4 · Event ------------------------------------------------------------------
    st.subheader("4 · Event")
    event_source = st.radio("Source", ["Predefined", "Custom date"], horizontal=True)

    if event_source == "Predefined":
        event_options = get_available_events(events_db)
        labels = [
            f"{e['event_name']}  ({pd.Timestamp(e['event_date']).date()})"
            for e in event_options
        ]
        idx = st.selectbox(
            "Event", range(len(labels)), format_func=lambda i: labels[i]
        )
        sel           = event_options[idx]
        event_id      = sel["event_id"]
        events_to_use = events_db

    else:
        # Bound the date picker to the data range so the default is always valid
        if returns is not None:
            min_d = returns.index.min().date()
            max_d = returns.index.max().date()
        elif factors is not None:
            min_d = factors.index.min().date()
            max_d = factors.index.max().date()
        else:
            min_d = events_db["event_date"].min().date()
            max_d = events_db["event_date"].max().date()

        import datetime
        default_d = min_d + (max_d - min_d) // 2

        custom_date = st.date_input(
            "Event date",
            value=default_d,
            min_value=min_d,
            max_value=max_d,
        )
        custom_name = st.text_input("Event name", value="Custom Event")
        event_id    = 9999
        events_to_use = pd.DataFrame({
            "event_date": [pd.Timestamp(custom_date)],
            "event_id":   [9999],
            "event_name": [custom_name],
            "event_type": ["custom"],
        })

    # 5 · Windows ----------------------------------------------------------------
    st.subheader("5 · Windows")
    c1, c2 = st.columns(2)
    est_start = c1.number_input("Est. start", value=-252, step=1)
    est_end   = c2.number_input("Est. end",   value=-6,   step=1)
    evt_start = c1.number_input("Evt. start", value=-5,   step=1)
    evt_end   = c2.number_input("Evt. end",   value=5,    step=1)

    # 6 · Run button -------------------------------------------------------------
    st.divider()
    run_clicked = st.button(
        "Run Analysis",
        type="primary",
        disabled=(returns is None or security_id is None or factors is None),
    )

# ── Main area ─────────────────────────────────────────────────────────────────
if returns is None:
    st.info("Upload a returns CSV on the left to get started.")
    st.stop()

if not run_clicked:
    st.info("Configure your inputs on the left, then click **Run Analysis**.")
    st.stop()

# Validate windows
errors = []
if est_start >= est_end:
    errors.append("Estimation window: start must be strictly before end.")
if evt_start >= evt_end:
    errors.append("Event window: start must be strictly before end.")
for msg in errors:
    st.error(msg)
if errors:
    st.stop()

# Warn when factors don't fully cover the selected event date
if factors is not None and event_source == "Custom date":
    evt_ts = pd.Timestamp(custom_date)
    if evt_ts < factors.index.min() or evt_ts > factors.index.max():
        st.warning(
            f"Event date {custom_date} is outside the factors date range "
            f"({factors.index.min().date()} – {factors.index.max().date()}). "
            "Results may be NaN."
        )

# Run analysis
with st.spinner("Running event study and GJR-GARCH…"):
    try:
        data = run_event_study_for_display(
            security_id=security_id,
            event_id=event_id,
            event_window=(int(evt_start), int(evt_end)),
            model=model,
            returns=returns,
            factors=factors,
            events=events_to_use,
            estimation_window=(int(est_start), int(est_end)),
        )
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.stop()

if data is None:
    st.warning(
        "No results — not enough data for the selected inputs. "
        "Try a shorter estimation window or a different event."
    )
    st.stop()

# ── Metrics row ───────────────────────────────────────────────────────────────
r = data["results_row"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Final CAR",   f"{r['final_car'] * 100:.2f}%")
m2.metric("t-statistic", f"{r['final_car_t_stat']:.2f}")
m3.metric("p-value",     f"{r['final_car_p_value']:.3f}")
m4.metric("Significant", "Yes ✓" if r["is_significant"] else "No")

# ── Two-panel plot ────────────────────────────────────────────────────────────
fig = plot_event_study(data)
st.pyplot(fig)
plt.close(fig)
