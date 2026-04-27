"""
Standalone demo for the event-study display module.
Run this file directly from PyCharm (or any terminal) after installing the
package into the active interpreter:

    pip install -e .

Then adjust the five USER SELECTIONS below and run.
"""

from pathlib import Path
import matplotlib.pyplot as plt

from event_studies.plot import (
    load_returns,
    load_factors,
    load_events,
    get_available_assets,
    get_available_events,
    get_available_models,
    run_event_study_for_display,
    plot_event_study,
)

DATA_DIR = Path(__file__).parent / "data_input"

returns = load_returns(DATA_DIR / "test_returns.csv")
factors = load_factors(DATA_DIR / "factors.csv")
events  = load_events(DATA_DIR / "event_test.csv")

print("Available assets :", get_available_assets(returns))
print("Available models :", get_available_models())
print("Available events :")
for ev in get_available_events(events):
    print(f"  id={ev['event_id']}  {ev['event_name']}  ({ev['event_date'].date()})")

# ── USER SELECTIONS ──────────────────────────────────────────────────────────
security_id       = "18671_UXN-2360_2587814"
event_id          = 1
event_window      = (-5, 20)
model             = "market_model"
estimation_window = (-60, -6)
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nRunning: {security_id} | event {event_id} | model={model} | window={event_window}")

data = run_event_study_for_display(
    security_id=security_id,
    event_id=event_id,
    event_window=event_window,
    model=model,
    returns=returns,
    factors=factors,
    events=events,
    estimation_window=estimation_window,
)

if data is None:
    print("No results — insufficient data for the selected inputs.")
else:
    r = data["results_row"]
    print(f"Final CAR : {r['final_car']:.4f}  (t={r['final_car_t_stat']:.2f},  p={r['final_car_p_value']:.3f})")
    fig = plot_event_study(data)
    plt.show()
