# Event Studies

Minimal Python library for conducting event studies in financial markets. Focused on stock-level analysis with clean DataFrame output.

## Installation

```bash
pip install pandas numpy scipy
```

## Quick Start

```python
import pandas as pd
from event_studies import EventStudy, EventStudyConfig

# Prepare data (user responsibility)
events = pd.DataFrame({
    'event_date': pd.to_datetime(['2023-01-15', '2023-02-20']),
    'event_id': ['event_1', 'event_2']
})
returns = pd.read_csv('returns.csv', index_col='date', parse_dates=True)
factors = pd.read_csv('factors.csv', index_col='date', parse_dates=True)

# Configure and run
config = EventStudyConfig(
    model='market_model',
    estimation_window=(-60, -11),
    event_window=(-5, 5),
    alpha=0.05
)

study = EventStudy(config)
results = study.run(events=events, returns=returns, factors=factors)

# Save results
results.to_csv('event_study_results.csv', index=False)
```

## Data Requirements

### Events DataFrame
Required columns:
- `event_date`: datetime
- `event_id`: unique identifiers
- Optional: `event_name`, `event_type`

### Returns DataFrame
- DatetimeIndex with trading dates
- Numeric columns for each security

### Factors DataFrame
- DatetimeIndex with trading dates
- Required factors depend on model:
  - `market_model`: ['Mkt']
  - `three_factor_model`: ['Mkt', 'SMB', 'HML']
  - `four_factor_model`: ['Mkt', 'SMB', 'HML', 'UMD']

## Models

- `constant_model`: Constant mean model
- `market_model`: Market model with market factor
- `three_factor_model`: Fama-French three-factor model
- `four_factor_model`: Carhart four-factor model

## Configuration

```python
EventStudyConfig(
    model='market_model',           # Model type
    estimation_window=(-252, -1),   # Days relative to event
    event_window=(-5, 5),           # Days relative to event
    alpha=0.05,                     # Significance level
    multiple_testing_correction=None # Optional: 'benjamini_hochberg'
)
```

## Output

Results DataFrame with one row per event-security pair:

- `event_id`, `event_date`, `event_name`, `event_type`, `security_id`
- `model`, `estimation_window_start/end`, `event_window_start/end`, `alpha`
- `final_car`: Cumulative abnormal return
- `final_car_t_stat`, `final_car_p_value`: Statistical tests
- `is_significant`: Boolean significance indicator
- `n_estimation_days`, `n_event_days`: Window sizes

## Multiple Studies

```python
from event_studies import run_multiple_event_studies

configs = [
    EventStudyConfig(model='market_model', event_window=(-3, 3)),
    EventStudyConfig(model='three_factor_model', event_window=(-5, 5))
]

results_list = run_multiple_event_studies(
    events_list=[events],
    returns=returns,
    factors=factors,
    configs=configs
)
```

## Example

See `examples/simple_event_study.ipynb` for a complete working example with sample data.