import pandas as pd
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "examples" / "data_input"


@pytest.fixture(scope="session")
def returns():
    return pd.read_csv(DATA_DIR / "returns.csv", index_col="date", parse_dates=True)


@pytest.fixture(scope="session")
def factors():
    return pd.read_csv(DATA_DIR / "factors.csv", index_col="date", parse_dates=True)


@pytest.fixture(scope="session")
def events():
    return pd.read_csv(DATA_DIR / "events.csv", parse_dates=["event_date"])
