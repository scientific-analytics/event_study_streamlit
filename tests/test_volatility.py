import numpy as np
import pandas as pd
import pytest

from event_studies import EventStudyConfig, cond_vol_gjrgarch, cond_vol_change

# 106 rows total — at the 100-obs threshold for GARCH.
# All three securities should converge on the full sample.
MARKET_CONFIG = EventStudyConfig(
    model="market_model",
    estimation_window=(-25, -3),
    event_window=(-2, 2),
)


class TestCondVolGjrgarch:
    def test_returns_dataframe(self, returns, factors):
        vol = cond_vol_gjrgarch(returns["AAPL"], factors, MARKET_CONFIG)
        assert isinstance(vol, pd.DataFrame)

    def test_columns(self, returns, factors):
        vol = cond_vol_gjrgarch(returns["AAPL"], factors, MARKET_CONFIG)
        assert list(vol.columns) == ["tot_vol", "resid_vol"]

    def test_index_is_datetimeindex(self, returns, factors):
        vol = cond_vol_gjrgarch(returns["AAPL"], factors, MARKET_CONFIG)
        assert isinstance(vol.index, pd.DatetimeIndex)

    def test_tot_vol_positive(self, returns, factors):
        vol = cond_vol_gjrgarch(returns["AAPL"], factors, MARKET_CONFIG)
        assert (vol["tot_vol"].dropna() > 0).all()

    def test_resid_vol_positive(self, returns, factors):
        vol = cond_vol_gjrgarch(returns["AAPL"], factors, MARKET_CONFIG)
        assert (vol["resid_vol"].dropna() > 0).all()

    def test_all_securities_produce_output(self, returns, factors):
        for sec in returns.columns:
            vol = cond_vol_gjrgarch(returns[sec], factors, MARKET_CONFIG)
            assert list(vol.columns) == ["tot_vol", "resid_vol"]

    def test_constant_mean_resid_vol_is_nan(self, returns, factors):
        config = EventStudyConfig(model="constant_model", estimation_window=(-25, -3), event_window=(-2, 2))
        vol = cond_vol_gjrgarch(returns["AAPL"], factors, config)
        assert vol["resid_vol"].isna().all()
        assert not vol["tot_vol"].isna().all()

    def test_too_few_obs_returns_all_nan(self, returns, factors):
        # 50 rows is below the 100-obs minimum
        vol = cond_vol_gjrgarch(returns["AAPL"].iloc[:50], factors.iloc[:50], MARKET_CONFIG)
        assert vol.isna().all().all()

    def test_no_config_uses_all_factor_columns(self, returns, factors):
        # without config, all factor columns are passed to mod2
        vol = cond_vol_gjrgarch(returns["AAPL"], factors)
        assert list(vol.columns) == ["tot_vol", "resid_vol"]

    def test_three_factor_model_uses_three_factors(self, returns, factors):
        config = EventStudyConfig(model="three_factor_model", estimation_window=(-25, -3), event_window=(-2, 2))
        vol = cond_vol_gjrgarch(returns["AAPL"], factors, config)
        # factors.csv has Mkt, SMB, HML — all three are used; should converge
        assert not vol["tot_vol"].isna().all()
        assert not vol["resid_vol"].isna().all()

    def test_output_scale_reasonable(self, returns, factors):
        # daily volatility in decimal form should be small (e.g. < 0.10)
        vol = cond_vol_gjrgarch(returns["AAPL"], factors, MARKET_CONFIG)
        assert (vol["tot_vol"].dropna() < 0.10).all()


class TestCondVolChange:
    def test_returns_dataframe(self, returns, factors, events):
        result = cond_vol_change(events, returns, factors, MARKET_CONFIG)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self, returns, factors, events):
        result = cond_vol_change(events, returns, factors, MARKET_CONFIG)
        required = {"event_id", "event_date", "event_name", "event_type",
                    "security_id", "model", "tot_vol_change", "resid_vol_change"}
        assert required.issubset(set(result.columns))

    def test_at_most_one_row_per_event_security(self, returns, factors, events):
        result = cond_vol_change(events, returns, factors, MARKET_CONFIG)
        assert len(result) <= len(events) * len(returns.columns)
        # no duplicates on (event_id, security_id)
        assert not result.duplicated(subset=["event_id", "security_id"]).any()

    def test_model_column_matches_config(self, returns, factors, events):
        result = cond_vol_change(events, returns, factors, MARKET_CONFIG)
        assert (result["model"] == MARKET_CONFIG.model).all()

    def test_tot_vol_change_is_finite(self, returns, factors, events):
        result = cond_vol_change(events, returns, factors, MARKET_CONFIG)
        non_nan = result["tot_vol_change"].dropna()
        assert np.isfinite(non_nan).all()

    def test_empty_events_returns_empty(self, returns, factors):
        empty_events = pd.DataFrame({
            "event_date": pd.Series([], dtype="datetime64[ns]"),
            "event_id": pd.Series([], dtype=int),
            "event_name": pd.Series([], dtype=str),
            "event_type": pd.Series([], dtype=str),
        })
        result = cond_vol_change(empty_events, returns, factors, MARKET_CONFIG)
        assert len(result) == 0

    def test_pct_available_strict_leq_loose(self, returns, factors, events):
        strict = cond_vol_change(events, returns, factors, MARKET_CONFIG, pct_available_evt=1.0)
        loose = cond_vol_change(events, returns, factors, MARKET_CONFIG, pct_available_evt=0.0)
        assert len(strict) <= len(loose)

    def test_three_factor_config_uses_three_factors(self, returns, factors, events):
        config = EventStudyConfig(model="three_factor_model", estimation_window=(-25, -3), event_window=(-2, 2))
        result = cond_vol_change(events, returns, factors, config)
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert (result["model"] == "three_factor_model").all()
