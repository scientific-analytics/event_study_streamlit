import numpy as np
import pandas as pd
import pytest

from event_studies import EventStudy, EventStudyConfig

# Data has 106 rows (2023-01-03 to 2023-06-07).
# First event is 2023-02-15 (~30 bdays from start), so estimation_window
# of (-25, -3) gives ~22 bdays of estimation data — enough for OLS.
SMALL_CONFIG = EventStudyConfig(
    model="market_model",
    estimation_window=(-25, -3),
    event_window=(-2, 2),
)


class TestEventStudyRunOutputStructure:
    def test_returns_tuple_of_two_dataframes(self, returns, factors, events):
        study = EventStudy(SMALL_CONFIG)
        result = study.run(events, returns, factors)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], pd.DataFrame)

    def test_results_df_required_columns(self, returns, factors, events):
        study = EventStudy(SMALL_CONFIG)
        results_df, _ = study.run(events, returns, factors)
        required = {
            "event_id", "event_date", "security_id", "model",
            "final_car", "final_car_t_stat", "final_car_p_value", "is_significant",
            "n_estimation_days", "n_event_days",
        }
        assert required.issubset(set(results_df.columns))

    def test_cum_car_required_columns(self, returns, factors, events):
        study = EventStudy(SMALL_CONFIG)
        _, cum_car = study.run(events, returns, factors)
        required = {"cum_car", "cum_car_variances", "effective_date", "effective_int", "security_id", "event_id"}
        assert required.issubset(set(cum_car.columns))

    def test_non_empty_for_valid_inputs(self, returns, factors, events):
        study = EventStudy(SMALL_CONFIG)
        results_df, cum_car = study.run(events, returns, factors)
        assert len(results_df) > 0
        assert len(cum_car) > 0

    def test_at_most_n_events_times_n_securities_rows(self, returns, factors, events):
        study = EventStudy(SMALL_CONFIG)
        results_df, _ = study.run(events, returns, factors)
        assert len(results_df) <= len(events) * len(returns.columns)

    def test_model_column_matches_config(self, returns, factors, events):
        study = EventStudy(SMALL_CONFIG)
        results_df, _ = study.run(events, returns, factors)
        assert (results_df["model"] == SMALL_CONFIG.model).all()

    def test_cum_car_rows_per_group_bounded_by_event_window(self, returns, factors, events):
        study = EventStudy(SMALL_CONFIG)
        _, cum_car = study.run(events, returns, factors)
        max_days = SMALL_CONFIG.event_window[1] - SMALL_CONFIG.event_window[0] + 1
        for _, group in cum_car.groupby(["event_id", "security_id"]):
            assert len(group) <= max_days


class TestEventStudyRunTypes:
    def test_final_car_is_float(self, returns, factors, events):
        study = EventStudy(SMALL_CONFIG)
        results_df, _ = study.run(events, returns, factors)
        assert pd.api.types.is_float_dtype(results_df["final_car"])

    def test_is_significant_is_bool_compatible(self, returns, factors, events):
        study = EventStudy(SMALL_CONFIG)
        results_df, _ = study.run(events, returns, factors)
        assert results_df["is_significant"].dtype in (bool, object, np.dtype("bool"))

    def test_p_values_between_0_and_1(self, returns, factors, events):
        study = EventStudy(SMALL_CONFIG)
        results_df, _ = study.run(events, returns, factors)
        pvals = results_df["final_car_p_value"].dropna()
        assert ((pvals >= 0) & (pvals <= 1)).all()

    def test_cum_car_variances_non_negative(self, returns, factors, events):
        study = EventStudy(SMALL_CONFIG)
        _, cum_car = study.run(events, returns, factors)
        assert (cum_car["cum_car_variances"].dropna() >= 0).all()


class TestEventStudyRunModels:
    def test_three_factor_model(self, returns, factors, events):
        config = EventStudyConfig(
            model="three_factor_model",
            estimation_window=(-25, -3),
            event_window=(-2, 2),
        )
        results_df, _ = EventStudy(config).run(events, returns, factors)
        assert len(results_df) > 0
        assert (results_df["model"] == "three_factor_model").all()

    def test_constant_mean_model(self, returns, factors, events):
        config = EventStudyConfig(
            model="constant_model",
            estimation_window=(-25, -3),
            event_window=(-2, 2),
        )
        results_df, _ = EventStudy(config).run(events, returns, factors)
        assert len(results_df) > 0


class TestEventStudyRunEdgeCases:
    def test_empty_events_returns_empty_dataframes(self, returns, factors):
        empty_events = pd.DataFrame({
            "event_date": pd.Series([], dtype="datetime64[ns]"),
            "event_id": pd.Series([], dtype=int),
            "event_name": pd.Series([], dtype=str),
            "event_type": pd.Series([], dtype=str),
        })
        study = EventStudy(SMALL_CONFIG)
        results_df, cum_car = study.run(empty_events, returns, factors)
        assert len(results_df) == 0
        assert len(cum_car) == 0

    def test_single_event(self, returns, factors, events):
        single_event = events.iloc[[0]].copy()
        study = EventStudy(SMALL_CONFIG)
        results_df, _ = study.run(single_event, returns, factors)
        assert len(results_df) > 0
        assert results_df["event_id"].nunique() == 1
