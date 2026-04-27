from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import warnings
import tqdm

from .config import EventStudyConfig
from .models import get_model
from .estimator import clean_estimation_data, compute_abnormal_returns, clean_estimation_X
from .statistics import significance_test_ar, significance_test_car
from .validator import validate_events_data, validate_returns_data, validate_factors_data, check_date_alignment
from .windows import define_windows


class EventStudy:
    def __init__(self, config: EventStudyConfig):
        self.config = config
        self._model = None

    def run(self, events: pd.DataFrame, returns: pd.DataFrame, factors: pd.DataFrame, pct_available_est: float = 0.9,
            pct_available_evt: float = 0.9) -> pd.DataFrame:
        events_df, returns_df, factors_df = self._validate_and_prepare_data(events, returns, factors)
        self._model = get_model(self.config.model, **self.config.model_params)
        results_df, cum_car = self._run_event_studies(events_df, returns_df, factors_df, pct_available_est, pct_available_evt)
        return results_df, cum_car

    def _validate_and_prepare_data(self, events: pd.DataFrame, returns: pd.DataFrame, factors: pd.DataFrame):
        events_df = validate_events_data(events.copy())
        returns_df = validate_returns_data(returns.copy(), min_securities=self.config.min_securities)
        factors_df = validate_factors_data(factors.copy(), self._get_required_factors())
        check_date_alignment(events_df, returns_df, factors_df)
        return events_df, returns_df, factors_df

    def _get_required_factors(self):
        factor_requirements = {
            'market_model': ['Mkt'],
            'three_factor_model': ['Mkt', 'SMB', 'HML'],
            'four_factor_model': ['Mkt', 'SMB', 'HML', 'UMD'],
        }
        return factor_requirements.get(self.config.model)

    def _run_event_studies(self, events_df: pd.DataFrame, returns_df: pd.DataFrame, factors_df: pd.DataFrame,
                           pct_available_est: float, pct_available_evt: float) -> pd.DataFrame:
        all_results = []
        all_cum_results = []
        for _, event in tqdm.tqdm(events_df.iterrows()):
            result = self._run_single_event_study(event, returns_df, factors_df,
                                                  pct_available_est, pct_available_evt)
            if result is None:
                continue
            event_results, event_sum_results = result
            if event_results:
                all_results.extend(event_results)
                all_cum_results.extend(event_sum_results)
        if all_results:
            return pd.DataFrame(all_results), pd.concat(all_cum_results)
        else:
            return pd.DataFrame(), pd.DataFrame()

    def _run_single_event_study(self, event: pd.Series, returns_df: pd.DataFrame, factors_df: pd.DataFrame,
                                pct_available_est: float, pct_available_evt:float):
        event_date = pd.Timestamp(event['event_date'])
        event_id = event.get('event_id', event.name)
        event_name = event.get('event_name', f'Event_{event_id}')
        event_type = event.get('event_type', 'unknown')

        est_dates, evt_dates = define_windows(event_date, self.config.estimation_window, self.config.event_window, returns_df.index)
        # if self.config.estimation_window == 'Hurricane season':
        #     self.config.estimation_window = (est_dates[0], est_dates[1])

        est_returns = returns_df.reindex(est_dates)
        evt_returns = returns_df.reindex(evt_dates)
        est_factors = factors_df.reindex(est_dates)
        evt_factors = factors_df.reindex(evt_dates)

        valid_securities = (est_returns.notna().mean() >= pct_available_est) & (evt_returns.notna().mean() >= pct_available_evt)
        est_returns = est_returns.loc[:, valid_securities]
        evt_returns = evt_returns.loc[:, valid_securities]

        if est_returns.shape[1] == 0:
            return None

        X_est = self._model.create_design_matrix(est_factors)
        X_evt = self._model.create_design_matrix(evt_factors)
        #X_est_clean, Y_est_clean = clean_estimation_data(X_est, est_returns.values)
        X_est_clean, Y_est_clean = clean_estimation_X(X_est, est_returns.values)

        #self._model.fit(X_est_clean, Y_est_clean) #FIXME given the clean_estimation_data, may drop rows even if one asset only is missing
        self._model.fit_single(X_est_clean, Y_est_clean) # now do not consider only if X is missing, and asset-by-asset estimates
        abnormal_returns, ar_variances, IplusH, sigma2 = compute_abnormal_returns(X_evt, self._model.parameters, evt_returns.values, X_est_clean, Y_est_clean)

        degrees_of_freedom = self._model.get_degrees_of_freedom(X_est_clean.shape[0])
        #ar_t_stats, ar_p_values, ar_significant = significance_test_ar(abnormal_returns, ar_variances, degrees_of_freedom, self.config.alpha)
        car, car_variances, car_t_stats, car_p_values, car_significant = significance_test_car(abnormal_returns, ar_variances, degrees_of_freedom, self.config.alpha, IplusH=IplusH, sigma2=sigma2)
        #effective_dates = pd.bdate_range(event_date + pd.Timedelta(self.config.event_window[0], unit='D'),
        #               event_date + pd.Timedelta(self.config.event_window[1], unit='D'))
        effective_dates = evt_dates.copy()
        effective_int = [(np.busday_count(event_date.date(), d.date())) for d in effective_dates]
        #TODO: if there is a missing DATE in the returns, the business days + 5 may go to actually 6 business days ahead
        #TODO: danger is a series with many missing in a row -> use clean_daily_series

        if self.config.estimation_window == 'Hurricane season':
            est_window_start_out, est_window_end_out  = est_dates[0], est_dates[1]
        else:
            est_window_start_out, est_window_end_out = self.config.estimation_window[0], self.config.estimation_window[1]


        results = []
        results_cum = []
        for j, security_id in enumerate(est_returns.columns):
            final_car = car[-1, j] if not np.isnan(car[-1, j]) else np.nan
            final_car_t_stat = car_t_stats[-1, j] if not np.isnan(car_t_stats[-1, j]) else np.nan
            final_car_p_value = car_p_values[-1, j] if not np.isnan(car_p_values[-1, j]) else np.nan
            is_significant = car_significant[-1, j] if not np.isnan(car_significant[-1, j]) else np.nan

            results.append({
                'event_id': event_id,
                'event_date': event_date.strftime('%Y-%m-%d'),
                'event_name': event_name,
                'event_type': event_type,
                'security_id': security_id,
                'model': self.config.model,
                'estimation_window_start': est_window_start_out,
                'estimation_window_end': est_window_end_out,
                'event_window_start': self.config.event_window[0],
                'event_window_end': self.config.event_window[1],
                'alpha': self.config.alpha,
                'final_car': float(final_car),
                'final_car_t_stat': float(final_car_t_stat),
                'final_car_p_value': float(final_car_p_value),
                'is_significant': bool(is_significant),
                'n_estimation_days': len(est_dates),
                'n_event_days': len(evt_dates),
                'degrees_of_freedom': int(degrees_of_freedom),
            })
            results_cum.append(pd.DataFrame({
                'event_id': event_id,
                'event_date': event_date.strftime('%Y-%m-%d'),
                'event_name': event_name,
                'event_type': event_type,
                'security_id': security_id,
                'model': self.config.model,
                'estimation_window_start': est_window_start_out,
                'estimation_window_end': est_window_end_out,
                'event_window_start': self.config.event_window[0],
                'event_window_end': self.config.event_window[1],
                'alpha': self.config.alpha,
                'cum_car': pd.Series(car[:, j]),
                'cum_car_variances': pd.Series(car_variances[:, j]),
                'effective_date': pd.Series(effective_dates),
                'effective_int': pd.Series(effective_int),
            }))
        return results, results_cum



def run_multiple_event_studies(events_list: List[pd.DataFrame], returns: pd.DataFrame, factors: pd.DataFrame, configs: List[EventStudyConfig]) -> List[pd.DataFrame]:
    all_results = []
    for config in configs:
        study = EventStudy(config)
        config_results = []
        for events in events_list:
            results = study.run(events=events, returns=returns, factors=factors)
            config_results.append(results)
        combined_df = pd.concat(config_results, ignore_index=True) if config_results else pd.DataFrame()
        all_results.append(combined_df)
    return all_results