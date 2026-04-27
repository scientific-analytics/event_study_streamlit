import numpy as np
import pandas as pd
import tqdm
from pandas.tseries.offsets import BDay
from arch.univariate import ConstantMean, ARX, GARCH, SkewStudent

_MODEL_FACTORS = {
    'constant_model': [],
    'market_model': ['Mkt'],
    'three_factor_model': ['Mkt', 'SMB', 'HML'],
    'four_factor_model': ['Mkt', 'SMB', 'HML', 'UMD'],
}


def cond_vol_gjrgarch(return_series: pd.Series, factors: pd.DataFrame, config=None) -> pd.DataFrame:
    if config is not None:
        cols = _MODEL_FACTORS.get(config.model, list(factors.columns))
        factors = factors[cols] if cols else pd.DataFrame(index=factors.index)

    has_factors = len(factors.columns) > 0

    if has_factors:
        common_dates = list(return_series.index.intersection(factors.index))
        ret_to_garch = pd.concat(
            [return_series.loc[common_dates], factors.loc[common_dates]], axis=1
        ).dropna() * 100
    else:
        common_dates = list(return_series.index)
        ret_to_garch = return_series.loc[common_dates].dropna().to_frame() * 100

    nan_result = pd.concat(
        [return_series.loc[common_dates].dropna()] * 2,
        axis=1, keys=['tot_vol', 'resid_vol']
    ) * np.nan

    if len(ret_to_garch) < 100:
        return nan_result

    y = ret_to_garch.iloc[:, 0].rename('ret')

    mod1 = ConstantMean(y=y, volatility=GARCH(1, 1, 1), distribution=SkewStudent())
    res1 = mod1.fit(disp="off")

    if not has_factors:
        if res1.convergence_flag == 0:
            vol = res1.conditional_volatility / 100
            return pd.concat([vol, vol * np.nan], axis=1, keys=['tot_vol', 'resid_vol'])
        return nan_result

    X = ret_to_garch.iloc[:, 1:]
    mod2 = ARX(y=y, x=X, volatility=GARCH(1, 1, 1), distribution=SkewStudent())
    res2 = mod2.fit(disp="off")

    if (res1.convergence_flag == 0) and (res2.convergence_flag == 0):
        return pd.concat(
            [res1.conditional_volatility, res2.conditional_volatility],
            axis=1, keys=['tot_vol', 'resid_vol']
        ) / 100
    return nan_result


def cond_vol_change(events: pd.DataFrame, returns: pd.DataFrame, factors: pd.DataFrame,
                    config, pct_available_evt: float = 0.95, change: str = 'pct') -> pd.DataFrame:
    # fit GARCH once per security on the full sample — no estimation window
    vol_dict = {}
    for security_id in tqdm.tqdm(returns.columns, desc="Fitting GARCH"):
        vol_dict[security_id] = cond_vol_gjrgarch(returns[security_id], factors, config)

    # vol change window starts one business day before the event window
    # so that vol[0] is the pre-event reference and vol[-1] is the post-event level
    vol_window = (config.event_window[0] - 1, config.event_window[1])
    available_dates = pd.DatetimeIndex(sorted(returns.index)).tz_localize(None)

    results = []
    for _, event in tqdm.tqdm(events.iterrows(), desc="Computing vol changes"):
        event_date = pd.Timestamp(event['event_date']).tz_localize(None)
        event_id = event.get('event_id', event.name)
        event_name = event.get('event_name', f'Event_{event_id}')
        event_type = event.get('event_type', 'unknown')

        actual_event_dates = available_dates[available_dates >= event_date]
        if len(actual_event_dates) == 0:
            continue
        actual_event_date = actual_event_dates[0]
        vol_start = actual_event_date + BDay(vol_window[0])
        vol_end = actual_event_date + BDay(vol_window[1])
        vol_dates = available_dates[(available_dates >= vol_start) & (available_dates <= vol_end)]

        if len(vol_dates) == 0:
            continue

        for security_id in returns.columns:
            vol_df = vol_dict.get(security_id)
            if vol_df is None:
                continue

            vol_window_data = vol_df.reindex(vol_dates)

            if vol_window_data['tot_vol'].notna().mean() < pct_available_evt:
                continue

            tot_series = vol_window_data['tot_vol'].dropna()
            resid_series = vol_window_data['resid_vol'].dropna()

            if change == 'pct':
                tot_vol_change = np.log(tot_series.iloc[-1] / tot_series.iloc[0]) if len(tot_series) >= 2 else np.nan
                resid_vol_change = np.log(resid_series.iloc[-1] / resid_series.iloc[0]) if len(resid_series) >= 2 else np.nan
            elif change == 'diff':
                tot_vol_change = tot_series.iloc[-1] - tot_series.iloc[0] if len(tot_series) >= 2 else np.nan
                resid_vol_change = resid_series.iloc[-1] - resid_series.iloc[0] if len(resid_series) >= 2 else np.nan

            results.append({
                'event_id': event_id,
                'event_date': event_date.strftime('%Y-%m-%d'),
                'event_name': event_name,
                'event_type': event_type,
                'security_id': security_id,
                'model': config.model,
                'tot_vol_change': tot_vol_change,
                'resid_vol_change': resid_vol_change,
            })

    return pd.DataFrame(results)