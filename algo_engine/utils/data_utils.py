__package__ = 'algo_engine.utils'

import argparse
import datetime
from typing import Literal, overload

import numpy as np
import pandas as pd

from exchange_profile.c_exchange_profile import SessionBreak, SessionTimeRange
from ..exchange_profile import PROFILE, SessionDate, SessionTime


@overload
def ts_indices(
        market_date: datetime.date | SessionDate = None,
        interval: datetime.timedelta | float = 60.,
        session_start: datetime.time | SessionTime = None,
        session_end: datetime.time | SessionTime = None,
        session_breaks: list[tuple[datetime.time, datetime.time]] | SessionBreak = None,
        ts_mode: Literal['start', 'end', 'both'] | str = 'end',
        ts_format='timestamp'
) -> list[float]:
    ...


@overload
def ts_indices(
        market_date: datetime.date | SessionDate = None,
        interval: datetime.timedelta | float = 60.,
        session_start: datetime.time | SessionTime = None,
        session_end: datetime.time | SessionTime = None,
        session_breaks: list[tuple[datetime.time, datetime.time]] | SessionBreak = None,
        ts_mode: Literal['start', 'end', 'both'] | str = 'end',
        ts_format='datetime'
) -> list[datetime.datetime]:
    ...


@overload
def ts_indices(
        market_date: datetime.date | SessionDate = None,
        interval: datetime.timedelta | float = 60.,
        session_start: datetime.time | SessionTime = None,
        session_end: datetime.time | SessionTime = None,
        session_breaks: list[tuple[datetime.time, datetime.time]] | SessionBreak = None,
        ts_mode: Literal['start', 'end', 'both'] | str = 'end',
        ts_format='session_time'
) -> list[SessionTime]:
    ...


def ts_indices(
        market_date: datetime.date | SessionDate = None,
        interval: datetime.timedelta | float = 60.,
        session_start: datetime.time | SessionTime = None,
        session_end: datetime.time | SessionTime = None,
        session_breaks: list[tuple[datetime.time, datetime.time]] | SessionBreak = None,
        ts_mode: Literal['start', 'end', 'both'] | str = 'end',
        ts_format: Literal['timestamp', 'datetime', 'session_time'] | str = 'timestamp',
        **kwargs
) -> list[float]:
    # Regularize input parameters
    if market_date is None:
        market_date = datetime.date.today()
    elif isinstance(market_date, datetime.date):
        market_date = SessionDate.from_pydate(market_date)
    elif isinstance(market_date, pd.Timestamp):
        market_date = SessionDate.from_pydate(market_date.date())
    elif isinstance(market_date, SessionDate):
        pass
    else:
        raise ValueError(f'Invalid market_date {market_date}!')

    if session_start is None:
        start_ts = PROFILE.session_start_ts
    elif isinstance(session_start, datetime.time):
        start_ts = SessionTime.from_pytime(session_start).ts
    elif isinstance(session_start, SessionTime):
        start_ts = session_start.ts
    else:
        raise ValueError(f'Invalid session_start {session_start}!')

    if session_end is None:
        end_ts = PROFILE.session_end_ts
    elif isinstance(session_end, datetime.time):
        end_ts = SessionTime.from_pytime(session_end).ts
    elif isinstance(session_end, SessionTime):
        end_ts = session_end.ts
    else:
        raise ValueError(f'Invalid session_end {session_end}!')

    if isinstance(interval, datetime.timedelta):
        interval = interval.total_seconds()
    elif isinstance(interval, (int, float)):
        interval = float(interval)
    elif isinstance(interval, pd.Timedelta):
        interval = interval.total_seconds()
    elif isinstance(interval, SessionTimeRange):
        interval = interval.elapsed_seconds
    else:
        raise ValueError(f'Invalid interval {interval}!')

    if session_breaks is None:
        breaks_ts = [(session_break.break_start_ts, session_break.break_end_ts) for session_break in PROFILE.session_breaks]
    elif isinstance(session_breaks, SessionBreak):
        breaks_ts = [(session_breaks.break_start_ts, session_breaks.break_end_ts)]
    elif isinstance(session_breaks, list):
        breaks_ts = [(PROFILE.time_to_seconds(session_break[0], False), PROFILE.time_to_seconds(session_break[1], False)) for session_break in session_breaks]
    else:
        raise ValueError(f'Invalid session_breaks {session_breaks}!')

    market_date: SessionDate
    md_ts = market_date.timestamp()
    start_ts: float
    end_ts: float
    breaks_ts: list[tuple[float, float]]
    interval: float

    # this is supposed to be the end_time of the given candle stick
    ts = start_ts + interval
    if ts_mode == 'both':
        end_ts += interval

    ts_index = []
    while ts <= end_ts:
        # check if the given market_time is in session break
        in_session = True

        if breaks_ts:
            for break_start, break_end in breaks_ts:
                if break_start < ts <= break_end:
                    in_session = False
                    break

        if ts_mode == 'start' or ts_mode == 'both':
            _ts = ts - interval
        elif ts_mode == 'end':
            _ts = ts
        else:
            raise ValueError(f'Invalid ts_mode {ts_mode}!')

        if in_session:
            if ts_format == 'timestamp':
                ts_index.append(_ts + md_ts)
            elif ts_format == 'datetime':
                ts_index.append(PROFILE.timestamp_to_datetime(_ts))
            elif ts_format == 'session_time':
                ts_index.append(SessionTime.from_ts(_ts))
            else:
                raise ValueError(f'Invalid ts_format {ts_format}!')

        ts += interval

    return ts_index


def fake_daily_data(
        start_date: datetime.date,
        end_date: datetime.date,
        p0: float = 100.,
        volatility: float = 0.20,
        calendar: list[datetime.date] = None,
        **kwargs
) -> pd.DataFrame:
    if calendar is None:
        calendar = [_.to_pydate() for _ in PROFILE.trade_calendar(start_date, end_date)]

    ttl_days = kwargs.get('ttl_days', 252)
    risk_free_rate = kwargs.get('risk_free_rate', 0.04)

    num_days = len(calendar)
    daily_volatility = volatility / np.sqrt(ttl_days)
    daily_risk_free_rate = np.log(1 + risk_free_rate) / ttl_days

    # Generate percentage changes
    pct_changes = np.random.lognormal(mean=daily_risk_free_rate, sigma=daily_volatility, size=num_days)

    # Generate close prices
    close_price = p0 * pct_changes.cumprod()

    # Generate open, high, low prices
    high_deviation = np.random.exponential(scale=daily_volatility, size=num_days)
    low_deviation = -np.random.exponential(scale=daily_volatility, size=num_days)

    high_price = close_price * np.exp(high_deviation)
    low_price = close_price * np.exp(low_deviation)
    open_price = np.random.uniform(low=low_price, high=high_price)

    data = pd.DataFrame({
        'date': list(calendar),
        'open_price': open_price,
        'high_price': high_price,
        'low_price': low_price,
        'close_price': close_price
    })

    data.set_index(keys='date', inplace=True)
    return data


def fake_data(
        market_date: datetime.date,
        p0: float = 100.,
        volatility: float = 0.20,
        interval: float = 60.,
        **kwargs
) -> pd.DataFrame:
    indices = ts_indices(market_date=market_date, interval=interval, **kwargs)

    ttl_days = kwargs.get('ttl_days', 252)
    risk_free_rate = kwargs.get('risk_free_rate', 0.04)
    num_obs = len(indices)
    obs_volatility = volatility / np.sqrt(ttl_days * num_obs)
    obs_risk_free_rate = np.log(1 + risk_free_rate) / ttl_days / num_obs

    pct_changes = np.random.lognormal(mean=obs_risk_free_rate, sigma=obs_volatility, size=num_obs)
    close_price = p0 * pct_changes.cumprod()

    # gamma distribution with shape = 1 is the same of exponential.
    # high_deviation = np.random.gamma(shape=1, scale=obs_volatility, size=num_obs)
    # low_deviation = -np.random.gamma(shape=1, scale=obs_volatility, size=num_obs)
    high_deviation = np.random.exponential(scale=obs_volatility, size=num_obs)
    low_deviation = -np.random.exponential(scale=obs_volatility, size=num_obs)

    high_price = close_price * np.exp(high_deviation)
    low_price = close_price * np.exp(low_deviation)
    # open_price = np.random.uniform(low=low_price, high=high_price)

    open_price = np.concatenate(([p0], close_price[:-1]))
    high_price = np.max([high_price, open_price], axis=0)
    low_price = np.min([low_price, open_price], axis=0)

    data = pd.DataFrame({
        'timestamp': indices,
        'open_price': open_price,
        'high_price': high_price,
        'low_price': low_price,
        'close_price': close_price
    })

    data.set_index(keys='timestamp', inplace=True)
    return data


def main():
    parser = argparse.ArgumentParser(description='Generate fake market data.')
    parser.add_argument('--ticker', type=str, default='FAKE', help='Ticker symbol for the fake data')
    parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, required=True, help='End date in YYYY-MM-DD format')
    parser.add_argument('--volatility', type=float, default=0.20, help='Annualized volatility of the fake data')
    parser.add_argument('--risk_free_rate', type=float, default=0.01, help='Risk-free rate for generating fake data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--minute_data', action='store_true', help='Generate minute data instead of daily data')
    parser.add_argument('--market_date', type=str, help='Market date for minute data in YYYY-MM-DD format')
    parser.add_argument('--p0', type=float, default=100., help='Starting price for minute data')

    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.minute_data:
        market_date = datetime.datetime.strptime(args.market_date, '%Y-%m-%d').date()
        data_set = fake_data(
            market_date=market_date,
            p0=args.p0,
            volatility=args.volatility,
            risk_free_rate=args.risk_free_rate
        )
    else:
        start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime(args.end_date, '%Y-%m-%d').date()
        data_set = fake_daily_data(
            ticker=args.ticker,
            start_date=start_date,
            end_date=end_date,
            volatility=args.volatility,
            risk_free_rate=args.risk_free_rate
        )

    return data_set


def _test(seed: int = 42):
    np.random.seed(seed)

    # Example usage:
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 4, 1)
    daily_data_set = fake_daily_data('FAKE', start_date, end_date)
    minute_data_set = fake_data(market_date=start_date)
    print(daily_data_set.head())
    print(minute_data_set.head())


if __name__ == '__main__':
    _test()
