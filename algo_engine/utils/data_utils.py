__package__ = 'algo_engine.utils'

import argparse
import datetime
from typing import Literal, overload

import numpy as np
import pandas as pd

from ..profile import Profile, PROFILE


@overload
def ts_indices(market_date, interval, session_start, session_end, session_break, time_zone, ts_mode: Literal['start', 'end', 'both'], ts_format='timestamp') -> list[float]:
    ...


@overload
def ts_indices(market_date, interval, session_start, session_end, session_break, time_zone, ts_mode: Literal['start', 'end', 'both'], ts_format='datetime') -> list[datetime.datetime]:
    ...


def ts_indices(
        market_date: datetime.date = None,
        interval: float = 60.,
        session_start: datetime.time = datetime.time.min,
        session_end: datetime.time = None,
        session_break: list[tuple[datetime.time, datetime.time]] = None,
        time_zone: datetime.tzinfo = None,
        ts_mode: Literal['start', 'end', 'both'] | str = 'end',
        ts_format: Literal['timestamp', 'datetime'] | str = 'timestamp'
) -> list[float]:
    if market_date is None:
        market_date = datetime.date.today()

    # this is supposed to be the end_time of the given candle stick
    market_time = datetime.datetime.combine(market_date, session_start, tzinfo=time_zone) + datetime.timedelta(seconds=interval)

    if not session_end:
        session_end = datetime.datetime.combine(market_date + datetime.timedelta(days=1), datetime.time(0), tzinfo=time_zone)
    # session end in next day
    elif session_end < session_start:
        session_end = datetime.datetime.combine(market_date + datetime.timedelta(days=1), session_end, tzinfo=time_zone)
    else:
        session_end = datetime.datetime.combine(market_date, session_end, tzinfo=time_zone)

    if ts_mode == 'both':
        session_end += datetime.timedelta(seconds=interval)

    ts_index = []
    while market_time <= session_end:
        # check if the given market_time is in session break
        in_session = True

        if session_break:
            for break_range in session_break:
                break_start, break_end = break_range

                if break_start < market_time.time() <= break_end:
                    in_session = False
                    break

        if ts_mode == 'start' or ts_mode == 'both':
            _market_time = market_time - datetime.timedelta(seconds=interval)
        elif ts_mode == 'end':
            _market_time = market_time
        else:
            raise ValueError(f'Invalid ts_mode {ts_mode}!')

        if in_session:
            if ts_format == 'timestamp':
                timestamp = _market_time.timestamp()
                ts_index.append(timestamp)
            elif ts_format == 'datetime':
                ts_index.append(_market_time)
            else:
                raise ValueError(f'Invalid ts_format {ts_format}!')

        market_time += datetime.timedelta(seconds=interval)

    return ts_index


def fake_daily_data(
        ticker: str,
        start_date: datetime.date,
        end_date: datetime.date,
        p0: float = 100.,
        volatility: float = 0.20,
        calendar: list[datetime.date] = None,
        **kwargs
) -> pd.DataFrame:
    if calendar is None:
        import exchange_calendars
        market = kwargs.get('market', 'SSE')
        calendar = exchange_calendars.get_calendar(market)
        sessions = calendar.sessions_in_range(start_date, end_date)
        calendar = sorted([_.date() for _ in sessions])

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
        'date': calendar,
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
        profile: Profile = None,
        session_start: datetime.time = None,
        session_end: datetime.time = None,
        session_break: list[tuple[datetime.time, datetime.time]] = None,
        **kwargs
) -> pd.DataFrame:
    if profile is None:
        profile = PROFILE

    session_start = profile.session_start if session_start is None else session_start
    session_end = profile.session_end if session_end is None else session_end
    session_break = profile.session_break if session_break is None else session_break
    time_zone = kwargs.get('time_zone', profile.time_zone)

    if not session_start:
        session_start = datetime.time.min

    _ts_indices = ts_indices(
        market_date=market_date,
        interval=interval,
        session_start=session_start,
        session_end=session_end,
        session_break=session_break,
        time_zone=time_zone,
        ts_mode='end',
    )

    ttl_days = kwargs.get('ttl_days', 252)
    risk_free_rate = kwargs.get('risk_free_rate', 0.04)
    num_obs = len(_ts_indices)
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
        'timestamp': _ts_indices,
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
