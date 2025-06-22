import datetime

from c_base import PROFILE, PROFILE_DEFAULT


def default_profile_test():
    dt = datetime.datetime.now()
    # test in_session
    print('in_session: ', PROFILE.is_market_session(dt))
    ts = dt.timestamp()
    print('in_session: ', PROFILE.is_market_session(ts))

    sd = datetime.date.today()
    ed = datetime.date.today() + datetime.timedelta(days=9)
    print('tz_offset: ', PROFILE.tz_offset)
    print(f'trade_calendar: from {sd} to {ed}', PROFILE.trade_calendar(sd, ed))
    print('trade_day_before: ', PROFILE.trading_days_before(sd, 1))
    print('trade_day_after: ', PROFILE.trading_days_after(sd, 1))
    print('nearest_trade_day: ', PROFILE.nearest_trading_date(sd))

    print('override with default :', PROFILE_DEFAULT.override_profile())
    print('ts of 10:00: ', PROFILE.time_to_seconds(datetime.time(10), break_adjusted=True))
    print(f'ts of {(t := datetime.datetime.combine(datetime.date(2024, 11, 11), datetime.time(11)))}: ', PROFILE.timestamp_to_seconds(t.timestamp(), break_adjusted=True))

    print('ts of 15:00: ', PROFILE.time_to_seconds(datetime.time(15), break_adjusted=False))


def cn_profile_test():
    from c_cn import PROFILE_CN
    PROFILE_CN.override_profile()

    start_time = datetime.datetime(2024, 11, 11, 9, 45, 31, 500_000)
    end_time = datetime.datetime(2024, 11, 18, 13, 21, 59, 750_000)
    out_session_time_0 = datetime.datetime(2024, 11, 10, 9, 50)
    out_session_time_1 = datetime.datetime(2024, 11, 11, 11, 45)
    n = 25

    print(f'testing profile {PROFILE}...')
    print(f'[0/{n}] session_length, expect=14400, got={PROFILE.session_length}')
    print(f'[1/{n}] time_to_seconds({start_time.time()}, False), expect=35131.5, got={PROFILE.time_to_seconds(start_time.time(), False)}')
    print(f'[2/{n}] time_to_seconds({start_time.time()}, True), expect=931.5, got={PROFILE.time_to_seconds(start_time.time(), True)}')
    print(f'[3/{n}] time_to_seconds({end_time.time()}, True), expect=8519.75, got={PROFILE.time_to_seconds(end_time.time(), True)}')
    print(f'[4/{n}] timestamp_to_seconds({end_time.time()}, True), expect=8519.75, got={PROFILE.timestamp_to_seconds(end_time.timestamp(), True)}')
    print(f'[5/{n}] trade_time_between({start_time}, {end_time}), expect=79588.25, got={PROFILE.trading_time_between(start_time, end_time)}')
    print(f'[6/{n}] is_market_session({start_time}), expect=True, got={PROFILE.is_market_session(start_time)}')
    print(f'[7/{n}] is_market_session({out_session_time_0}), expect=False, got={PROFILE.is_market_session(out_session_time_0)}')
    print(f'[8/{n}] is_market_session({out_session_time_1}), expect=False, got={PROFILE.is_market_session(out_session_time_1)}')
    print(f'[9/{n}] <this is expected to failed. as using timestamp is actually equals to using time()>. is_market_session({out_session_time_0.timestamp()}), expect=False, got={PROFILE.is_market_session(out_session_time_0.timestamp())}')
    print(f'[10/{n}] is_market_session({out_session_time_1.time()}), expect=False, got={PROFILE.is_market_session(out_session_time_1.time())}')
    print(f'[11/{n}] is_market_session({start_time.time()}), expect=True, got={PROFILE.is_market_session(start_time.time())}')
    print(f'[12/{n}] is_market_session({end_time.timestamp()}), expect=True, got={PROFILE.is_market_session(end_time.timestamp())}')
    print(f'[13/{n}] trade_calendar({start_time.date()}, {end_time.date()}), expect=[6 dates ...], got={PROFILE.trade_calendar(start_time.date(), end_time.date())}')
    print(f'[14/{n}] trading_days_before({start_time.date()}, 1), expect=2024-11-08, got={PROFILE.trading_days_before(start_time.date(), 1)}')
    print(f'[15/{n}] trading_days_before({datetime.date(2024, 11, 15)}, 2), expect=2024-11-13, got={PROFILE.trading_days_before(datetime.date(2024, 11, 15), 2)}')
    print(f'[16/{n}] trading_days_before({datetime.date(2024, 11, 17)}, 2), expect=2024-11-14, got={PROFILE.trading_days_before(datetime.date(2024, 11, 17), 2)}')
    print(f'[17/{n}] trading_days_after({datetime.date(2024, 11, 15)}, 1), expect=2024-11-18, got={PROFILE.trading_days_after(datetime.date(2024, 11, 15), 1)}')
    print(f'[18/{n}] trading_days_after({datetime.date(2024, 11, 12)}, 2), expect=2024-11-14, got={PROFILE.trading_days_after(datetime.date(2024, 11, 12), 2)}')
    print(f'[19/{n}] trading_days_after({datetime.date(2024, 11, 16)}, 2), expect=2024-11-19, got={PROFILE.trading_days_after(datetime.date(2024, 11, 16), 2)}')
    print(f'[20/{n}] trading_days_between({start_time.date()}, {end_time.date()}), expect=5 got={PROFILE.trading_days_between(start_time.date(), end_time.date())}')
    print(f'[21/{n}] trading_days_between({start_time.date()}, {datetime.date(2024, 11, 16)}), expect=5 got={PROFILE.trading_days_between(start_time.date(), end_time.date())}')
    print(f'[22/{n}] nearest_trading_date({start_time.date()}), expect={start_time.date()} got={PROFILE.nearest_trading_date(start_time.date())}')
    print(f'[23/{n}] nearest_trading_date({datetime.date(2024, 11, 16)}), expect={datetime.date(2024, 11, 15)} got={PROFILE.nearest_trading_date(datetime.date(2024, 11, 16))}')
    print(f'[24/{n}] nearest_trading_date({datetime.date(2024, 11, 16)}, "next"), expect={datetime.date(2024, 11, 18)} got={PROFILE.nearest_trading_date(datetime.date(2024, 11, 16), 'next')}')
    print(f'[25/{n}] .range_break, expect={{ dict of 3 break-bounds}} got={PROFILE.range_break}')


def main():
    # default_profile_test()
    cn_profile_test()


if __name__ == '__main__':
    main()
