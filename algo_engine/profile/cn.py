import datetime
import functools

from . import Profile


class ProfileCN(Profile):
    def __init__(self):
        super().__init__(
            session_start=datetime.time(9, 30),
            session_end=datetime.time(15, 0),
            session_break=[(datetime.time(11, 30), datetime.time(13, 0))]
        )

        self.trade_calendar = {}

    def override_profile(self, profile: Profile):
        profile.session_start = self.session_start
        profile.session_end = self.session_end
        profile.session_break.clear()
        profile.session_break.extend(self.session_break)

        profile.trade_calendar = self.trade_calendar

        # profile.trade_calendar = cls.trade_calendar
        # profile.is_trade_day = cls.is_trade_day
        # profile.trade_days_between = cls.trade_days_between
        # profile.time_to_seconds = cls.time_to_seconds

        profile.trade_time_between = self.trade_time_between
        profile.in_trade_session = self.in_trade_session

    @functools.lru_cache
    def trade_calendar(self, start_date: datetime.date, end_date: datetime.date, market='XSHG', tz='UTC') -> list[datetime.date]:
        import pandas as pd

        if market in self.trade_calendar:
            trade_calendar = self.trade_calendar[market]
        else:
            import exchange_calendars
            trade_calendar = self.trade_calendar[market] = exchange_calendars.get_calendar(market)

        calendar = trade_calendar.sessions_in_range(
            pd.Timestamp(start_date, tz=tz),
            pd.Timestamp(end_date, tz=tz)
        )

        # noinspection PyTypeChecker
        result = list(pd.to_datetime(calendar).date)

        return result

    @functools.lru_cache
    def is_trade_day(self, market_date: datetime.date, market='XSHG', tz='UTC') -> bool:
        if market in self.trade_calendar:
            trade_calendar = self.trade_calendar[market]
        else:
            import exchange_calendars
            trade_calendar = self.trade_calendar[market] = exchange_calendars.get_calendar(market)

        return trade_calendar.is_session(market_date)

    def trade_days_between(self, start_date: datetime.date, end_date: datetime.date = datetime.date.today(), **kwargs) -> int:
        """
        Returns the number of trade days between the given date, which is the pre-open of the start_date to the pre-open of the end_date.
        :param start_date: the given trade date
        :param end_date: the given trade date
        :return: integer number of days
        """
        assert start_date <= end_date, "The end date must not before the start date"

        if start_date == end_date:
            offset = 0
        else:
            market_date_list = self.trade_calendar(start_date=start_date, end_date=end_date, **kwargs)
            if not market_date_list:
                offset = 0
            else:
                last_trade_date = market_date_list[-1]
                offset = len(market_date_list)

                if last_trade_date == end_date:
                    offset -= 1

        return offset

    @classmethod
    def time_to_seconds(cls, t: datetime.time):
        return (t.hour * 60 + t.minute) * 60 + t.second + t.microsecond / 1000

    def trade_time_between(self, start_time: datetime.datetime | datetime.time | float | int, end_time: datetime.datetime | datetime.time | float | int, fmt='timedelta', **kwargs):
        if start_time is None or end_time is None:
            if fmt == 'timestamp':
                return 0.
            elif fmt == 'timedelta':
                return datetime.timedelta(0)
            else:
                raise NotImplementedError(f'Invalid fmt {fmt}, should be "timestamp" or "timedelta"')

        session_start = kwargs.pop('session_start', self.session_start)
        session_break = kwargs.pop('session_break', self.session_break)
        session_end = kwargs.pop('session_end', self.session_end)
        session_length_0 = datetime.timedelta(seconds=self.time_to_seconds(session_break[0]) - self.time_to_seconds(session_start))
        session_length_1 = datetime.timedelta(seconds=self.time_to_seconds(session_end) - self.time_to_seconds(session_break[1]))
        session_length = session_length_0 + session_length_1
        implied_date = datetime.date.today()

        if isinstance(start_time, (float, int)):
            start_time = datetime.datetime.fromtimestamp(start_time, tz=self.timezone)
            implied_date = start_time.date()

        if isinstance(end_time, (float, int)):
            end_time = datetime.datetime.fromtimestamp(end_time, tz=self.timezone)
            implied_date = end_time.date()

        if isinstance(start_time, datetime.time):
            start_time = datetime.datetime.combine(implied_date, start_time)

        if isinstance(end_time, datetime.time):
            end_time = datetime.datetime.combine(implied_date, end_time)

        offset = datetime.timedelta()

        market_time = start_time.time()

        # calculate the timespan from start_time to session_end
        if market_time <= session_start:
            offset += session_length
        elif session_start < market_time <= session_break[0]:
            offset += datetime.datetime.combine(start_time.date(), session_break[0]) - start_time
            offset += session_length_1
        elif session_break[0] < market_time <= session_break[1]:
            offset += session_length_1
        elif session_break[1] < market_time <= session_end:
            offset += datetime.datetime.combine(start_time.date(), session_end) - start_time
        else:
            offset += datetime.timedelta(0)

        offset -= session_length

        market_time = end_time.time()

        # calculate the timespan from session_start to end_time
        if market_time <= session_start:
            offset += datetime.timedelta(0)
        elif session_start < market_time <= session_break[0]:
            offset += end_time - datetime.datetime.combine(end_time.date(), session_start)
        elif session_break[0] < market_time <= session_break[1]:
            offset += session_length_0
        elif session_break[1] < market_time <= session_end:
            offset += end_time - datetime.datetime.combine(end_time.date(), session_break[1])
            offset += session_length_0
        else:
            offset += session_length

        # calculate market_date difference
        if start_time.date() != end_time.date():
            offset += session_length * self.trade_days_between(start_date=start_time.date(), end_date=end_time.date(), **kwargs)

        if fmt == 'timestamp':
            return offset.total_seconds()
        elif fmt == 'timedelta':
            return offset
        else:
            raise NotImplementedError(f'Invalid fmt {fmt}, should be "timestamp" or "timedelta"')

    def in_trade_session(self, market_time: datetime.datetime | float | int = None) -> bool:
        if market_time is None:
            market_time = datetime.datetime.now()

        if isinstance(market_time, (float, int)):
            market_time = datetime.datetime.fromtimestamp(market_time, tz=self.timezone)

        market_date = market_time.date()
        market_time = market_time.time()

        if not self.is_trade_day(market_date=market_date):
            return False

        if market_time < datetime.time(9, 30):
            return False

        if datetime.time(11, 30) < market_time < datetime.time(13, 00):
            return False

        if market_time > datetime.time(15, 00):
            return False

        return True


PROFILE_CN = ProfileCN()
