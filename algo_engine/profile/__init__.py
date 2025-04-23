import abc
import datetime
from typing import Self, Literal


class Profile(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            profile_id: str,
            session_start: datetime.time | None = None,
            session_end: datetime.time | None = None,
            session_break: list[tuple[datetime.time, datetime.time]] = None
    ):
        self.profile_id = profile_id
        self.session_start = session_start
        self.session_end = session_end
        self.session_break = [] if session_break is None else session_break

        self.time_zone = None

    def __repr__(self):
        return f'<Profile {self.profile_id}>({id(self)})'

    def override_profile(self, profile: Self = None) -> Self:
        if profile is None:
            profile = PROFILE

        profile.profile_id = self.profile_id
        profile.session_start = self.session_start
        profile.session_end = self.session_end

        if profile.session_break is None or self.session_break is None:
            profile.session_break = self.session_break
        else:
            profile.session_break.clear()
            profile.session_break.extend(self.session_break)

        profile.trade_time_between = self.trade_time_between
        profile.is_market_session = self.is_market_session
        profile.trade_calendar = self.trade_calendar

        return profile

    @abc.abstractmethod
    def trade_time_between(self, start_time: datetime.datetime | float, end_time: datetime.datetime | float, **kwargs) -> datetime.timedelta:
        ...

    @abc.abstractmethod
    def is_market_session(self, timestamp: float | int | datetime.datetime, **kwargs) -> bool:
        ...

    @abc.abstractmethod
    def trade_calendar(self, start_date: datetime.date, end_date: datetime.date, **kwargs) -> list[datetime.date]:
        ...

    def trading_days_before(self, market_date: datetime.date, days: int, **kwargs) -> datetime.date:
        """
        Calculate the trading date that is `days` trading days before the given market_date.

        Args:
            market_date: The reference date
            days: Number of trading days to go back (must be positive)

        Returns:
            The trading date that is `days` trading days before market_date

        Raises:
            ValueError: If days is not positive
        """
        if days <= 0:
            raise ValueError("days must be positive")

        # Calculate how many years we need to go back (250 trading days ≈ 1 year)
        years_back = (days // 250) + 1
        start_date = market_date - datetime.timedelta(days=365 * years_back)

        # Get the trade calendar
        trade_dates = self.trade_calendar(start_date=start_date, end_date=market_date, **kwargs)

        # Find the market_date in the calendar (it should be the last or near last)
        if market_date == trade_dates[-1]:
            idx = len(trade_dates) - days - 1
        else:
            idx = len(trade_dates) - days

        if idx < 0:
            return self.trading_days_before(market_date=trade_dates[0], days=-idx, **kwargs)

        return trade_dates[idx]

    def trading_days_after(self, market_date: datetime.date, days: int, **kwargs) -> datetime.date:
        """
        Calculate the trading date that is `days` trading days after the given market_date.

        Args:
            market_date: The reference date
            days: Number of trading days to go forward (must be positive)

        Returns:
            The trading date that is `days` trading days after market_date

        Raises:
            ValueError: If days is not positive
        """
        if days <= 0:
            raise ValueError("days must be positive")

        # Calculate how many years we need to look ahead (250 trading days ≈ 1 year)
        years_ahead = (days // 250) + 1
        end_date = market_date + datetime.timedelta(days=365 * years_ahead)

        # Get the trade calendar
        trade_dates = self.trade_calendar(start_date=market_date, end_date=end_date, **kwargs)

        # Find the market_date in the calendar (it should be the first or near first)
        if market_date == trade_dates[0]:
            idx = days
        else:
            # If market_date isn't a trading day, the first date is the next trading day
            idx = days - 1

        if idx >= len(trade_dates):
            # Need to look further ahead
            return self.trading_days_after(market_date=trade_dates[-1], days=idx - len(trade_dates) + 1, **kwargs)

        return trade_dates[idx]

    def nearest_trading_date(
            self,
            market_date: datetime.date,
            method: Literal['previous', 'next'] = 'previous',
            **kwargs
    ) -> datetime.date:
        """
        Find the nearest trading date relative to the given market_date.

        Args:
            market_date: The reference date
            method: Either 'previous' (default) or 'next' to determine which side to look when the date isn't a trading day

        Keyword Args:
            **kwargs: Additional arguments passed to trade_calendar

        Returns:
            The nearest trading date according to the specified method

        Raises:
            ValueError: If method is not 'previous' or 'next'
        """
        if method not in ('previous', 'next'):
            raise ValueError("method must be either 'previous' or 'next'")

        # If not a trading day, find nearest according to method
        if method == 'previous':
            # Get previous trading day by looking back 1 year (safe upper bound)
            previous_dates = self.trade_calendar(
                start_date=market_date - datetime.timedelta(days=30),
                end_date=market_date,
                **kwargs
            )
            return previous_dates[-1]
        else:
            # Get next trading day by looking ahead 1 year (safe upper bound)
            next_dates = self.trade_calendar(
                start_date=market_date,
                end_date=market_date + datetime.timedelta(days=30),
                **kwargs
            )
            return next_dates[0]

    @property
    def range_break(self) -> list[dict]:
        """
        an range break designed for plotly.
        """
        range_break = []

        if not self.session_break:
            return range_break

        # Convert session_break to range_break format
        for start, end in self.session_break:
            start_hour = start.hour + start.minute / 60
            end_hour = end.hour + end.minute / 60
            range_break.append(dict(bounds=[start_hour, end_hour], pattern="hour"))

        # Add the additional fixed non-trading periods
        if self.session_start is not None and self.session_start != datetime.time.min:
            range_break.append(
                dict(bounds=[0, self.session_start.hour + self.session_start.minute / 60], pattern="hour"),
            )

        if self.session_end is not None and self.session_end != datetime.time.max:
            range_break.append(
                dict(bounds=[self.session_end.hour + self.session_end.minute / 60, 24], pattern="hour"),
            )

        return range_break


class DefaultProfile(Profile):
    def __init__(self):
        super().__init__(
            profile_id='non-stop',
            session_start=datetime.time.min,
            session_end=None,
            session_break=None
        )

    def trade_time_between(self, start_time: datetime.datetime | float, end_time: datetime.datetime | float, **kwargs) -> datetime.timedelta:
        if start_time is not None and isinstance(start_time, (float, int)):
            start_time = datetime.datetime.fromtimestamp(start_time, tz=self.time_zone)

        if end_time is not None and isinstance(end_time, (float, int)):
            end_time = datetime.datetime.fromtimestamp(end_time, tz=self.time_zone)

        if start_time is None or end_time is None:
            return datetime.timedelta(seconds=0)

        if start_time > end_time:
            return datetime.timedelta(seconds=0)

        return end_time - start_time

    def is_market_session(self, timestamp: float | int | datetime.datetime, **kwargs) -> bool:
        return True

    def trade_calendar(self, start_date: datetime.date, end_date: datetime.date, **kwargs) -> list[datetime.date]:
        return [start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)]


from .cn import PROFILE_CN

PROFILE = DefaultProfile()

__all__ = ['Profile', 'PROFILE', 'PROFILE_CN']
