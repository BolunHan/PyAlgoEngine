import abc
import datetime
from typing import Self


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
