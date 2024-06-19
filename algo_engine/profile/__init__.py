import abc
import datetime
from typing import Self


class Profile(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            session_start: datetime.time | None = None,
            session_end: datetime.time | None = None,
            session_break: list[tuple[datetime.time, datetime.time]] = None
    ):
        self.session_start = session_start
        self.session_end = session_end
        self.session_break = [] if session_break is None else session_break

        self.time_zone = None

    @abc.abstractmethod
    def override_profile(self, profile: Self = None): ...

    @abc.abstractmethod
    def trade_time_between(self, start_time: datetime.datetime | float, end_time: datetime.datetime | float, **kwargs) -> datetime.timedelta:
        ...

    @abc.abstractmethod
    def is_market_session(self, timestamp: float | int | datetime.datetime) -> bool:
        ...

    @property
    def range_break(self) -> list[dict]:
        """
        an range break designed for plotly.
        """
        return []


class DefaultProfile(Profile):
    def __init__(self):
        super().__init__(
            session_start=datetime.time(0),
            session_end=None,
            session_break=None
        )

    def override_profile(self, profile: Self = None):
        pass

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

    def is_market_session(self, timestamp: float | int | datetime.datetime) -> bool:
        return True


from .cn import PROFILE_CN

PROFILE = DefaultProfile()

__all__ = ['Profile', 'PROFILE', 'PROFILE_CN']
