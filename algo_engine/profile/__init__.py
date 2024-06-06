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

        self.timezone = None

    @abc.abstractmethod
    def override_profile(self, profile: Self): ...

    @abc.abstractmethod
    def trade_time_between(self, start_time: datetime.datetime | float, end_time: datetime.datetime | float, **kwargs) -> datetime.timedelta:
        ...

    @abc.abstractmethod
    def in_trade_session(self, market_time: datetime.datetime | float) -> bool:
        ...


class DefaultProfile(Profile):
    def __init__(self):
        super().__init__(
            session_start=datetime.time(0),
            session_end=None,
            session_break=None
        )

    def override_profile(self, profile: Self):
        pass

    def trade_time_between(self, start_time: datetime.datetime | float, end_time: datetime.datetime | float, **kwargs) -> datetime.timedelta:
        if start_time is not None and isinstance(start_time, (float, int)):
            start_time = datetime.datetime.fromtimestamp(start_time, tz=self.timezone)

        if end_time is not None and isinstance(end_time, (float, int)):
            end_time = datetime.datetime.fromtimestamp(end_time, tz=self.timezone)

        if start_time is None or end_time is None:
            return datetime.timedelta(seconds=0)

        if start_time > end_time:
            return datetime.timedelta(seconds=0)

        return end_time - start_time

    def in_trade_session(self, market_time: datetime.datetime | float) -> bool:
        return True


from .cn import PROFILE_CN

PROFILE = DefaultProfile()

__all__ = ['Profile', 'PROFILE', 'PROFILE_CN']
