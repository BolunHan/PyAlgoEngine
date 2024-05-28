import abc
import datetime


class Profile(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            session_start: datetime.time | None = None,
            session_end: datetime.time | None = None,
            session_break: tuple[datetime.time, datetime.time] | None = None
    ):
        self.session_start: datetime.time | None = session_start
        self.session_end: datetime.time | None = session_end
        self.session_break: tuple[datetime.time, datetime.time] | None = session_break

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

    def trade_time_between(self, start_time: datetime.datetime | float, end_time: datetime.datetime | float, **kwargs) -> datetime.timedelta:
        if start_time is not None and isinstance(start_time, (float, int)):
            start_time = datetime.datetime.fromtimestamp(start_time)

        if end_time is not None and isinstance(end_time, (float, int)):
            end_time = datetime.datetime.fromtimestamp(end_time)

        if start_time is None or end_time is None:
            return datetime.timedelta(seconds=0)

        if start_time > end_time:
            return datetime.timedelta(seconds=0)

        return end_time - start_time

    def in_trade_session(self, market_time: datetime.datetime | float) -> bool:
        return True


from .cn import CN_Profile

__all__ = ['Profile', 'DefaultProfile', 'CN_Profile']
