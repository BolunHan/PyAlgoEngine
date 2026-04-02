from __future__ import annotations

from datetime import time as py_time, date as py_date
from enum import IntEnum
from typing import Any, Self

__all__ = ["SessionType", "SessionPhase", "AuctionPhase", "SessionTime", "SessionTimeRange", "SessionDate", "SessionDateRange"]


class SessionType(IntEnum):
    """Describes the type of trading session for a given market date.

    The integer values map directly to the underlying C enum constants used
    by the exchange profile implementation. Use these values to inspect or
    annotate calendar/session types.

    Attributes
    ----------
    NON_TRADING: int
        Market is closed for the date.
    NORMINAL: int
        Normal trading day.
    SUSPENDED: int
        Trading suspended for the date.
    HALF_DAY: int
        Half trading day.
    CIRCUIT_BREAK: int
        Trading day with a circuit breaker in effect.
    """

    NON_TRADING: SessionType
    NORMINAL: SessionType
    SUSPENDED: SessionType
    HALF_DAY: SessionType
    CIRCUIT_BREAK: SessionType


class SessionPhase(IntEnum):
    """The phase of the trading session at a particular time-of-day.

    The values correspond to the phases used inside the C exchange profile
    code (e.g. PREOPEN, CONTINUOUS, BREAK, CLOSED, ...). The value can be
    retrieved from a :class:`SessionTime` via the ``session_phase`` property.
    """

    UNKNOWN: SessionPhase
    PREOPEN: SessionPhase
    OPEN_AUCTION: SessionPhase
    CONTINUOUS: SessionPhase
    BREAK: SessionPhase
    SUSPENDED: SessionPhase
    CLOSE_AUCTION: SessionPhase
    CLOSED: SessionPhase


class AuctionPhase(IntEnum):
    """Auction-specific phase during call auctions.

    These values map to the auction phases used by the exchange profile.
    """

    ACTIVE: AuctionPhase
    NO_CANCEL: AuctionPhase
    FROZEN: AuctionPhase
    UNCROSSING: AuctionPhase
    DONE: AuctionPhase


def local_utc_offset_seconds() -> int:
    """Returns the local UTC offset in seconds for the current system timezone.

    This is used internally to adjust timestamps when converting between
    unix timestamps and session times, since the exchange profile operates
    in local time. The function computes the offset by comparing the current
    local time to UTC time.

    Returns
    -------
    int
        Local UTC offset in seconds (e.g. 28800 for UTC/GMT +8 hours).
    """
    ...


TimeLike = SessionTime | py_time


class SessionTime(object):
    """Represents a time-of-day within an exchange session.

    This class is a lightweight wrapper around a C struct (``session_time_t``)
    and exposes a Python-friendly API for construction, inspection, and
    comparisons. Instances are typically created by calling the class with
    hour/minute/second/nanosecond arguments, or by converting from
    ``datetime.time`` via ``from_pytime``.

    The object is immutable from the Python-level: its fields are exposed as
    read-only properties. The underlying implementation caches a floating
    seconds-since-midnight timestamp used to compute ``elapsed_seconds`` when
    an exchange profile is active.

    Example
    -------
    >>> st = SessionTime(9, 30, 0)
    >>> st.hour
    9

    Parameters
    ----------
    hour : int, optional
        Hour in 24-hour clock (0-23). Default is 0.
    minute : int, optional
        Minute (0-59). Default is 0.
    second : int, optional
        Second (0-59). Default is 0.
    nanosecond : int, optional
        Nanoseconds portion (0-999_999_999). Default is 0.
    """

    def __init__(self, hour: int = 0, minute: int = 0, second: int = 0, nanosecond: int = 0) -> None: ...

    @classmethod
    def from_pytime(cls, t: py_time) -> Self: ...

    @classmethod
    def from_timestamp(cls, unix_ts: float) -> Self: ...

    def to_pytime(self) -> py_time: ...

    def isoformat(self, *args: Any, **kwargs: Any) -> str: ...

    @classmethod
    def fromisoformat(cls, time_str: str) -> Self: ...

    # ----- Read-only properties -----
    @property
    def hour(self) -> int: ...

    @property
    def minute(self) -> int: ...

    @property
    def second(self) -> int: ...

    @property
    def microsecond(self) -> int: ...

    @property
    def nanosecond(self) -> int: ...

    @property
    def elapsed_seconds(self) -> float: ...

    @property
    def ts(self) -> float: ...

    @property
    def session_phase(self) -> SessionPhase: ...

    def __lt__(self, other: TimeLike) -> bool: ...

    def __le__(self, other: TimeLike) -> bool: ...

    def __eq__(self, other: TimeLike) -> bool: ...

    def __ne__(self, other: TimeLike) -> bool: ...

    def __gt__(self, other: TimeLike) -> bool: ...

    def __ge__(self, other: TimeLike) -> bool: ...

    def __sub__(self, other: TimeLike) -> SessionTimeRange: ...


class SessionTimeRange(object):
    """Represents a duration between two :class:`SessionTime` instances.

    The runtime type is a C-backed struct wrapper which contains an internal
    header with fields ``start``, ``end`` and ``elapsed_seconds``. Instances
    are usually created via subtraction (``end - start``) or by the C APIs.
    """

    def __init__(self, start_time: SessionTime, end_time: SessionTime) -> None: ...

    @property
    def elapsed_seconds(self) -> float: ...

    @property
    def start_time(self) -> SessionTime: ...

    @property
    def end_time(self) -> SessionTime: ...


DateLike = SessionDate | py_date


class SessionDate(object):
    """Represents a calendar date in the exchange profile.

    The class mirrors the C-backed ``session_date_t`` structure and provides
    construction, conversion, arithmetic and validation helpers.
    """

    def __init__(self, year: int, month: int, day: int) -> None: ...

    def __hash__(self) -> int: ...

    def __format__(self, format_spec: str) -> str: ...

    def __sub__(self, other: DateLike) -> SessionDateRange: ...

    def __lt__(self, other: DateLike) -> bool: ...

    def __le__(self, other: DateLike) -> bool: ...

    def __eq__(self, other: DateLike) -> bool: ...

    def __ne__(self, other: DateLike) -> bool: ...

    def __gt__(self, other: DateLike) -> bool: ...

    def __ge__(self, other: DateLike) -> bool: ...

    @staticmethod
    def is_leap_year(year: int) -> bool: ...

    @staticmethod
    def days_in_month(year: int, month: int) -> int: ...

    @classmethod
    def from_ts(cls, unix_ts: float) -> SessionDate: ...

    @classmethod
    def from_ordinal(cls, ordinal: int) -> SessionDate: ...

    @classmethod
    def from_pydate(cls, dt: py_date) -> SessionDate: ...

    def to_pydate(self) -> py_date: ...

    def to_ordinal(self) -> int: ...

    def add_days(self, days: int) -> SessionDate: ...

    def is_valid(self) -> bool: ...

    def ctime(self) -> str: ...

    @classmethod
    def fromisocalendar(cls, year: int, week: int, weekday: int) -> SessionDate: ...

    @classmethod
    def fromisoformat(cls, date_str: str) -> SessionDate: ...

    def isocalendar(self) -> tuple[int, int, int]: ...

    def isoformat(self, *args, **kwargs) -> str: ...

    def strftime(self, format: str) -> str: ...

    def weekday(self) -> int: ...

    @property
    def year(self) -> int: ...

    @property
    def month(self) -> int: ...

    @property
    def day(self) -> int: ...

    @property
    def session_type(self) -> SessionType: ...


class SessionDateRange(object):
    """Represents a contiguous range of :class:`SessionDate` instances.

    Constructed from a start and end :class:`SessionDate`. Exposes length via
    ``n_days``, sequence access and iteration.
    """

    def __init__(self, start_date: SessionDate, end_date: SessionDate) -> None: ...

    def __iter__(self): ...

    def __getitem__(self, idx: int) -> SessionDate: ...

    @property
    def n_days(self) -> int: ...

    @property
    def start_date(self) -> SessionDate: ...

    @property
    def end_date(self) -> SessionDate: ...

    @property
    def dates(self) -> tuple[SessionDate, ...]: ...
