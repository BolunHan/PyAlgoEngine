from __future__ import annotations

from datetime import time as py_time
from enum import IntEnum
from typing import Any, Self

__all__ = ["SessionType", "SessionPhase", "AuctionPhase", "SessionTime"]


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

    # ----- Rich comparisons -----
    def __lt__(self, other: object) -> bool: ...

    def __le__(self, other: object) -> bool: ...

    def __eq__(self, other: object) -> bool: ...

    def __ne__(self, other: object) -> bool: ...

    def __gt__(self, other: object) -> bool: ...

    def __ge__(self, other: object) -> bool: ...

    def __repr__(self) -> str: ...
