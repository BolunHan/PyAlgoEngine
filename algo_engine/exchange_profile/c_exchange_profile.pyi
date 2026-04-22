import datetime
import enum
from datetime import timedelta
from typing import Self, overload, override

__all__ = [
    "SessionType",
    "SessionPhase",
    "AuctionPhase",
    "SessionTime",
    "SessionTimeRange",
    "SessionDate",
    "SessionDateRange",
    "SessionDateTime",
    "CallAuction",
    "SessionBreak",
    "ExchangeProfile",

    'PROFILE',
    'PROFILE_DEFAULT',
    'PROFILE_CN'
]


class SessionType(enum.IntEnum):
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


class SessionPhase(enum.IntEnum):
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


class AuctionPhase(enum.IntEnum):
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


def unix_to_datetime(unix_ts: float) -> datetime.datetime:
    """Convert a UNIX timestamp to a timezone-aware datetime in the local timezone.

    This function is used internally for timestamp conversions in the exchange
    profile. It takes a UNIX timestamp (seconds since epoch) and returns a
    datetime object that is aware of the local timezone, which is necessary
    for accurate conversions to session times.

    Args:
        unix_ts: The UNIX timestamp to convert.
    Returns:
        A timezone-aware datetime object corresponding to the provided timestamp in the local timezone.
    Raises:
        ValueError: If the provided timestamp is negative or otherwise invalid.
    """
    ...


TimeLike = SessionTime | datetime.time


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
    def from_pytime(cls, t: datetime.time) -> Self: ...

    @classmethod
    def from_ts(cls, ts: float) -> Self:
        """Create a SessionTime from a seconds elapsed since midnight timestamp.

        Args:
            ts: Seconds since midnight (can be fractional).
        Returns:
            SessionTime instance corresponding to the provided timestamp.
        Raises:
            ValueError: If the timestamp is negative or exceeds the number of seconds in a day.
        """
        ...

    @classmethod
    def from_timestamp(cls, unix_ts: float) -> Self: ...

    def to_pytime(self) -> datetime.time: ...

    def isoformat(self, *args, **kwargs) -> str: ...

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


DateLike = SessionDate | datetime.date


class SessionDate(datetime.date):
    """Represents a calendar date in the exchange profile.

    ``SessionDate`` subclasses ``datetime.date`` and mirrors a C-backed
    ``session_date_t`` header used by exchange-profile operations. Because
    ``datetime.date`` allocates its date payload in ``__new__`` at the C level,
    a ``SessionDate`` can exist in two distinct states depending on whether
    ``__init__`` was executed.

    Fully initialized (bound):
        Both the Python date payload and the internal C header pointer are
        valid. Created via ``SessionDate(year, month, day)`` or any class
        factory (``from_pydate``, ``from_unix``, ``from_ordinal``,
        ``fromisoformat``, ``fromisocalendar``, ``today``)::

            >>> sd = SessionDate(2024, 11, 11)
            >>> repr(sd)
            '<SessionDate>(2024-11-11)'
            >>> sd.addr
            '0x55b3aff65a10'

    Partially initialized (unbound):
        The Python date payload is valid (``year``/``month``/``day`` readable),
        but the internal C header pointer is null. Arises when only ``__new__``
        runs without ``__init__``, for example via ``SessionDate.__new__(...)``
        or inherited ``datetime.date`` classmethods that are not overridden::

            >>> partial_sd = SessionDate.__new__(SessionDate, 2024, 11, 11)
            >>> repr(partial_sd)
            '<SessionDate Unbound>(2024-11-11)'
            >>> partial_sd.addr
            '0x0'

    For an unbound SessionDate instance, calling any method that relies on the internal C header
    (e.g. ``to_pydate()``, ``to_ordinal()``, ``is_valid()``, ``session_type``, etc.) will automatically call ``c_sync()`` cython internal method,
    which will attempt to initialize the C header based on the Python date payload.

    Unbound instance has a different __hash__ value, dict lookup will treat it as a different key.
    For a consistent behavior, it is recommended to always use the fully initialized (bound) SessionDate instances.
    The only known way to create unbound instances is via ``SessionDate.__new__``. All other methods are tested to return fully bound instances.
    Use ``test.exchange_profile.test_03_c_exchange_profile_date_initialization_contract`` to verify the expected behavior across different python major versions.
    ``datetime.date`` API is not within Limited-ABI, and is subjected to change across python versions, use with caution.

    ``SessionDateEx`` provides a implementation of SessionDate that does not rely on ``datetime.date``.
    For cross-platform / version programming, it may be safer to use ``SessionDateEx`` instead of ``SessionDate``.
    """

    def __init__(self, year: int, month: int, day: int) -> None: ...

    def __hash__(self) -> int: ...

    @overload
    def __sub__(self, other: timedelta) -> SessionDate: ...

    @overload
    def __sub__(self, other: DateLike) -> SessionDateRange: ...

    def __sub__(self, other): ...

    def __rsub__(self, other: DateLike) -> SessionDateRange: ...

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
    def today(cls) -> SessionDate: ...

    @classmethod
    def from_unix(cls, unix_ts: float) -> SessionDate: ...

    @classmethod
    def from_ordinal(cls, ordinal: int) -> SessionDate: ...

    @classmethod
    def from_pydate(cls, dt: datetime.date) -> SessionDate: ...

    def to_pydate(self) -> datetime.date: ...

    def to_ordinal(self) -> int: ...

    def add_days(self, days: int) -> SessionDate: ...

    def is_valid(self) -> bool: ...

    def is_weekend(self) -> bool: ...

    def fork(self) -> SessionDate:
        """Create a new SessionDate instance with the same underlying C struct (not owned), but a separate Python wrapper."""
        ...

    @classmethod
    def fromisocalendar(cls, year: int, week: int, weekday: int) -> SessionDate: ...

    @classmethod
    def fromisoformat(cls, date_str: str) -> SessionDate: ...

    def isocalendar(self) -> tuple[int, int, int]: ...

    def isoformat(self, *args, **kwargs) -> str: ...

    def strftime(self, format: str) -> str: ...

    def weekday(self) -> int: ...

    def timestamp(self) -> float:
        """Return the UNIX timestamp (seconds since epoch) corresponding to this date at midnight in the profile's local time zone."""
        ...

    @property
    def year(self) -> int: ...

    @property
    def month(self) -> int: ...

    @property
    def day(self) -> int: ...

    @property
    def session_type(self) -> SessionType: ...

    @property
    def addr(self) -> str:
        """Return the memory address of the underlying C struct as a hex string (for debugging purposes)."""
        ...


DateLikeStandalone = SessionDateEx | DateLike


class SessionDateEx(object):
    """A standalone implementation of SessionDate that does not rely on datetime.date.
    Most of the signature is the same as ``SessionDate``, but without ``fork()`` and ``.addr`` property.
    And it is guaranteed to have valid ``session_date_t* header`` initialized.
    """

    def __init__(self, year: int, month: int, day: int) -> None: ...

    def __hash__(self) -> int: ...

    def __format__(self, format_spec: str) -> str: ...

    @overload
    def __sub__(self, other: timedelta) -> SessionDateEx: ...

    @overload
    def __sub__(self, other: DateLikeStandalone) -> SessionDateRange: ...

    def __sub__(self, other): ...

    def __rsub__(self, other: DateLikeStandalone) -> SessionDateRange: ...

    def __add__(self, other: timedelta) -> SessionDateEx: ...

    def __iadd__(self, other: timedelta) -> Self: ...

    def __lt__(self, other: DateLikeStandalone) -> bool: ...

    def __le__(self, other: DateLikeStandalone) -> bool: ...

    def __eq__(self, other: DateLikeStandalone) -> bool: ...

    def __ne__(self, other: DateLikeStandalone) -> bool: ...

    def __gt__(self, other: DateLikeStandalone) -> bool: ...

    def __ge__(self, other: DateLikeStandalone) -> bool: ...

    @staticmethod
    def is_leap_year(year: int) -> bool: ...

    @staticmethod
    def days_in_month(year: int, month: int) -> int: ...

    @classmethod
    def today(cls) -> SessionDateEx: ...

    @classmethod
    def from_unix(cls, unix_ts: float) -> SessionDateEx: ...

    @classmethod
    def from_ordinal(cls, ordinal: int) -> SessionDateEx: ...

    @classmethod
    def from_pydate(cls, dt: datetime.date) -> SessionDateEx: ...

    def to_pydate(self) -> datetime.date: ...

    def to_ordinal(self) -> int: ...

    def add_days(self, days: int) -> SessionDateEx: ...

    def is_valid(self) -> bool: ...

    def is_weekend(self) -> bool: ...

    def fork(self) -> SessionDateEx:
        """Create a new SessionDateEx instance with the same underlying C struct (not owned), but a separate Python wrapper."""
        ...

    def ctime(self) -> str: ...

    @classmethod
    def fromisocalendar(cls, year: int, week: int, weekday: int) -> SessionDateEx: ...

    @classmethod
    def fromisoformat(cls, date_str: str) -> SessionDateEx: ...

    def isocalendar(self) -> tuple[int, int, int]: ...

    def isoformat(self, *args, **kwargs) -> str: ...

    def strftime(self, format: str) -> str: ...

    def weekday(self) -> int: ...

    def timestamp(self) -> float: ...

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

    def __contains__(self, item: DateLike) -> bool: ...

    def __len__(self) -> int: ...

    def index(self, item: DateLike) -> int: ...

    def to_list(self) -> list[datetime.date]:
        """Convert the SessionDateRange to a list of datetime.date objects.

        Returns:
            List[datetime.date]: A list of datetime.date instances corresponding to the session dates in the range.
        """
        ...

    @property
    def n_days(self) -> int: ...

    @property
    def start_date(self) -> SessionDate: ...

    @property
    def end_date(self) -> SessionDate: ...

    @property
    def dates(self) -> tuple[SessionDate, ...]: ...


class SessionDateTime(object):
    """A combined date and time representation for exchange sessions.
    Simplest wrapper around a C struct that contains a SessionDate and SessionTime.
    """

    @classmethod
    def from_pydatetime(cls, t: datetime.datetime) -> SessionDateTime: ...

    def to_pydatetime(self) -> datetime.datetime: ...

    def fork(self):
        """Create a new SessionDateTime instance with the same underlying C struct (not owned), but a separate Python wrapper."""
        ...

    def update(self, unix_ts: float) -> Self:
        """Update the date and time based on a new UNIX timestamp, modifying the current instance in-place.

        This method allows reusing the same SessionDateTime wrapper to point to different instants in time by updating its internal C struct with a new timestamp. The date and time properties will reflect the new timestamp after the update.

        Args:
            unix_ts: The new UNIX timestamp to update the SessionDateTime to.
        Returns:
            Self: The updated SessionDateTime instance (same wrapper, updated internal state).
        """
        ...

    @classmethod
    def from_unix(cls, unix_ts: float) -> SessionDateTime: ...

    @property
    def time(self) -> SessionTime: ...

    @property
    def date(self) -> SessionDateEx: ...

    @property
    def timestamp(self) -> float:
        """Return the cached UNIX timestamp."""
        ...

    @property
    def ts(self) -> float:
        """Return the cached total seconds since midnight."""
        ...

    @property
    def ordinal(self) -> int:
        """Return the cached ordinal of the date."""
        ...


class CallAuction(object):
    """Call auction metadata describing auction windows inside a trading day.

    Lightweight wrapper around the C `call_auction` structure.
    """

    @property
    def auction_start(self) -> SessionTime:
        """The scheduled auction start time."""
        ...

    @property
    def active(self) -> SessionTimeRange | None:
        """Active matching window for the auction, or ``None`` if not present."""
        ...

    @property
    def no_cancel(self) -> SessionTimeRange | None:
        """No-cancel window during the auction, or ``None`` if not present."""
        ...

    @property
    def frozen(self) -> SessionTimeRange | None:
        """Frozen period for the auction, or ``None`` if not present."""
        ...

    @property
    def uncross(self) -> SessionTime:
        """The uncrossing instant as a :class:`SessionTime`."""
        ...

    @property
    def auction_end(self) -> SessionTime:
        """The scheduled auction end time."""
        ...


class SessionBreak(object):
    """Represents a break interval inside a session.

    Lightweight wrapper around the C ``session_break`` linked-list node.
    """

    @property
    def break_start(self) -> SessionTime:
        """Start time of the break."""
        ...

    @property
    def break_end(self) -> SessionTime:
        """End time of the break."""
        ...

    @property
    def break_start_ts(self) -> float:
        """Cached timestamp (seconds from midnight) for break start."""
        ...

    @property
    def break_end_ts(self) -> float:
        """Cached timestamp (seconds from midnight) for break end."""
        ...

    @property
    def break_length_seconds(self) -> float:
        """Precomputed break length in seconds."""
        ...

    @property
    def next(self) -> SessionBreak | None:
        """Link to the next break, or ``None``."""
        ...


TS_LIKE = datetime.time | float | int


class ExchangeProfile(object):
    """Represents an exchange profile with session configuration and helpers.

    This object mirrors the C ``exchange_profile`` struct and provides
    convenience Python methods that dispatch to the underlying C function
    pointers. Instances are typically obtained from module-level globals
    ``PROFILE_DEFAULT`` and ``PROFILE_CN`` or via the :meth:`c_from_header`
    factory.

    Note: Many methods accept both Python-level wrappers (``SessionDate``/
    ``SessionTime``) and their standard-library counterparts (``datetime.date``/
    ``datetime.time``) for convenience.
    """

    def __repr__(self) -> str: ...

    def activate(self) -> None:
        """Activate this profile as the global exchange profile for the process/thread.

        This calls into the C-level activation callback and sets the global
        profile pointer used by other APIs.
        """
        ...

    def deactivate(self) -> None:
        """Deactivate this profile and restore the default profile.

        If this profile is the currently active one the implementation will
        switch back to ``PROFILE_DEFAULT``.
        """
        ...

    def trade_calendar(self, start_date: DateLike, end_date: DateLike) -> SessionDateRange:
        """Return a :class:`SessionDateRange` containing trading dates between two dates.

        Args:
            start_date: inclusive start; may be a :class:`SessionDate` or :class:`datetime.date`.
            end_date: inclusive end; may be a :class:`SessionDate` or :class:`datetime.date`.

        Returns:
            SessionDateRange: C-backed range object with ``dates`` sequence of :class:`SessionDate`.
        """
        ...

    def resolve_auction_phase(self, session_time: TimeLike) -> AuctionPhase:
        """Resolve auction-phase at the provided time-of-day.

        Args:
            session_time: :class:`SessionTime` or :class:`datetime.time`.

        Returns:
            AuctionPhase: an enum value representing the auction sub-phase.
        """
        ...

    def resolve_session_phase(self, session_time: TimeLike) -> SessionPhase:
        """Resolve high-level session phase at the provided time-of-day (preopen/open/continuous/etc.)."""
        ...

    def resolve_session_type(self, session_date: DateLike) -> SessionType:
        """Return the :class:`SessionType` for the provided market date.

        Args:
            session_date: :class:`SessionDate` or :class:`datetime.date`.
        """
        ...

    def timestamp_to_datetime(self, unix_ts: float) -> datetime.datetime:
        """Convert a UNIX timestamp to a timezone-aware datetime in the profile's time zone."""
        ...

    def time_to_seconds(self, t: datetime.time, break_adjusted: bool = True) -> float:
        """Convert a time-of-day to seconds since trading session start.

        Args:
            t: datetime.time instance representing a clock time in the profile's local zone.
            break_adjusted: If True, subtract non-trading break durations (e.g., lunch break).

        Returns:
            float: elapsed seconds since session open (breaks excluded when requested) or
            raw seconds-of-day if break_adjusted is False.
        """
        ...

    def timestamp_to_seconds(self, t: float, break_adjusted: bool = True) -> float:
        """Convert a UNIX timestamp (seconds) to seconds since session open, optionally break-adjusted."""
        ...

    def break_adjusted(self, elapsed_seconds: float) -> float:
        """Return the break-adjusted elapsed seconds for a raw seconds-since-midnight value."""
        ...

    def trading_time_between(self, start_time: TS_LIKE, end_time: TS_LIKE) -> float:
        """Compute total trading seconds between two datetime (or UNIX timestamps).

        Accepts either naive datetime (interpreted in profile.time_zone) or numeric
        UNIX timestamps. The result is the total trading-time (seconds) excluding
        non-trading periods (e.g., lunch break).
        """
        ...

    def is_market_session(self, timestamp) -> bool:
        """Return True if the provided timestamp/time/datetime is in continuous market session.

        Accepts numeric UNIX timestamps, :class:`datetime.time` or :class:`datetime.datetime`.
        """
        ...

    def is_auction_session(self, timestamp) -> bool:
        """Return True if the provided timestamp/time/datetime is in an auction period."""
        ...

    def trading_days_before(self, market_date: datetime.date, days: int) -> datetime.date:
        """Return the market date that is a given number of trading days before the provided date.

        If the provided date is not a trading day, it will be treated as if it were the nearest **NEXT** trading day.
        That is, if days == 1 and the provided date is SAT or SUN, the result will be FRI, not THU.
        """
        ...

    def trading_days_after(self, market_date: datetime.date, days: int) -> datetime.date:
        """Return the market date that is a given number of trading days after the provided date.

        If the provided date is not a trading day, it will be treated as if it were the nearest **PREVIOUS** trading day.
        That is, if days == 1 and the provided date is SAT or SUN, the result will be MON, not TUE.
        """
        ...

    def trading_days_between(self, start_date: datetime.date, end_date: datetime.date) -> int:
        """Return the number of trading days between two market dates (inclusive)."""
        ...

    def nearest_trading_date(self, market_date: datetime.date, method: str = 'previous') -> datetime.date:
        """Return the nearest trading date to the provided date.

        Args:
            market_date: The reference date for which to find the nearest trading date.
            method: Method to resolve ties when the provided date is exactly between two trading days. Options are 'previous' (default) or 'next'.

        Returns:
            datetime.date: The nearest trading date to the provided date.
        """
        ...

    def is_trading_day(self, market_date: datetime.date) -> bool: ...

    @property
    def profile_id(self) -> str:
        """Unique identifier for the profile (e.g. "UTC_NONSTOP_DEFAULT" (the default one), "CN_STOCK")."""
        ...

    @property
    def session_start(self) -> SessionTime:
        """Continuous session start time."""
        ...

    @property
    def session_end(self) -> SessionTime:
        """Continuous session end time."""
        ...

    @property
    def open_call_auction(self) -> CallAuction | None:
        """Open call auction metadata (maybe ``None``)."""
        ...

    @property
    def close_call_auction(self) -> CallAuction | None:
        """Close call auction metadata (maybe ``None``)."""
        ...

    @property
    def session_breaks(self) -> tuple[SessionBreak, ...]:
        """Tuple of :class:`SessionBreak` describing intraday breaks."""
        ...

    @property
    def time_zone(self) -> datetime.timezone:
        """IANA time zone as a :class:`zoneinfo.datetime.timezone` instance.

        The live implementation returns a datetime.timezone object which can be used
        to construct timezone-aware datetime for timestamp conversions.
        """
        ...

    @property
    def session_start_ts(self) -> float: ...

    @property
    def session_end_ts(self) -> float: ...

    @property
    def session_length_seconds(self) -> float: ...

    @property
    def tz_offset_seconds(self) -> float: ...

    @property
    def range_break(self) -> list[dict]:
        """Return a list of range break dicts for Plotly X-axis configuration based on the session breaks of the currently active exchange profile."""
        ...

    @property
    def trade_calendar_cache(self) -> SessionDateRange | None:
        """Return a cached SessionDateRange for the current active exchange profile, or None."""
        ...


PROFILE: ExchangeProfile
PROFILE_DEFAULT: ExchangeProfile
PROFILE_CN: ExchangeProfile
