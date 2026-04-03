import enum

from cpython.datetime cimport time as py_time, date as py_date
from cpython.object cimport Py_LT, Py_LE, Py_EQ, Py_NE, Py_GT, Py_GE
from cpython.unicode cimport PyUnicode_FromString
from libc.stdlib cimport calloc, free


class SessionType(enum.IntEnum):
    NON_TRADING = session_type.SESSION_TYPE_NON_TRADING
    NORMINAL = session_type.SESSION_TYPE_NORMINAL
    SUSPENDED = session_type.SESSION_TYPE_SUSPENDED
    HALF_DAY = session_type.SESSION_TYPE_HALF_DAY
    CIRCUIT_BREAK = session_type.SESSION_TYPE_CIRCUIT_BREAK


class SessionPhase(enum.IntEnum):
    UNKNOWN = session_phase.SESSION_PHASE_UNKNOWN
    PREOPEN = session_phase.SESSION_PHASE_PREOPEN
    OPEN_AUCTION = session_phase.SESSION_PHASE_OPEN_AUCTION
    CONTINUOUS = session_phase.SESSION_PHASE_CONTINUOUS
    BREAK = session_phase.SESSION_PHASE_BREAK
    SUSPENDED = session_phase.SESSION_PHASE_SUSPENDED
    CLOSE_AUCTION = session_phase.SESSION_PHASE_CLOSE_AUCTION
    CLOSED = session_phase.SESSION_PHASE_CLOSED


class AuctionPhase(enum.IntEnum):
    ACTIVE = auction_phase.AUCTION_PHASE_ACTIVE
    NO_CANCEL = auction_phase.AUCTION_PHASE_NO_CANCEL
    FROZEN = auction_phase.AUCTION_PHASE_FROZEN
    UNCROSSING = auction_phase.AUCTION_PHASE_UNCROSSING
    DONE = auction_phase.AUCTION_PHASE_DONE


def local_utc_offset_seconds():
    return c_utc_offset_seconds()


cdef class SessionTime:
    def __cinit__(self, uint8_t hour=0, uint8_t minute=0, uint8_t second=0, uint32_t nanosecond=0, *, bint no_alloc=False):
        if no_alloc:
            return
        self.header = c_ex_profile_session_time_new(hour, minute, second, nanosecond)
        if not self.header:
            raise MemoryError(f"Failed to allocate memory for {self.__class__.__name__}")
        self.owner = True

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            free(<void*> self.header)

    @staticmethod
    cdef SessionTime c_from_header(const session_time_t* header, bint owner):
        cdef SessionTime session_time = SessionTime.__new__(SessionTime, no_alloc=True)
        session_time.header = header
        session_time.owner = owner
        return session_time

    # === Python Interfaces ===

    def __repr__(self):
        return f"<{self.__class__.__name__}>({self.to_pytime().__str__()})"

    def __sub__(self, object other):
        if isinstance(other, SessionTime):
            return SessionTimeRange(other, self)
        elif isinstance(other, py_time):
            return SessionTimeRange(SessionTime.from_pytime(other), self)
        else:
            raise TypeError(f'Can not subtract {self.__class__.__name__} with {other.__class__.__name__}')

    def __richcmp__(self, object other, int op):
        cdef int cmp
        if isinstance(other, SessionTime):
             cmp = c_ex_profile_time_compare(self.header, (<SessionTime> other).header)
        elif isinstance(other, py_time):
            cmp = c_ex_profile_time_compare(self.header, (<SessionTime> SessionTime.from_pytime(other)).header)
        else:
            raise TypeError(f'Can not compare {self.__class__.__name__} with {other.__class__.__name__}')

        if op == Py_LT:
            return cmp < 0
        elif op == Py_LE:
            return cmp <= 0
        elif op == Py_EQ:
            return cmp == 0
        elif op == Py_NE:
            return cmp != 0
        elif op == Py_GT:
            return cmp > 0
        elif op == Py_GE:
            return cmp >= 0
        else:
            raise ValueError(f'Invalid comparison operator: {op}')

    @classmethod
    def from_pytime(cls, py_time t):
        cdef SessionTime instance = SessionTime.__new__(SessionTime, t.hour, t.minute, t.second, t.microsecond * 1000)
        return instance

    @classmethod
    def from_timestamp(cls, double unix_ts):
        cdef SessionTime instance = SessionTime.__new__(SessionTime)
        if c_ex_profile_session_time_from_ts(unix_ts, <session_time_t*> instance.header) != 0:
            raise ValueError(f"Invalid timestamp: {unix_ts}")
        return instance

    def to_pytime(self):
        return py_time(self.header.hour, self.header.minute, self.header.second, self.header.nanosecond // 1000)

    def isoformat(self, *args, **kwargs):
        return self.to_pytime().isoformat(*args, **kwargs)

    @classmethod
    def fromisoformat(cls, str time_str):
        return cls.from_pytime(py_time.fromisoformat(time_str))

    property hour:
        def __get__(self):
            return self.header.hour

    property minute:
        def __get__(self):
            return self.header.minute

    property second:
        def __get__(self):
            return self.header.second

    property microsecond:
        def __get__(self):
            return self.header.nanosecond // 1000

    property nanosecond:
        def __get__(self):
            return self.header.nanosecond

    property elapsed_seconds:
        def __get__(self):
            return self.header.elapsed_seconds

    property ts:
        def __get__(self):
            return c_ex_profile_time_to_ts(self.header.hour, self.header.minute, self.header.second, self.header.nanosecond)

    property session_phase:
        def __get__(self):
            return SessionPhase(self.header.phase)


cdef class SessionTimeRange:
    def __cinit__(self, SessionTime start_time, SessionTime end_time, *, bint no_alloc=False):
        if no_alloc:
            return
        self.header = c_ex_profile_session_trange_between_time(start_time.header, end_time.header)
        if not self.header:
            raise MemoryError(f"Failed to allocate memory for {self.__class__.__name__}")
        self.owner = True
        self.start_time = SessionTime.c_from_header(&self.header.start, False)
        self.end_time = SessionTime.c_from_header(&self.header.end, False)

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            free(<void*> self.header)

    @staticmethod
    cdef SessionTimeRange c_from_header(const session_time_range_t* header, bint owner):
        cdef SessionTimeRange instance = SessionTimeRange.__new__(SessionTimeRange, None, None, no_alloc=True)
        instance.header = header
        instance.owner = owner
        instance.start_time = SessionTime.c_from_header(&header.start, False)
        instance.end_time = SessionTime.c_from_header(&header.end, False)
        return instance

    # === Python Interfaces ===

    def __repr__(self):
        return f"<{self.__class__.__name__}>(elapsed_seconds={self.header.elapsed_seconds:,.3f}, start={self.start_time.to_pytime()}, end={self.end_time.to_pytime()})"

    property elapsed_seconds:
        def __get__(self):
            return self.header.elapsed_seconds


cdef class SessionDate:
    def __cinit__(self, uint16_t year, uint8_t month, uint8_t day, *, bint no_alloc=False):
        if no_alloc:
            return
        self.header = c_ex_profile_session_date_new(year, month, day)
        if not self.header:
            raise MemoryError(f'Failed to allocate memory for {self.__class__.__name__}')
        self.owner = True

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            free(<void*> self.header)

    @staticmethod
    cdef SessionDate c_from_header(const session_date_t* header, bint owner):
        cdef SessionDate instance = SessionDate.__new__(SessionDate, 0, 0, 0, no_alloc=True)
        instance.header = header
        instance.owner = owner
        return instance

    # === Python Interfaces ===

    def __repr__(self):
        return f'<{self.__class__.__name__}>({self.header.year:4d}-{self.header.month:02d}-{self.header.day:02d})'

    def __hash__(self):
        return c_ex_profile_date_to_ordinal(self.header)

    def __format__(self, str format_spec):
        return self.to_pydate().__format__(format_spec)

    def __sub__(self, object other):
        if isinstance(other, SessionDate):
            return SessionDateRange.__new__(SessionDateRange, other, self)
        elif isinstance(other, py_date):
            return SessionDateRange.__new__(SessionDateRange, SessionDate.from_pydate(other), self)
        else:
            raise TypeError(f'Can not subtract {self.__class__.__name__} with {other.__class__.__name__}')

    def __richcmp__(self, object other, int op):
        cdef int cmp
        if isinstance(other, SessionDate):
            cmp = c_ex_profile_date_compare(self.header, (<SessionDate> other).header)
        elif isinstance(other, py_date):
            cmp = c_ex_profile_date_compare(self.header, (<SessionDate> SessionDate.from_pydate(other)).header)
        else:
            raise TypeError(f'Can not compare {self.__class__.__name__} with {other.__class__.__name__}')

        if op == Py_LT:
            return cmp < 0
        elif op == Py_LE:
            return cmp <= 0
        elif op == Py_EQ:
            return cmp == 0
        elif op == Py_NE:
            return cmp != 0
        elif op == Py_GT:
            return cmp > 0
        elif op == Py_GE:
            return cmp >= 0
        else:
            raise ValueError(f'Invalid comparison operator: {op}')

    @staticmethod
    def is_leap_year(uint16_t year):
        return c_ex_profile_is_leap_year(year)

    @staticmethod
    def days_in_month(uint16_t year, uint8_t month):
        return c_ex_profile_days_in_month(year, month)

    @classmethod
    def from_ts(cls, double unix_ts):
        cdef session_date_t* header = <session_date_t*> calloc(1, sizeof(session_date_t))
        if not header:
            raise MemoryError(f'Failed to allocate memory for {cls.__name__}')
        cdef int ret_code = c_ex_profile_session_date_from_ts(unix_ts, header)
        if ret_code != 0:
            free(header)
            raise RuntimeError(f'c_ex_profile_session_date_from_ts failed with err code: {ret_code}')
        cdef SessionDate instance = cls.__new__(cls, 0, 0, 0, no_alloc=True)
        instance.header = header
        instance.owner = True
        return instance

    @classmethod
    def from_ordinal(cls, uint32_t ordinal):
        cdef session_date_t* header = <session_date_t*> calloc(1, sizeof(session_date_t))
        if not header:
            raise MemoryError(f'Failed to allocate memory for {cls.__name__}')
        cdef int ret_code = c_ex_profile_date_from_ordinal(ordinal, header)
        if ret_code != 0:
            free(header)
            raise RuntimeError(f'c_ex_profile_date_from_ordinal failed with err code: {ret_code}')
        cdef SessionDate instance = cls.__new__(cls, 0, 0, 0, no_alloc=True)
        instance.header = header
        instance.owner = True
        return instance

    @classmethod
    def from_pydate(cls, py_date dt):
        return cls.__new__(cls, dt.year, dt.month, dt.day)

    def to_pydate(self):
        return py_date(self.header.year, self.header.month, self.header.day)

    def to_ordinal(self):
        return c_ex_profile_date_to_ordinal(self.header)

    def add_days(self, size_t days):
        cdef session_date_t* out = <session_date_t*> calloc(1, sizeof(session_date_t))
        if not out:
            raise MemoryError(f'Failed to allocate memory for {self.__class__.__name__}')
        cdef int ret_code = c_ex_profile_days_after(self.header, days, out)
        if ret_code != 0:
            raise RuntimeError(f'c_ex_profile_days_after failed with err code: {ret_code}')
        cdef SessionDate instance = self.__class__.__new__(self.__class__, 0, 0, 0, no_alloc=True)
        instance.header = out
        instance.owner = True
        return instance

    def is_valid(self):
        return c_ex_profile_date_is_valid(self.header)

    def ctime(self):
        return self.to_pydate().ctime()

    @classmethod
    def fromisocalendar(cls, int year, int week, int weekday):
        return cls.from_pydate(py_date.fromisocalendar(year, week, weekday))

    @classmethod
    def fromisoformat(cls, str date_str):
        return cls.from_pydate(py_date.fromisoformat(date_str))

    def isocalendar(self):
        return self.to_pydate().isocalendar()

    def isoformat(self, *args, **kwargs):
        return self.to_pydate().isoformat(*args, **kwargs)

    def strftime(self, format):
        return self.to_pydate().strftime(format)

    def weekday(self):
        return self.to_pydate().weekday()

    property year:
        def __get__(self):
            return self.header.year

    property month:
        def __get__(self):
            return self.header.month

    property day:
        def __get__(self):
            return self.header.day

    property session_type:
        def __get__(self):
            return SessionType(self.header.stype)


cdef class SessionDateRange:
    def __cinit__(self, SessionDate start_date, SessionDate end_date, *, bint no_alloc=False):
        if no_alloc:
            return
        self.header = c_ex_profile_session_drange_between(start_date.header, end_date.header)
        if not self.header:
            raise MemoryError(f"Failed to allocate memory for {self.__class__.__name__}")
        self.owner = True
        self.start_date = SessionDate.c_from_header(&self.header.start, False)
        self.end_date = SessionDate.c_from_header(&self.header.end, False)

        cdef size_t i
        cdef size_t ttl = self.header.n_days
        cdef list dates = []
        for i in range(ttl):
            dates.append(SessionDate.c_from_header(self.header.dates + i, False))
        self.dates = tuple(dates)

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            free(<void*> self.header)

    @staticmethod
    cdef SessionDateRange c_from_header(const session_date_range_t* header, bint owner):
        cdef SessionDateRange instance = SessionDateRange.__new__(SessionDateRange, None, None, no_alloc=True)
        instance.header = header
        instance.owner = owner
        instance.start_date = SessionDate.c_from_header(&header.start, False)
        instance.end_date = SessionDate.c_from_header(&header.end, False)
        return instance

    # === Python Interfaces ===

    def __repr__(self):
        return f"<{self.__class__.__name__}>(n_days={self.header.n_days:,}, start={self.start_date.to_pydate()}, end={self.end_date.to_pydate()})"

    def __iter__(self):
        return self.dates.__iter__()

    def __getitem__(self, Py_ssize_t idx):
        return self.dates[idx]

    property n_days:
        def __get__(self):
            return self.header.n_days


cdef class CallAuction:
    def __init__(self):
        raise NotImplementedError(f'{self.__class__.__name__} should not be initialized from python interface.')

    @staticmethod
    cdef CallAuction c_from_header(const call_auction* header):
        cdef CallAuction instance = CallAuction.__new__(CallAuction)
        instance.header = header
        return instance

    property auction_start:
        def __get__(self):
            return SessionTime.c_from_header(&self.header.auction_start, False)

    property auction_end:
        def __get__(self):
            return SessionTime.c_from_header(&self.header.auction_end, False)

    property uncross:
        def __get__(self):
            return SessionTime.c_from_header(&self.header.uncross, False)

    property active:
        def __get__(self):
            if self.header.active:
                return SessionTimeRange.c_from_header(self.header.active, False)
            return None

    property no_cancel:
        def __get__(self):
            if self.header.no_cancel:
                return SessionTimeRange.c_from_header(self.header.no_cancel, False)
            return None

    property frozen:
        def __get__(self):
            if self.header.frozen:
                return SessionTimeRange.c_from_header(self.header.frozen, False)
            return None


cdef class SessionBreak:
    def __init__(self):
        raise NotImplementedError(f'{self.__class__.__name__} should not be initialized from python interface.')

    @staticmethod
    cdef SessionBreak c_from_header(const session_break* header):
        cdef SessionBreak instance = SessionBreak.__new__(SessionBreak)
        instance.header = header
        return instance

    property break_start:
        def __get__(self):
            return SessionTime.c_from_header(&self.header.break_start, False)

    property break_end:
        def __get__(self):
            return SessionTime.c_from_header(&self.header.break_end, False)

    property break_start_ts:
        def __get__(self):
            return self.header.break_start_ts

    property break_end_ts:
        def __get__(self):
            return self.header.break_end_ts

    property break_length_seconds:
        def __get__(self):
            return self.header.break_length_seconds

    property next_break:
        def __get__(self):
            if self.header.next:
                return SessionBreak.c_from_header(self.header.next)
            return None


cdef class ExchangeProfile:
    def __init__(self):
        raise NotImplementedError(f'{self.__class__.__name__} should not be initialized from python interface.')

    @staticmethod
    cdef ExchangeProfile c_from_header(const exchange_profile* header):
        cdef ExchangeProfile instance = ExchangeProfile.__new__(ExchangeProfile)

        instance.header = header
        instance.profile_id = PyUnicode_FromString(header.profile_id)
        instance.session_start = SessionTime.c_from_header(&header.session_start, False)
        instance.session_end = SessionTime.c_from_header(&header.session_end, False)

        if header.open_call_auction:
            instance.open_call_auction = CallAuction.c_from_header(header.open_call_auction)
        else:
            instance.open_call_auction = None

        if header.close_call_auction:
            instance.close_call_auction = CallAuction.c_from_header(header.close_call_auction)
        else:
            instance.close_call_auction = None

        cdef list breaks = []
        cdef session_break* current_break = header.session_breaks
        while current_break:
            breaks.append(SessionBreak.c_from_header(current_break))
            current_break = current_break.next
        instance.session_breaks = tuple(breaks)

        instance.time_zone = PyUnicode_FromString(header.time_zone) if header.time_zone else None

        return instance

    def __repr__(self):
        return f"<{self.__class__.__name__}>({self.profile_id})"

    def activate(self):
        c_ex_profile_activate(self.header)

    def deactivate(self):
        if EX_PROFILE == self.header and EX_PROFILE != &EX_PROFILE_DEFAULT:
            c_ex_profile_activate(&EX_PROFILE_DEFAULT)

    def trade_calendar(self, object start_date, object end_date):
        cdef const session_date_t* c_start_date = NULL
        cdef const session_date_t* c_end_date = NULL

        if isinstance(start_date, SessionDate):
            c_start_date = (<SessionDate> start_date).header
        elif isinstance(start_date, py_date):
            temp_start = SessionDate.from_pydate(start_date)
            c_start_date = (<SessionDate> temp_start).header
        else:
            raise TypeError(f'Unsupported type for start_date: {start_date.__class__.__name__}')

        if isinstance(end_date, SessionDate):
            c_end_date = (<SessionDate> end_date).header
        elif isinstance(end_date, py_date):
            temp_end = SessionDate.from_pydate(end_date)
            c_end_date = (<SessionDate> temp_end).header
        else:
            raise TypeError(f'Unsupported type for end_date: {end_date.__class__.__name__}')

        cdef session_date_range_t* drange = self.header.trade_calendar_func(c_start_date, c_end_date)
        if not drange:
            raise RuntimeError(f'{self} failed to get trade calendar from start_date={start_date} to end_date={end_date}')
        return SessionDateRange.c_from_header(drange, True)

    def resolve_auction_phase(self, object session_time):
        cdef double ts = 0.0
        cdef const session_time_t* c_time
        if isinstance(session_time, SessionTime):
            c_time = (<SessionTime> session_time).header
            ts = c_ex_profile_time_to_ts(c_time.hour, c_time.minute, c_time.second, c_time.nanosecond)
        elif isinstance(session_time, py_time):
             ts = c_ex_profile_time_to_ts(session_time.hour, session_time.minute, session_time.second, session_time.microsecond * 1000)
        else:
            raise TypeError(f'Unsupported type for session_time: {session_time.__class__.__name__}')
        return AuctionPhase(self.header.resolve_auction_phase_func(ts))

    def resolve_session_phase(self, object session_time):
        cdef double ts = 0.0
        cdef const session_time_t* c_time
        if isinstance(session_time, SessionTime):
            c_time = (<SessionTime> session_time).header
            ts = c_ex_profile_time_to_ts(c_time.hour, c_time.minute, c_time.second, c_time.nanosecond)
        elif isinstance(session_time, py_time):
            ts = c_ex_profile_time_to_ts(session_time.hour, session_time.minute, session_time.second, session_time.microsecond * 1000)
        else:
            raise TypeError(f'Unsupported type for session_time: {session_time.__class__.__name__}')
        return SessionPhase(self.header.resolve_session_phase_func(ts))

    def resolve_session_type(self, object session_date):
        cdef const session_date_t* c_date
        if isinstance(session_date, SessionDate):
            c_date = (<SessionDate> session_date).header
            return SessionType(self.header.resolve_session_type_func(c_date.year, c_date.month, c_date.day))
        elif isinstance(session_date, py_date):
            return SessionType(self.header.resolve_session_type_func(session_date.year, session_date.month, session_date.day))
        else:
            raise TypeError(f'Unsupported type for date: {session_date.__class__.__name__}')

    property session_start_ts:
        def __get__(self):
            return self.header.session_start_ts

    property session_end_ts:
        def __get__(self):
            return self.header.session_end_ts

    property session_length_seconds:
        def __get__(self):
            return self.header.session_length_seconds

    property tz_offset_seconds:
        def __get__(self):
            return self.header.tz_offset_seconds


cdef ExchangeProfile DEFAULT_PROFILE = ExchangeProfile.c_from_header(&EX_PROFILE_DEFAULT)
globals()['DEFAULT_PROFILE'] = DEFAULT_PROFILE
