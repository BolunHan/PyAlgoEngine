import enum
from datetime import timezone

from cpython.datetime cimport timedelta
from cpython.object cimport Py_LT, Py_LE, Py_EQ, Py_NE, Py_GT, Py_GE, PyObject
from cpython.unicode cimport PyUnicode_FromString
from libc.stdlib cimport calloc, free

from . import LOGGER


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


cpdef double local_utc_offset_seconds():
    return c_utc_offset_seconds()


cdef int c_ex_profile_unix_to_datetime(double unix_ts, session_datetime_t* out):
    cdef int ret_code = c_ex_profile_session_time_from_unix(unix_ts, &out.time)
    if ret_code == 0:
        return c_ex_profile_date_from_ordinal(c_ex_profile_unix_to_ordinal(unix_ts, EX_PROFILE.tz_offset_seconds if EX_PROFILE else 0.0), &out.date)
    return ret_code


cpdef py_datetime unix_to_datetime(double unix_ts):
    cdef session_datetime_t dt
    cdef int ret_code = c_ex_profile_unix_to_datetime(unix_ts, &dt)
    if ret_code != 0:
        raise ValueError(f'unix_ts: {unix_ts} out of range.')
    return py_datetime.__new__(py_datetime, dt.date.year, dt.date.month, dt.date.day, dt.time.hour, dt.time.minute, dt.time.second, dt.time.nanosecond // 1000, PROFILE.time_zone)


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
    def from_ts(cls, double ts):
        if ts >= 24 * 3600 or ts < 0:
            raise ValueError(f"Timestamp out of range for a single day: {ts}")
        cdef SessionTime instance = SessionTime.__new__(SessionTime)
        if c_ex_profile_session_time_from_ts(ts, <session_time_t*> instance.header) != 0:
            raise ValueError(f"Invalid timestamp: {ts}")
        return instance

    @classmethod
    def from_timestamp(cls, double unix_ts):
        cdef SessionTime instance = SessionTime.__new__(SessionTime)
        if c_ex_profile_session_time_from_unix(unix_ts, <session_time_t*> instance.header) != 0:
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
    def today(cls):
        return cls.from_pydate(py_date.today())

    @classmethod
    def from_unix(cls, double unix_ts):
        cdef session_date_t* header = <session_date_t*> calloc(1, sizeof(session_date_t))
        if not header:
            raise MemoryError(f'Failed to allocate memory for {cls.__name__}')
        cdef int ret_code = c_ex_profile_session_date_from_unix(unix_ts, header)
        if ret_code != 0:
            free(header)
            raise RuntimeError(f'c_ex_profile_session_date_from_unix failed with err code: {ret_code}')
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

    def is_weekend(self):
        return c_ex_profile_is_weekend(self.header)

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

    def timestamp(self):
        return c_ex_profile_session_date_to_unix(self.header)

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
        cdef size_t i
        cdef size_t ttl = header.n_days
        cdef list dates = []
        for i in range(ttl):
            dates.append(SessionDate.c_from_header(header.dates + i, False))
        instance.dates = tuple(dates)
        return instance

    # === Python Interfaces ===

    def __repr__(self):
        return f"<{self.__class__.__name__}>(n_days={self.header.n_days:,}, start={self.start_date.to_pydate()}, end={self.end_date.to_pydate()})"

    def __iter__(self):
        return self.dates.__iter__()

    def __getitem__(self, Py_ssize_t idx):
        return self.dates[idx]

    def __contains__(self, object item):
        cdef session_date_t dt
        if isinstance(item, py_date):
            dt.year = item.year
            dt.month = item.month
            dt.day = item.day
        elif isinstance(item, SessionDate):
            dt.year = item.header.year
            dt.month = item.header.month
            dt.day = item.header.day
        else:
            raise TypeError(f'Unsupported type for membership test: {item.__class__.__name__}')
        cdef size_t i = c_ex_profile_session_date_index(&dt, self.header)
        if i == <size_t> -1:
            return False
        cdef const session_date_t* found = self.header.dates + i
        return c_ex_profile_date_compare(found, &dt) == 0

    def __len__(self):
        return self.n_days

    def index(self, object item):
        cdef session_date_t dt
        if isinstance(item, py_date):
            dt.year = item.year
            dt.month = item.month
            dt.day = item.day
        elif isinstance(item, SessionDate):
            dt.year = item.header.year
            dt.month = item.header.month
            dt.day = item.header.day
        else:
            raise TypeError(f'Unsupported type for index lookup: {item.__class__.__name__}')
        cdef size_t i = c_ex_profile_session_date_index(&dt, self.header)
        if i == <size_t> -1:
            raise ValueError(f'{item} not found in {self}')
        cdef const session_date_t* found = self.header.dates + i
        if c_ex_profile_date_compare(found, &dt) == 0:
            return i
        raise ValueError(f'{item} not found in {self}')

    cpdef list to_list(self):
        cdef list out = []
        cdef session_date_t dt
        for i in range(self.header.n_days):
            dt = self.header.dates[i]
            out.append(py_date.__new__(py_date, dt.year, dt.month, dt.day))
        return out

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

    def __dealloc__(self):
        if self.listener_id:
            c_ex_profile_deregister_activation_listener(self.listener_id)

    @staticmethod
    cdef ExchangeProfile c_new_bound_instance():
        cdef ExchangeProfile instance = ExchangeProfile.__new__(ExchangeProfile)
        instance.c_bind(EX_PROFILE)
        cdef uintptr_t listener_id = c_ex_profile_register_activation_listener(
            ExchangeProfile.c_listener_adapter,
            <void*> <PyObject*> instance
        )
        instance.listener_id = listener_id
        return instance

    @staticmethod
    cdef ExchangeProfile c_from_header(const exchange_profile* header):
        cdef ExchangeProfile instance = ExchangeProfile.__new__(ExchangeProfile)
        instance.c_bind(header)
        return instance

    @staticmethod
    cdef void c_listener_adapter(const exchange_profile* previous_profile, const exchange_profile* new_profile, void* user_data) noexcept:
        cdef ExchangeProfile py_profile = <ExchangeProfile> <PyObject*> user_data
        py_profile.c_bind(new_profile)
        LOGGER.info(f'Active ExchangeProfile switched: {PyUnicode_FromString(previous_profile.profile_id)} -> {PyUnicode_FromString(new_profile.profile_id)}')

    cdef inline void c_bind(self, const exchange_profile* header):
        self.header = header
        self.profile_id = PyUnicode_FromString(header.profile_id)
        self.session_start = SessionTime.c_from_header(&header.session_start, False)
        self.session_end = SessionTime.c_from_header(&header.session_end, False)

        if header.open_call_auction:
            self.open_call_auction = CallAuction.c_from_header(header.open_call_auction)
        else:
            self.open_call_auction = None

        if header.close_call_auction:
            self.close_call_auction = CallAuction.c_from_header(header.close_call_auction)
        else:
            self.close_call_auction = None

        cdef list breaks = []
        cdef const session_break* current_break = header.session_breaks
        while current_break:
            breaks.append(SessionBreak.c_from_header(current_break))
            current_break = current_break.next
        self.session_breaks = tuple(breaks)
        self.time_zone = timezone(
            offset=timedelta(seconds=<Py_ssize_t> header.tz_offset_seconds),
            name=PyUnicode_FromString(header.time_zone)
        ) if header.time_zone else None

    cdef inline bint c_time_in_market_session(self, py_time t):
        cdef double ts = c_ex_profile_time_to_ts(t.hour, t.minute, t.second, t.microsecond * 1000)
        cdef phase = self.header.resolve_session_phase(ts)
        return phase == session_phase.SESSION_PHASE_CONTINUOUS

    cdef inline bint c_timestamp_in_market_session(self, double unix_ts):
        cdef double ts = c_ex_profile_unix_to_ts(unix_ts)
        cdef phase = self.header.resolve_session_phase(ts)
        return phase == session_phase.SESSION_PHASE_CONTINUOUS

    cdef inline bint c_timestamp_in_auction_session(self, double unix_ts):
        cdef double ts = c_ex_profile_unix_to_ts(unix_ts)
        cdef phase = self.header.resolve_session_phase(ts)
        return phase == session_phase.SESSION_PHASE_OPEN_AUCTION or phase == session_phase.SESSION_PHASE_CLOSE_AUCTION

    cdef inline SessionDateRange c_trade_calendar(self, py_date start_date, py_date end_date):
        cdef SessionDate d1 = SessionDate.from_pydate(start_date)
        cdef SessionDate d2 = SessionDate.from_pydate(end_date)
        cdef session_date_range_t* drange = c_ex_profile_session_drange_between(d1.header, d2.header)
        if drange:
            return SessionDateRange.c_from_header(drange, True)
        return None

    cdef inline bint c_date_in_market_session(self, py_date market_date):
        cdef session_date_t target
        target.year = market_date.year
        target.month = market_date.month
        target.day = market_date.day
        return c_ex_profile_is_trading_day(&target)

    cdef inline double c_time_to_seconds(self, py_time t, bint break_adjusted):
        cdef double ts =  c_ex_profile_time_to_ts(t.hour, t.minute, t.second, t.microsecond * 1000)
        if break_adjusted:
            return c_ex_profile_ts_to_elapsed(ts)
        return ts

    cdef inline py_datetime c_timestamp_to_datetime(self, double unix_ts):
        cdef session_date_t out_date
        cdef session_time_t out_time
        cdef int ret_code = c_ex_profile_session_date_from_unix(unix_ts, &out_date)
        if ret_code != 0:
            raise ValueError(f'unix_ts: {unix_ts} out of range.')
        ret_code = c_ex_profile_session_time_from_unix(unix_ts, &out_time)
        if ret_code != 0:
            raise ValueError(f'unix_ts: {unix_ts} out of range.')
        return py_datetime.__new__(py_datetime, out_date.year, out_date.month, out_date.day, out_time.hour, out_time.minute, out_time.second, out_time.nanosecond // 1000, self.time_zone)

    cdef inline py_date c_timestamp_to_date(self, double unix_ts):
        cdef session_date_t out_date
        cdef int ret_code = c_ex_profile_session_date_from_unix(unix_ts, &out_date)
        if ret_code != 0:
            raise ValueError(f'unix_ts: {unix_ts} out of range.')
        return py_date.__new__(py_datetime, out_date.year, out_date.month, out_date.day)

    cdef inline double c_timestamp_to_seconds(self, double t, bint break_adjusted):
        cdef double ts = c_ex_profile_unix_to_ts(t)
        if break_adjusted:
            return c_ex_profile_ts_to_elapsed(ts)
        return ts

    cdef inline double c_break_adjusted(self, double ts):
        return c_ex_profile_ts_to_elapsed(ts)

    cdef inline double c_trading_time_between(self, py_datetime start_time, py_datetime end_time):
        cdef session_date_t d1
        cdef session_time_t t1
        d1.year = start_time.year
        d1.month = start_time.month
        d1.day = start_time.day
        t1.hour = start_time.hour
        t1.minute = start_time.minute
        t1.second = start_time.second
        t1.nanosecond = start_time.microsecond * 1000

        cdef session_date_t d2
        cdef session_time_t t2
        d2.year = end_time.year
        d2.month = end_time.month
        d2.day = end_time.day
        t2.hour = end_time.hour
        t2.minute = end_time.minute
        t2.second = end_time.second
        t2.nanosecond = end_time.microsecond * 1000

        cdef ssize_t date_diff = 0
        cdef double ts_diff = 0
        if c_ex_profile_date_compare(&d1, &d2) == 0:
            pass
        else:
            c_ex_profile_trading_days_between(&d1, &d2, &date_diff)
            ts_diff += self.header.session_length_seconds * date_diff
        cdef double ts1 = c_ex_profile_time_to_ts(t1.hour, t1.minute, t1.second, t1.nanosecond)
        cdef double ts2 = c_ex_profile_time_to_ts(t2.hour, t2.minute, t2.second, t2.nanosecond)
        cdef elapsed1 = c_ex_profile_ts_to_elapsed(ts1)
        cdef elapsed2 = c_ex_profile_ts_to_elapsed(ts2)
        ts_diff += elapsed2 - elapsed1
        return ts_diff

    cdef inline py_date c_trading_days_before(self, py_date market_date, size_t days):
        cdef session_date_t d
        d.year = market_date.year
        d.month = market_date.month
        d.day = market_date.day

        cdef session_date_t out
        c_ex_profile_session_trading_days_before(&d, days, &out)
        return py_date(out.year, out.month, out.day)

    cdef inline py_date c_trading_days_after(self, py_date market_date, size_t days):
        cdef session_date_t d
        d.year = market_date.year
        d.month = market_date.month
        d.day = market_date.day

        cdef session_date_t out
        c_ex_profile_session_trading_days_after(&d, days, &out)
        return py_date(out.year, out.month, out.day)

    cdef inline ssize_t c_trading_days_between(self, py_date start_date, py_date end_date):
        cdef session_date_t d1
        d1.year = start_date.year
        d1.month = start_date.month
        d1.day = start_date.day

        cdef session_date_t d2
        d2.year = end_date.year
        d2.month = end_date.month
        d2.day = end_date.day

        cdef ssize_t out = 0
        cdef int ret_code = c_ex_profile_trading_days_between(&d1, &d2, &out)
        if ret_code == 0:
            return out
        else:
            raise RuntimeError(f'c_ex_profile_trading_days_between failed with err code: {ret_code}')

    cdef inline py_date c_nearest_trading_date(self, py_date market_date, bint previous):
        cdef session_date_t dt
        dt.year = market_date.year
        dt.month = market_date.month
        dt.day = market_date.day
        cdef session_date_t out
        cdef int ret_code = c_ex_profile_nearest_trading_date(&dt, <c_bool> previous, &out)
        if ret_code == 0:
            return py_date(out.year, out.month, out.day)
        raise ValueError(f'Can not locate a nearest trading date for {market_date}')

    # === Python Interfaces ===

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

        cdef session_date_range_t* drange = self.header.trade_calendar(c_start_date, c_end_date)
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
        return AuctionPhase(self.header.resolve_auction_phase(ts))

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
        return SessionPhase(self.header.resolve_session_phase(ts))

    def resolve_session_type(self, object session_date):
        cdef const session_date_t* c_date
        if isinstance(session_date, SessionDate):
            c_date = (<SessionDate> session_date).header
            return SessionType(self.header.resolve_session_type(c_date.year, c_date.month, c_date.day))
        elif isinstance(session_date, py_date):
            return SessionType(self.header.resolve_session_type(session_date.year, session_date.month, session_date.day))
        else:
            raise TypeError(f'Unsupported type for date: {session_date.__class__.__name__}')

    def timestamp_to_datetime(self, double unix_ts):
        return self.c_timestamp_to_datetime(unix_ts)

    def time_to_seconds(self, py_time t, bint break_adjusted=True):
        cdef double ts = c_ex_profile_time_to_ts(t.hour, t.minute, t.second, t.microsecond * 1000)
        cdef phase = self.header.resolve_session_phase(ts)
        assert phase == session_phase.SESSION_PHASE_CONTINUOUS, f'{t} not in continuous trading session'
        if break_adjusted:
            return c_ex_profile_ts_to_elapsed(ts)
        return ts

    def timestamp_to_seconds(self, double t, bint break_adjusted=True):
        cdef double ts = c_ex_profile_unix_to_ts(t)
        cdef phase = self.header.resolve_session_phase(ts)
        assert phase == session_phase.SESSION_PHASE_CONTINUOUS, f'{t} not in continuous trading session'
        if break_adjusted:
            return c_ex_profile_ts_to_elapsed(ts)
        return ts

    def break_adjusted(self, double elapsed_seconds):
        return c_ex_profile_ts_to_elapsed(elapsed_seconds)

    def trading_time_between(self, object start_time, object end_time):
        cdef py_datetime st, et
        cdef double ts

        if isinstance(start_time, (float, int)):
            st = py_datetime.fromtimestamp(start_time, tz=self.time_zone)
            assert self.c_timestamp_in_market_session(<double> start_time), f'{start_time} not in trading session'
        else:
            st = <py_datetime> start_time
            assert self.c_time_in_market_session(<py_time> st.time()), f'{start_time} not in trading session'

        if isinstance(end_time, (float, int)):
            et = py_datetime.fromtimestamp(end_time, tz=self.time_zone)
            assert self.c_timestamp_in_market_session(<double> end_time), f'{end_time} not in trading session'
        else:
            et = <py_datetime> end_time
            assert self.c_time_in_market_session(<py_time> et.time()), f'{end_time} not in trading session'

        assert start_time <= end_time, f'The {end_time=} must not prior to the {start_time=}'
        cdef seconds = self.c_trading_time_between(start_time=st, end_time=et)
        return seconds

    def is_market_session(self, object timestamp):
        if isinstance(timestamp, (float, int)):
            return self.c_timestamp_in_market_session(<double> timestamp)
        elif isinstance(timestamp, py_time):
            return self.c_time_in_market_session(<py_time> timestamp)
        elif isinstance(timestamp, py_datetime):
            if self.c_date_in_market_session(<py_date> (timestamp.date())):
                return self.c_time_in_market_session(<py_time> (timestamp.time()))
            else:
                return False
        raise TypeError(f'Invalid timestamp type {type(timestamp)}, except a datetime, time or numeric.')

    def is_auction_session(self, object timestamp):
        cdef double ts
        if isinstance(timestamp, (float, int)):
            return self.c_timestamp_in_auction_session(<double> timestamp)
        if isinstance(timestamp, py_time):
            ts = <double> py_datetime.combine(py_date.today(), <py_time> timestamp).timestamp()
            return self.c_timestamp_in_auction_session(ts)
        if isinstance(timestamp, py_datetime):
            ts = <double> timestamp.timestamp()
            return self.c_timestamp_in_auction_session(ts)
        raise TypeError(f'Invalid timestamp type {type(timestamp)}, except a datetime, time or numeric.')

    def trading_days_before(self, py_date market_date, ssize_t days):
        assert days > 0, "days must be positive"
        return self.c_trading_days_before(market_date=market_date, days=days)

    def trading_days_after(self, py_date market_date, ssize_t days):
        assert days > 0, "days must be positive"
        return self.c_trading_days_after(market_date=market_date, days=days)

    def trading_days_between(self, py_date start_date, py_date end_date):
        assert start_date <= end_date, f'The {end_date=} must not prior to the {start_date=}'
        return self.c_trading_days_between(start_date=start_date, end_date=end_date)

    def nearest_trading_date(self, py_date market_date, str method='previous'):
        cdef bint previous

        if method == 'previous':
            previous = True
        elif method == 'next':
            previous = False
        else:
            raise ValueError("method must be either 'previous' or 'next'")

        return self.c_nearest_trading_date(market_date=market_date, previous=previous)

    def is_trading_day(self, py_date market_date):
        return self.c_date_in_market_session(market_date)

    property bound_instance:
        def __get__(self):
            return bool(self.listener_id)

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

    property range_break:
        def __get__(self):
            range_break = []
            if not self.session_breaks:
                return range_break

            # Convert session_break to range_break format
            cdef SessionBreak session_break
            for session_break in self.session_breaks:
                start = session_break.break_start
                end = session_break.break_end
                start_hour = start.hour + start.minute / 60
                end_hour = end.hour + end.minute / 60
                range_break.append(dict(bounds=[start_hour, end_hour], pattern="hour"))

            # Add the additional fixed non-trading periods
            if self.session_start is not None and self.session_start != py_time.min:
                range_break.append(dict(bounds=[0, self.session_start.hour + self.session_start.minute / 60], pattern="hour"))
            if self.session_end is not None and self.session_end != py_time.max:
                range_break.append(dict(bounds=[self.session_end.hour + self.session_end.minute / 60, 24], pattern="hour"))
            return range_break

    property trade_calendar_cache:
        def __get__(self):
            if EX_TRADE_CALENDAR_CACHE:
                return SessionDateRange.c_from_header(EX_TRADE_CALENDAR_CACHE, False)
            return None


cdef ExchangeProfile PROFILE = ExchangeProfile.c_new_bound_instance()
globals()['PROFILE'] = PROFILE

cdef ExchangeProfile PROFILE_DEFAULT = ExchangeProfile.c_from_header(&EX_PROFILE_DEFAULT)
globals()['PROFILE_DEFAULT'] = PROFILE_DEFAULT

cdef ExchangeProfile PROFILE_CN = ExchangeProfile.c_from_header(&EX_PROFILE_CN)
globals()['PROFILE_CN'] = PROFILE_CN
