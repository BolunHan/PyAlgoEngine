from libcpp cimport bool as c_bool

from cpython.datetime cimport datetime as py_datetime, date as py_date, time as py_time
from libc.stdint cimport uint8_t, uint16_t, uint32_t, int64_t


cdef extern from "c_ex_profile_base.h":
    const double SECONDS_PER_DAY
    const double SECONDS_PER_HOUR
    const double SECONDS_PER_MINUTE
    const double NANOS_PER_SECOND
    const double MICROS_PER_SECOND
    const int64_t UNIX_EPOCH_ORDINAL
    const size_t EX_PROFILE_ID_SIZE
    const size_t EX_PROFILE_MIN_YEAR
    const size_t EX_PROFILE_MAX_YEAR
    const uint32_t EX_PROFILE_MAX_ORDINAL

    const uint8_t DAYS_IN_MONTH_TABLE[12]
    const uint16_t DAYS_BEFORE_MONTH_TABLE[12]

    ctypedef enum session_type:
        SESSION_TYPE_NON_TRADING
        SESSION_TYPE_NORMINAL
        SESSION_TYPE_SUSPENDED
        SESSION_TYPE_HALF_DAY
        SESSION_TYPE_CIRCUIT_BREAK

    ctypedef enum session_phase:
        SESSION_PHASE_UNKNOWN
        SESSION_PHASE_PREOPEN
        SESSION_PHASE_OPEN_AUCTION
        SESSION_PHASE_CONTINUOUS
        SESSION_PHASE_BREAK
        SESSION_PHASE_SUSPENDED
        SESSION_PHASE_CLOSE_AUCTION
        SESSION_PHASE_CLOSED

    ctypedef enum auction_phase:
        AUCTION_PHASE_ACTIVE
        AUCTION_PHASE_NO_CANCEL
        AUCTION_PHASE_FROZEN
        AUCTION_PHASE_UNCROSSING
        AUCTION_PHASE_DONE

    ctypedef struct session_time_t:
        double elapsed_seconds
        uint8_t hour
        uint8_t minute
        uint8_t second
        uint32_t nanosecond
        session_phase phase

    ctypedef struct session_date_t:
        uint16_t year
        uint8_t month
        uint8_t day
        session_type stype

    ctypedef struct session_time_range_t:
        session_time_t start
        session_time_t end
        double elapsed_seconds

    ctypedef struct session_date_range_t:
        session_date_t start
        session_date_t end
        size_t n_days
        session_date_t dates[]

    ctypedef struct call_auction:
        session_time_t auction_start
        const session_time_range_t* active
        const session_time_range_t* no_cancel
        const session_time_range_t* frozen
        session_time_t uncross
        session_time_t auction_end

    ctypedef struct session_break:
        session_time_t break_start
        session_time_t break_end
        double break_start_ts
        double break_end_ts
        double break_length_seconds
        session_break* next

    ctypedef void (*c_ex_profile_on_activate)(const exchange_profile* profile) noexcept nogil
    ctypedef void (*c_ex_profile_on_deactivate)(const exchange_profile* profile) noexcept nogil
    ctypedef session_date_range_t* (*c_ex_profile_trade_calendar)(const session_date_t* start_date, const session_date_t* end_date) noexcept nogil
    ctypedef auction_phase (*c_ex_profile_resolve_auction_phase)(double ts) noexcept nogil
    ctypedef session_phase (*c_ex_profile_resolve_session_phase)(double ts) noexcept nogil
    ctypedef session_type (*c_ex_profile_resolve_session_type)(uint16_t year, uint8_t month, uint8_t day) noexcept nogil

    ctypedef struct exchange_profile:
        char profile_id[EX_PROFILE_ID_SIZE]
        session_time_t session_start
        session_time_t session_end
        double session_start_ts
        double session_end_ts
        double session_length_seconds
        const call_auction* open_call_auction
        const call_auction* close_call_auction
        const session_break* session_breaks
        const char* time_zone
        double tz_offset_seconds
        # Function pointers for exchange-specific logic
        c_ex_profile_on_activate on_activate_func
        c_ex_profile_on_deactivate on_deactivate_func
        c_ex_profile_trade_calendar trade_calendar_func
        c_ex_profile_resolve_auction_phase resolve_auction_phase_func
        c_ex_profile_resolve_session_phase resolve_session_phase_func
        c_ex_profile_resolve_session_type resolve_session_type_func

    extern const exchange_profile* EX_PROFILE
    extern const session_date_range_t* EX_TRADE_CALENDAR_CACHE

    extern const exchange_profile EX_PROFILE_DEFAULT;

    double c_utc_offset_seconds() noexcept nogil
    int c_ex_profile_time_compare(const void* t1, const void* t2) noexcept nogil
    double c_ex_profile_time_to_ts(uint8_t hour, uint8_t minute, uint8_t second, uint32_t nanosecond) noexcept nogil
    double c_ex_profile_unix_to_ts(double unix_ts) noexcept nogil
    double c_ex_profile_ts_to_elapsed(double ts) noexcept nogil

    int c_ex_profile_date_compare(const void* d1, const void* d2) noexcept nogil
    uint32_t c_ex_profile_days_before_year(uint16_t year) noexcept nogil
    c_bool c_ex_profile_is_leap_year(uint16_t year) noexcept nogil
    uint8_t c_ex_profile_days_in_month(uint16_t year, uint8_t month) noexcept nogil
    c_bool c_ex_profile_date_is_valid(const session_date_t* date) noexcept nogil
    uint32_t c_ex_profile_date_to_ordinal(const session_date_t* date) noexcept nogil
    int c_ex_profile_date_from_ordinal(uint32_t ordinal, session_date_t* out) noexcept nogil
    int c_ex_profile_date_from_year_day(uint16_t year, uint32_t day_of_year, session_date_t* out) noexcept nogil
    int c_ex_profile_next_day(const session_date_t* date, session_date_t* out) noexcept nogil
    int c_ex_profile_previous_day(const session_date_t* date, session_date_t* out) noexcept nogil
    int c_ex_profile_days_after(const session_date_t* date, size_t days_after, session_date_t* out) noexcept nogil
    int c_ex_profile_days_before(const session_date_t* date, size_t days_before, session_date_t* out) noexcept nogil
    c_bool c_ex_profile_is_weekend(const session_date_t* date) noexcept nogil
    session_date_range_t* c_ex_profile_date_range(const session_date_t* start_date, const session_date_t* end_date) noexcept nogil

    void c_ex_profile_activate(const exchange_profile* profile) noexcept nogil

    session_time_t* c_ex_profile_session_time_new(uint8_t hour, uint8_t minute, uint8_t second, uint32_t nanosecond) noexcept nogil
    int c_ex_profile_session_time_from_ts(double unix_ts, session_time_t* out) noexcept nogil
    session_time_range_t* c_ex_profile_session_trange_between_time(const session_time_t* start_time, const session_time_t* end_time) noexcept nogil
    session_time_range_t* c_ex_profile_session_trange_between_ts(double start_unix_ts, double end_unix_ts) noexcept nogil

    session_date_t* c_ex_profile_session_date_new(uint16_t year, uint8_t month, uint8_t day) noexcept nogil
    int c_ex_profile_session_date_from_ts(double unix_ts, session_date_t* out) noexcept nogil
    size_t c_ex_profile_session_date_index(const session_date_t* date, const session_date_range_t* drange) noexcept nogil
    size_t c_ex_profile_session_ymd_index(uint16_t year, uint8_t month, uint8_t day, const session_date_t* date_array, size_t n_days) noexcept nogil
    session_date_range_t* c_ex_profile_session_drange_between(const session_date_t* start_date, const session_date_t* end_date) noexcept nogil
    int c_ex_profile_session_trading_days_before(const session_date_t* market_date, size_t days, session_date_t* out) noexcept nogil
    int c_ex_profile_session_trading_days_after(const session_date_t* market_date, size_t days, session_date_t* out) noexcept nogil
    int c_ex_profile_nearest_trading_date(const session_date_t* market_date, c_bool previous, session_date_t* out) noexcept nogil


cdef extern from "c_ex_profile_cn.h":
    extern const exchange_profile EX_PROFILE_CN


cdef class SessionTime:
    cdef const session_time_t* header
    cdef bint owner

    @staticmethod
    cdef SessionTime c_from_header(const session_time_t* header, bint owner)


cdef class SessionTimeRange:
    cdef const session_time_range_t* header
    cdef bint owner

    cdef readonly SessionTime start_time
    cdef readonly SessionTime end_time

    @staticmethod
    cdef SessionTimeRange c_from_header(const session_time_range_t* header, bint owner)


cdef class SessionDate:
    cdef const session_date_t* header
    cdef bint owner

    @staticmethod
    cdef SessionDate c_from_header(const session_date_t* header, bint owner)


cdef class SessionDateRange:
    cdef const session_date_range_t* header
    cdef bint owner

    cdef readonly SessionDate start_date
    cdef readonly SessionDate end_date
    cdef readonly tuple dates

    @staticmethod
    cdef SessionDateRange c_from_header(const session_date_range_t* header, bint owner)


cdef class CallAuction:
    cdef const call_auction* header

    @staticmethod
    cdef CallAuction c_from_header(const call_auction* header)


cdef class SessionBreak:
    cdef const session_break* header

    @staticmethod
    cdef SessionBreak c_from_header(const session_break* header)


cdef class ExchangeProfile:
    cdef const exchange_profile* header

    cdef readonly str profile_id
    cdef readonly SessionTime session_start
    cdef readonly SessionTime session_end
    cdef readonly CallAuction open_call_auction
    cdef readonly CallAuction close_call_auction
    cdef readonly tuple[SessionBreak] session_breaks
    cdef readonly str time_zone

    @staticmethod
    cdef ExchangeProfile c_from_header(const exchange_profile* header)


cdef class ProfileCompatible:
    cdef dict __dict__
    cdef const exchange_profile** header

    cdef inline bint c_time_in_market_session(self, py_time t):
        cdef const exchange_profile* active_profile = self.header[0]
        cdef double ts = c_ex_profile_time_to_ts(t.hour, t.minute, t.second, t.microsecond * 1000)
        cdef phase = active_profile.resolve_session_phase_func(ts)
        return phase == session_phase.SESSION_PHASE_CONTINUOUS

    cdef inline bint c_timestamp_in_market_session(self, double unix_ts):
        cdef const exchange_profile* active_profile = self.header[0]
        cdef double ts = c_ex_profile_unix_to_ts(unix_ts)
        cdef phase = active_profile.resolve_session_phase_func(ts)
        return phase == session_phase.SESSION_PHASE_CONTINUOUS

    cdef inline bint c_timestamp_in_auction_session(self, double unix_ts):
        cdef const exchange_profile* active_profile = self.header[0]
        cdef double ts = c_ex_profile_unix_to_ts(unix_ts)
        cdef phase = active_profile.resolve_session_phase_func(ts)
        return phase == session_phase.SESSION_PHASE_OPEN_AUCTION or phase == session_phase.SESSION_PHASE_CLOSE_AUCTION

    cdef inline SessionDateRange c_trade_calendar(self, py_date start_date, py_date end_date):
        cdef SessionDate d1 = SessionDate.from_pydate(start_date)
        cdef SessionDate d2 = SessionDate.from_pydate(end_date)
        cdef session_date_range_t* drange = c_ex_profile_date_range(d1.header, d2.header)
        if drange:
            return SessionDateRange.c_from_header(drange, True)
        return None

    cdef inline bint c_date_in_market_session(self, py_date market_date):
        cdef session_date_t target
        target.year = market_date.year
        target.month = market_date.month
        target.day = market_date.day
        cdef size_t i = c_ex_profile_session_date_index(&target, EX_TRADE_CALENDAR_CACHE)
        if i == <size_t> -1:
            return False
        cdef const session_date_t* got = EX_TRADE_CALENDAR_CACHE.dates + i
        if c_ex_profile_date_compare(&target, got) == 0:
            return True
        return False

    cdef inline double c_time_to_seconds(self, py_time t, bint break_adjusted):
        cdef double ts =  c_ex_profile_time_to_ts(t.hour, t.minute, t.second, t.microsecond * 1000)
        if break_adjusted:
            return c_ex_profile_ts_to_elapsed(ts)
        return ts

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

        cdef double ts_diff = 0
        cdef const exchange_profile* active_profile = self.header[0]
        if c_ex_profile_date_compare(&d1, &d2) == 0:
            pass
        else:
            ts_diff += active_profile.session_length_seconds * (c_ex_profile_session_date_index(&d2, EX_TRADE_CALENDAR_CACHE) - c_ex_profile_session_date_index(&d1, EX_TRADE_CALENDAR_CACHE))
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

        cdef size_t idx1 = c_ex_profile_session_date_index(&d1, EX_TRADE_CALENDAR_CACHE)
        cdef size_t idx2 = c_ex_profile_session_date_index(&d2, EX_TRADE_CALENDAR_CACHE)
        if idx1 == <size_t> -1 or idx2 == <size_t> -1:
            raise ValueError(f'One of the provided dates is out of range for the exchange calendar: {start_date} or {end_date}')
        return <ssize_t> idx2 - <ssize_t> idx1

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

    cpdef double time_to_seconds(self, py_time t, bint break_adjusted=*)

    cpdef double timestamp_to_seconds(self, double t, bint break_adjusted=*)

    cpdef double break_adjusted(self, double elapsed_seconds)

    cpdef double trading_time_between(self, object start_time, object end_time)

    cpdef bint is_market_session(self, object timestamp)

    cpdef bint is_auction_session(self, object timestamp)

    cpdef list trade_calendar(self, py_date start_date, py_date end_date)

    cpdef py_date trading_days_before(self, py_date market_date, ssize_t days)

    cpdef py_date trading_days_after(self, py_date market_date, ssize_t days)

    cpdef size_t trading_days_between(self, py_date start_date, py_date end_date)

    cpdef py_date nearest_trading_date(self, py_date market_date, str method=*)

    cpdef bint is_trading_day(self, py_date market_date)


cdef ProfileCompatible PROFILE

cdef ExchangeProfile PROFILE_DEFAULT
cdef ExchangeProfile PROFILE_CN
