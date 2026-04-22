from libcpp cimport bool as c_bool

from cpython.datetime cimport datetime as py_datetime, date as py_date, time as py_time
from libc.stdint cimport uint8_t, uint16_t, uint32_t, int64_t, uintptr_t


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

    ctypedef struct session_datetime_t:
        session_time_t time
        session_date_t date
        double unix_ts
        double ts
        uint32_t ordinal

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

    ctypedef void (*ex_profile_on_activate_func)(const exchange_profile* profile) noexcept nogil
    ctypedef void (*ex_profile_on_deactivate_func)(const exchange_profile* profile) noexcept nogil
    ctypedef session_date_range_t* (*ex_profile_trade_calendar_func)(const session_date_t* start_date, const session_date_t* end_date) noexcept nogil
    ctypedef auction_phase (*ex_profile_resolve_auction_phase_func)(double ts) noexcept nogil
    ctypedef session_phase (*ex_profile_resolve_session_phase_func)(double ts) noexcept nogil
    ctypedef session_type (*ex_profile_resolve_session_type_func)(uint16_t year, uint8_t month, uint8_t day) noexcept nogil

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
        ex_profile_on_activate_func on_activate
        ex_profile_on_deactivate_func on_deactivate
        ex_profile_trade_calendar_func trade_calendar
        ex_profile_resolve_auction_phase_func resolve_auction_phase
        ex_profile_resolve_session_phase_func resolve_session_phase
        ex_profile_resolve_session_type_func resolve_session_type

    ctypedef void (*ex_profile_activation_callback)(const exchange_profile* previous_profile, const exchange_profile* new_profile, void* user_data) noexcept

    ctypedef struct ex_profile_activation_listener:
        uintptr_t listener_id;
        ex_profile_activation_callback callback;
        void* user_data;
        ex_profile_activation_listener* next;

    extern const exchange_profile* EX_PROFILE
    extern const session_date_range_t* EX_TRADE_CALENDAR_CACHE
    extern ex_profile_activation_listener* EX_PROFILE_ACTIVATION_LISTENERS

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
    uint32_t c_ex_profile_unix_to_ordinal(double unix_ts, double tz_offset_seconds) noexcept nogil
    int c_ex_profile_date_from_ordinal(uint32_t ordinal, session_date_t* out) noexcept nogil
    int c_ex_profile_date_from_year_day(uint16_t year, uint32_t day_of_year, session_date_t* out) noexcept nogil
    int c_ex_profile_next_day(const session_date_t* date, session_date_t* out) noexcept nogil
    int c_ex_profile_previous_day(const session_date_t* date, session_date_t* out) noexcept nogil
    int c_ex_profile_days_after(const session_date_t* date, size_t days_after, session_date_t* out) noexcept nogil
    int c_ex_profile_days_before(const session_date_t* date, size_t days_before, session_date_t* out) noexcept nogil
    c_bool c_ex_profile_is_weekend(const session_date_t* date) noexcept nogil
    session_date_range_t* c_ex_profile_date_range(const session_date_t* start_date, const session_date_t* end_date) noexcept nogil

    void c_ex_profile_activate(const exchange_profile* profile) noexcept nogil
    uintptr_t c_ex_profile_register_activation_listener(ex_profile_activation_callback callback, void* user_data)
    int c_ex_profile_deregister_activation_listener(uintptr_t listener_id)

    session_time_t* c_ex_profile_session_time_new(uint8_t hour, uint8_t minute, uint8_t second, uint32_t nanosecond) noexcept nogil
    int c_ex_profile_session_time_from_ts(double ts, session_time_t* out) noexcept nogil
    int c_ex_profile_session_time_from_unix(double unix_ts, session_time_t* out) noexcept nogil
    session_time_range_t* c_ex_profile_session_trange_between_time(const session_time_t* start_time, const session_time_t* end_time) noexcept nogil
    session_time_range_t* c_ex_profile_session_trange_between_unix(double start_unix_ts, double end_unix_ts) noexcept nogil

    session_date_t* c_ex_profile_session_date_new(uint16_t year, uint8_t month, uint8_t day) noexcept nogil
    int c_ex_profile_session_date_from_unix(double unix_ts, session_date_t* out) noexcept nogil
    double c_ex_profile_session_date_to_unix(const session_date_t* date) noexcept nogil
    size_t c_ex_profile_session_date_index(const session_date_t* date, const session_date_range_t* drange) noexcept nogil
    size_t c_ex_profile_session_ymd_index(uint16_t year, uint8_t month, uint8_t day, const session_date_t* date_array, size_t n_days) noexcept nogil
    session_date_range_t* c_ex_profile_session_drange_between(const session_date_t* start_date, const session_date_t* end_date) noexcept nogil
    int c_ex_profile_session_trading_days_before(const session_date_t* market_date, size_t days, session_date_t* out) noexcept nogil
    int c_ex_profile_session_trading_days_after(const session_date_t* market_date, size_t days, session_date_t* out) noexcept nogil
    int c_ex_profile_nearest_trading_date(const session_date_t* market_date, c_bool previous, session_date_t* out) noexcept nogil
    c_bool c_ex_profile_is_trading_day(const session_date_t* market_date) noexcept nogil
    int c_ex_profile_trading_days_between(const session_date_t* start_date, const session_date_t* end_date, ssize_t* out) noexcept nogil

    int c_ex_profile_session_datetime_from_unix(double unix_ts, session_datetime_t* out) noexcept nogil
    int c_ex_profile_session_datetime_update(session_datetime_t* dt, double unix_ts) noexcept nogil


cdef extern from "c_ex_profile_cn.h":
    extern const exchange_profile EX_PROFILE_CN

    c_bool c_ex_profile_cn_date_in_list(const session_date_t* date, const session_date_t* date_list, size_t n) noexcept nogil
    c_bool c_ex_profile_cn_is_holiday(const session_date_t* date) noexcept nogil
    c_bool c_ex_profile_cn_is_circuit_break(const session_date_t* date) noexcept nogil
    void c_ex_profile_cn_get_calendar() noexcept nogil



cpdef double local_utc_offset_seconds()
cdef int c_ex_profile_unix_to_datetime(double unix_ts, session_datetime_t* out)
cpdef py_datetime unix_to_datetime(double unix_ts)


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


cdef class SessionDateStandalone:
    cdef const session_date_t* header
    cdef bint owner

    @staticmethod
    cdef SessionDateStandalone c_from_header(const session_date_t* header, bint owner)

    cpdef SessionDateStandalone fork(self)


cdef class SessionDate(py_date):
    cdef const session_date_t* header
    cdef bint owner

    @staticmethod
    cdef SessionDate c_from_header(const session_date_t* header, bint owner)

    cdef const session_date_t* c_sync(self)

    cpdef SessionDate fork(self)


cdef class SessionDateRange:
    cdef const session_date_range_t* header
    cdef bint owner

    cdef readonly SessionDate start_date
    cdef readonly SessionDate end_date
    cdef readonly tuple dates

    @staticmethod
    cdef SessionDateRange c_from_header(const session_date_range_t* header, bint owner)

    cpdef list to_list(self)


cdef class SessionDateTime:
    cdef session_datetime_t* header
    cdef bint owner

    cdef readonly SessionTime time
    cdef readonly SessionDate date

    @staticmethod
    cdef SessionDateTime c_from_header(session_datetime_t* header, bint owner=?)


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
    cdef uintptr_t listener_id

    cdef readonly str profile_id
    cdef readonly SessionTime session_start
    cdef readonly SessionTime session_end
    cdef readonly CallAuction open_call_auction
    cdef readonly CallAuction close_call_auction
    cdef readonly tuple[SessionBreak] session_breaks
    cdef readonly object time_zone

    @staticmethod
    cdef ExchangeProfile c_new_bound_instance()

    @staticmethod
    cdef ExchangeProfile c_from_header(const exchange_profile* header)

    @staticmethod
    cdef void c_listener_adapter(const exchange_profile* previous_profile, const exchange_profile* new_profile, void* user_data) noexcept

    cdef inline void c_bind(self, const exchange_profile* header)

    cdef inline bint c_time_in_market_session(self, py_time t)

    cdef inline bint c_timestamp_in_market_session(self, double unix_ts)

    cdef inline bint c_timestamp_in_auction_session(self, double unix_ts)

    cdef inline SessionDateRange c_trade_calendar(self, py_date start_date, py_date end_date)

    cdef inline bint c_date_in_market_session(self, py_date market_date)

    cdef inline py_datetime c_timestamp_to_datetime(self, double unix_ts)

    cdef inline py_date c_timestamp_to_date(self, double unix_ts)

    cdef inline double c_time_to_seconds(self, py_time t, bint break_adjusted)

    cdef inline double c_timestamp_to_seconds(self, double t, bint break_adjusted)

    cdef inline double c_break_adjusted(self, double ts)

    cdef inline double c_trading_time_between(self, py_datetime start_time, py_datetime end_time)

    cdef inline py_date c_trading_days_before(self, py_date market_date, size_t days)

    cdef inline py_date c_trading_days_after(self, py_date market_date, size_t days)

    cdef inline ssize_t c_trading_days_between(self, py_date start_date, py_date end_date)

    cdef inline py_date c_nearest_trading_date(self, py_date market_date, bint previous)


cdef ExchangeProfile PROFILE
cdef ExchangeProfile PROFILE_DEFAULT
cdef ExchangeProfile PROFILE_CN
