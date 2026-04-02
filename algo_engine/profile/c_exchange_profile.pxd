from libcpp cimport bool as c_bool
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

    ctypedef void (*c_ex_profile_on_activate)(exchange_profile* profile) noexcept nogil
    ctypedef void (*c_ex_profile_on_deactivate)(exchange_profile* profile) noexcept nogil
    ctypedef session_date_range_t* (*c_ex_profile_trade_calendar)(session_date_t* start_date, session_date_t* end_date) noexcept nogil
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
        call_auction* open_call_auction
        call_auction* close_call_auction
        session_break* session_breaks
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

    double c_utc_offset_seconds() noexcept nogil
    int c_ex_profile_time_compare(const void* t1, const void* t2) noexcept nogil
    double c_ex_profile_time_to_ts(uint8_t hour, uint8_t minute, uint8_t second, uint32_t nanosecond) noexcept nogil
    double c_ex_profile_unix_to_ts(double unix_ts) noexcept nogil
    double c_ex_profile_ts_to_elapsed(double elapsed) noexcept nogil

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
    session_date_range_t* c_ex_profile_date_range(const session_date_t* start_date, const session_date_t* end_date) noexcept nogil

    void c_ex_profile_activate(exchange_profile* profile) noexcept nogil

    session_time_t* c_ex_profile_session_time_new(uint8_t hour, uint8_t minute, uint8_t second, uint32_t nanosecond) noexcept nogil
    int c_ex_profile_session_time_from_ts(double unix_ts, session_time_t* out) noexcept nogil
    session_time_range_t* c_ex_profile_session_trange_between_time(const session_time_t* start_time, const session_time_t* end_time) noexcept nogil
    session_time_range_t* c_ex_profile_session_trange_between_ts(double start_unix_ts, double end_unix_ts) noexcept nogil

    session_date_t* c_ex_profile_session_date_new(uint16_t year, uint8_t month, uint8_t day) noexcept nogil
    int c_ex_profile_session_date_from_ts(double unix_ts, session_date_t* out) noexcept nogil
    size_t c_ex_profile_session_date_index(const session_date_t* date, const session_date_range_t* drange) noexcept nogil
    session_date_range_t* c_ex_profile_session_drange_between(const session_date_t* start_date, const session_date_t* end_date) noexcept nogil
    int c_ex_profile_session_trading_days_before(const session_date_t* market_date, size_t days, session_date_t* out) noexcept nogil
    int c_ex_profile_session_trading_days_after(const session_date_t* market_date, size_t days, session_date_t* out) noexcept nogil
    int c_ex_profile_nearest_trading_date(const session_date_t* market_date, c_bool previous, session_date_t* out) noexcept nogil


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
