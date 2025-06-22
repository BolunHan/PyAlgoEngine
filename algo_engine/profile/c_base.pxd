from cpython.datetime cimport time, datetime, date
from libc.stdint cimport uint8_t

cdef ProfileDispatcher C_PROFILE
cdef Profile C_PROFILE_DEFAULT

ctypedef list c_trade_calendar(date start_date, date end_date)

ctypedef bint c_timestamp_in_market_session(double t)

ctypedef bint c_time_in_market_session(time t)

ctypedef bint c_date_in_market_session(date t)


cdef class ProfileDispatcher:
    cdef dict __dict__

    cdef public str profile_id
    cdef public object session_start
    cdef public object session_end
    cdef public object time_zone
    cdef public list session_break

    cdef double session_start_ts
    cdef double session_end_ts
    cdef uint8_t session_break_num
    cdef double* session_break_start
    cdef double* session_break_length
    cdef public double tz_offset
    cdef public double session_length

    cdef c_trade_calendar* c_trade_calendar
    cdef c_timestamp_in_market_session* c_timestamp_in_market_session
    cdef c_time_in_market_session* c_time_in_market_session
    cdef c_date_in_market_session* c_date_in_market_session

    cdef void c_refresh_cached_values(self)

    cdef inline double c_time_to_seconds(self, time t, bint break_adjusted)

    cdef inline double c_timestamp_to_seconds(self, double t, bint break_adjusted)

    cdef inline double c_break_adjusted(self, double elapsed_seconds)

    cdef double c_trading_time_between(self, datetime start_time, datetime end_time)

    cdef date c_trading_days_before(self, date market_date, size_t days)

    cdef date c_trading_days_after(self, date market_date, size_t days)

    cdef size_t c_trading_days_between(self, date start_date, date end_date)

    cdef date c_nearest_trading_date(self, date market_date, bint previous=*)

    cpdef double time_to_seconds(self, time t, bint break_adjusted=*)

    cpdef double timestamp_to_seconds(self, double t, bint break_adjusted=*)

    cpdef double break_adjusted(self, double elapsed_seconds)

    cpdef double trading_time_between(self, object start_time, object end_time)

    cpdef bint is_market_session(self, object timestamp)

    cpdef list trade_calendar(self, date start_date, date end_date)

    cpdef date trading_days_before(self, date market_date, int days)

    cpdef date trading_days_after(self, date market_date, int days)

    cpdef size_t trading_days_between(self, date start_date, date end_date)

    cpdef date nearest_trading_date(self, date market_date, str method=*)

    cpdef bint is_trading_day(self, date market_date)


cdef class Profile:
    cdef public str profile_id
    cdef public time session_start
    cdef public time session_end
    cdef public list session_break
    cdef public object time_zone

    @staticmethod
    cdef list c_trade_calendar(date start_date, date end_date)

    @staticmethod
    cdef bint c_timestamp_in_market_session(double t)

    @staticmethod
    cdef bint c_time_in_market_session(time t)

    @staticmethod
    cdef bint c_date_in_market_session(date t)

    cdef void c_override_meta(self, ProfileDispatcher profile)

    cdef void c_override_func_ptr(self, ProfileDispatcher profile)

    cpdef ProfileDispatcher override_profile(self)
