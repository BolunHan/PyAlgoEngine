from cpython.datetime cimport time, datetime, date, time_hour, time_minute, time_second, time_microsecond
from libc.math cimport fmod

from .c_base cimport ProfileDispatcher

C_PROFILE_CN = ProfileCN()
PROFILE_CN = C_PROFILE_CN


cdef class ProfileCN(Profile):
    def __cinit__(self):
        self.profile_id = 'cn'
        self.session_start = time(9, 30)
        self.session_end = time(15, 0)
        self.session_break = [(time(11, 30), time(13, 0))]
        self.time_zone = None
        self.tz_offset = datetime.now().astimezone().tzinfo.utcoffset(None).total_seconds()

        self.trade_calendar_cache = None
        self.func_cache_c_trade_calendar = {}
        self.func_cache_c_date_in_market_session = {}

    def __init__(self):
        import exchange_calendars
        self.trade_calendar_cache = exchange_calendars.get_calendar('XSHG')

    @staticmethod
    cdef list c_trade_calendar(date start_date, date end_date):
        cdef list calendar
        cdef tuple key = (start_date, end_date)
        cdef object result = C_PROFILE_CN.func_cache_c_trade_calendar.get(key)

        if result is None:
            calendar = list(_.date() for _ in C_PROFILE_CN.trade_calendar_cache.sessions_in_range(start_date, end_date))
            C_PROFILE_CN.func_cache_c_trade_calendar[key] = calendar.copy()
            return calendar

        calendar = result[:]
        return calendar

    @staticmethod
    cdef bint c_timestamp_in_market_session(double t):
        cdef double elapsed_seconds = fmod(t + C_PROFILE_CN.tz_offset, 86400.0)

        if 34200.0 <= elapsed_seconds <= 41400.0:
            return True
        elif 46800.0 <= elapsed_seconds <= 54000.0:
            return True
        else:
            return False

    @staticmethod
    cdef bint c_time_in_market_session(time t):
        cdef double elapsed_seconds = (time_hour(t) * 60 + time_minute(t)) * 60 + time_second(t) + time_microsecond(t) / 1_000_000

        if 34200.0 <= elapsed_seconds <= 41400.0:
            return True
        elif 46800.0 <= elapsed_seconds <= 54000.0:
            return True
        else:
            return False

    @staticmethod
    cdef bint c_date_in_market_session(date t):
        cdef dict cache = C_PROFILE_CN.func_cache_c_date_in_market_session
        cdef bint is_trade_day

        if t in cache:
            is_trade_day = cache[t]
        else:
            is_trade_day = cache[t] = C_PROFILE_CN.trade_calendar_cache.is_session(t)

        return is_trade_day

    cdef void c_override_func_ptr(self, ProfileDispatcher profile):
        self.set_process_timezone()

        profile.c_trade_calendar = ProfileCN.c_trade_calendar
        profile.c_timestamp_in_market_session = ProfileCN.c_timestamp_in_market_session
        profile.c_time_in_market_session = ProfileCN.c_time_in_market_session
        profile.c_date_in_market_session = ProfileCN.c_date_in_market_session

    def set_process_timezone(self):
        import os
        import time

        if os.name != 'posix':
            return

        os.environ['TZ'] = "Asia/Shanghai"
        time.tzset()
