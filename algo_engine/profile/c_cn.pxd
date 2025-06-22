from .c_base cimport Profile


cdef ProfileCN C_PROFILE_CN


cdef class ProfileCN(Profile):
    cdef double tz_offset
    cdef object trade_calendar_cache
    cdef dict func_cache_c_trade_calendar
    cdef dict func_cache_c_date_in_market_session
