from .c_exchange_profile cimport exchange_profile, ExchangeProfile


cdef extern from "c_ex_profile_base.h":
    extern const exchange_profile* EX_PROFILE


cdef ExchangeProfile PROFILE
