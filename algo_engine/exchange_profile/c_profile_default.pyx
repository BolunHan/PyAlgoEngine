cdef ExchangeProfile PROFILE_DEFAULT = ExchangeProfile.c_from_header(&EX_PROFILE_DEFAULT)

globals()['PROFILE_DEFAULT'] = PROFILE_DEFAULT
