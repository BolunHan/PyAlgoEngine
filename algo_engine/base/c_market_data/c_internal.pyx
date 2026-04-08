from cpython.unicode cimport PyUnicode_AsUTF8
from libc.stdint cimport uintptr_t

from .c_market_data cimport c_init_buffer, md_data_type


cdef class InternalData(MarketData):
    def __init__(
            self,
            *,
            str ticker,
            double timestamp,
            uint32_t code,
            **kwargs
    ):
        self.header = c_init_buffer(
            md_data_type.DTYPE_INTERNAL,
            PyUnicode_AsUTF8(ticker),
            timestamp
        )

        self.header.internal.code = code

        self.data_addr = <uintptr_t> self.header
        self.owner = True

        if kwargs:
            self.__dict__.update(kwargs)

    def __repr__(self):
        return f"<{self.__class__.__name__}>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker}, code={self.code})"

    property code:
        def __get__(self):
            return self.header.internal.code


from . cimport c_market_data

c_market_data.internal_from_header = internal_from_header
