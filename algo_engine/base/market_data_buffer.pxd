from libc.stdint cimport uint8_t, uint32_t, uint64_t

from .market_data cimport MarketData

# Define buffer header structure
cdef struct _BufferHeader:
    uint8_t dtype           # Data type (0 for mixed)
    bint sorted          # Sorted flag
    uint32_t count          # Number of entries
    uint32_t current_index  # Current index for iteration
    uint64_t pointer_offset # Offset to pointer array
    uint64_t max_entries   # Maximum number of pointers that can be stored
    uint64_t data_offset    # Offset to data section
    uint64_t tail_offset    # Current offset at the end of used data
    uint64_t max_offset     # Maximum offset (size of data section)
    double current_timestamp # Current maximum timestamp


cdef class MarketDataBuffer:
    cdef char * _buffer  # Pointer to the buffer
    cdef Py_buffer _view  # Buffer view
    cdef bint _view_obtained  # Flag to track if buffer view was obtained
    cdef _BufferHeader * _header  # Pointer to the header section
    cdef uint64_t * _pointers  # Pointer to the pointer array section
    cdef char * _data  # Pointer to the data section

    cpdef void push(self, MarketData market_data)

    cdef void _set_fields(self, char * buffer, uint8_t dtype, dict kwargs)

    cpdef void sort(self, bint inplace=?)

    cpdef bytes to_bytes(self)

    @staticmethod
    cdef void _set_transaction_fields(char * buffer, dict kwargs)

    @staticmethod
    cdef void _set_order_fields(char * buffer, dict kwargs)

    @staticmethod
    cdef void _set_tick_lite_fields(char * buffer, dict kwargs)

    @staticmethod
    cdef void _set_tick_fields(char * buffer, dict kwargs)

    @staticmethod
    cdef void _set_bar_fields(char * buffer, dict kwargs)