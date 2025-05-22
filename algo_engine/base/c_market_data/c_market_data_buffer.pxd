# cython: language_level=3
from libc.stdint cimport uint8_t, uint32_t, uint64_t

from .c_market_data cimport MarketData, MAX_WORKERS

# Define buffer header structure
cdef struct _BufferHeader:
    uint8_t dtype                # Data type (0 for mixed)
    bint sorted                  # Sorted flag
    uint32_t count               # Number of entries
    uint32_t current_index       # Current index for iteration
    uint64_t pointer_offset      # Offset to pointer array
    uint64_t capacity            # Maximum number of pointers that can be stored
    uint64_t data_offset         # Offset to data section
    uint64_t tail_offset         # Current offset at the end of used data
    uint64_t max_offset          # Maximum offset (size of data section)
    double current_timestamp     # Current maximum timestamp


cdef packed struct _RingBufferHeader:
    _BufferHeader buffer_header  # Base header
    uint64_t head                # Index of oldest element
    uint64_t tail                # Index where next element will be added


cdef packed struct _ConcurrentBufferHeader:
    _BufferHeader buffer_header  # Base header
    uint64_t heads[MAX_WORKERS]  # Array of head indices (one per worker)
    uint64_t tail                # Index where next element will be added
    uint8_t n_workers           # Actual number of workers


cdef class MarketDataBuffer:
    cdef char * _buffer  # Pointer to the buffer
    cdef Py_buffer _view  # Buffer view
    cdef bint _view_obtained  # Flag to track if buffer view was obtained
    cdef _BufferHeader * _header  # Pointer to the header section
    cdef uint64_t * _offsets  # Pointer to the pointer array section
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


cdef class MarketDataRingBuffer:
    cdef char * _buffer  # Pointer to the buffer
    cdef Py_buffer _view  # Buffer view
    cdef bint _view_obtained  # Flag to track if buffer view was obtained
    cdef _RingBufferHeader* _header  # Pointer to the header section
    cdef uint64_t * _offsets  # Pointer to the pointer array section
    cdef char * _data  # Pointer to the data section

    cdef uint64_t data_head(self)

    cdef uint64_t data_tail(self)

    cpdef bint is_empty(self)

    cpdef bint is_full(self)

    @staticmethod
    cdef bytes read(char* data, uint64_t offset, size_t size, uint64_t max_offset)

    @staticmethod
    cdef void write(char* data, uint64_t offset, char* src, size_t size, uint64_t max_offset)

    cpdef void put(self, MarketData market_data)

    cpdef MarketData get(self, uint64_t idx)

    cpdef MarketData listen(self)


cdef class MarketDataConcurrentBuffer:
    cdef char * _buffer  # Pointer to the buffer
    cdef Py_buffer _view  # Buffer view
    cdef bint _view_obtained  # Flag to track if buffer view was obtained
    cdef _ConcurrentBufferHeader* _header  # Pointer to the header section
    cdef uint64_t * _offsets  # Pointer to the pointer array section
    cdef char * _data  # Pointer to the data section

    cdef uint64_t data_head(self)
    cdef uint64_t data_tail(self)

    cpdef uint64_t get_head(self, uint32_t worker_id)

    cpdef uint64_t min_head(self)

    cpdef bint is_empty(self, uint32_t worker_id)

    cpdef bint is_empty_all(self)

    cpdef bint is_full(self)

    cpdef void put(self, MarketData market_data)

    cpdef MarketData get(self, uint64_t idx)

    cpdef MarketData listen(self, uint32_t worker_id, bint block=?, double timeout=?)