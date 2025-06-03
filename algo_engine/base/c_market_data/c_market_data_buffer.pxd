# cython: language_level=3
from libc.stdint cimport uint8_t, uint32_t, uint64_t

from .c_market_data cimport _MarketDataVirtualBase, _MarketDataBuffer


cdef packed struct _BufferHeader:
    bint sorted                    # Sorted flag
    uint32_t ptr_capacity          # Maximum number of pointers that can be stored ~4.2GB
    uint32_t ptr_offset            # Offset to find the pointer array
    uint32_t ptr_tail              # Index where the pointer of the next element will be added
    uint64_t data_capacity         # Maximum of data that can be stored, ~18.4EB
    uint64_t data_offset           # Offset to data section
    uint64_t data_tail             # Offset where the data of the next element will be added
    double current_timestamp       # Current maximum timestamp


cdef packed struct _RingBufferHeader:
    uint32_t ptr_capacity
    uint32_t ptr_offset
    uint32_t ptr_head              # Index of oldest element
    uint32_t ptr_tail
    uint64_t data_capacity
    uint64_t data_offset
    uint64_t data_tail


cdef packed struct _WorkerHeader:
    uint32_t ptr_head              # Index of the next pointer to read


cdef struct _ConcurrentBufferHeader:
    uint8_t n_workers              # number of workers
    uint32_t worker_header_offset  # Offset to find the worker header section
    uint32_t ptr_capacity
    uint32_t ptr_offset
    uint32_t ptr_tail
    uint64_t data_capacity
    uint64_t data_offset
    uint64_t data_tail
    # the next session of buffer should be the head array, all these sections are not included in the header. and can be regenerated once the header is determined.
    # uint64_t heads[n_workers]
    # then the pointer array
    # uint64_t ptr[ptr_capacity]
    # then the data buffer
    # char data[data_capacity]


cdef class MarketDataBuffer:
    cdef _BufferHeader* _header    # Pointer to the header section
    cdef char* _buffer             # Pointer to the buffer
    cdef Py_buffer _view           # Buffer view
    cdef bint _view_obtained       # Flag to track if buffer view was obtained
    cdef uint32_t _ptr_capacity
    cdef uint64_t* _ptr_array
    cdef size_t _estimated_entry_size
    cdef uint64_t _data_capacity
    cdef char* _data_array
    cdef size_t _idx

    @staticmethod
    cdef size_t c_buffer_size(uint32_t n_internal_data=*, uint32_t n_transaction_data=*, uint32_t n_order_data=*, uint32_t n_tick_data_lite=*, uint32_t n_tick_data=*, uint32_t n_bar_data=*)

    @staticmethod
    cdef MarketDataBuffer c_from_buffer(object buffer)

    cdef bytes c_to_bytes(self)

    @staticmethod
    cdef c_from_bytes(bytes data, object buffer=*)

    cdef void c_put(self, _MarketDataBuffer* market_data_ptr)

    cdef object c_get(self, uint32_t idx)

    @staticmethod
    cdef void _set_internal_fields(void* buffer, uint32_t code)

    @staticmethod
    cdef void _set_transaction_fields(void* buffer, double price, double volume, uint8_t side, double multiplier=*, double notional=*, object transaction_id=*, object buy_id=*, object sell_id=*)

    @staticmethod
    cdef void _set_order_fields(void* buffer, double price, double volume, uint8_t side, object order_id=*, uint8_t order_type=*)

    @staticmethod
    cdef void _set_tick_lite_fields(void* buffer, double last_price, double bid_price, double bid_volume, double ask_price, double ask_volume, double total_traded_volume=*, double total_traded_notional=*, uint32_t total_trade_count=*)

    @staticmethod
    cdef void _set_tick_fields(void* buffer)

    @staticmethod
    cdef void _set_bar_fields(void* buffer, double high_price, double low_price, double open_price, double close_price, double bar_span, double volume=*, double notional=*, uint32_t trade_count=*)

    cdef void c_sort(self)


cdef class MarketDataRingBuffer:
    cdef _RingBufferHeader* _header
    cdef char* _buffer
    cdef Py_buffer _view
    cdef bint _view_obtained
    cdef uint32_t _ptr_capacity
    cdef uint64_t* _ptr_array
    cdef size_t _estimated_entry_size
    cdef uint64_t _data_capacity
    cdef char* _data_array

    cdef bint c_is_empty(self)

    cdef uint32_t c_get_ptr_distance(self, uint32_t ptr_idx)

    cpdef bint c_is_full(self)

    cdef void c_write(self, const char* data, uint64_t length)

    cdef void c_read(self, uint64_t data_offset, uint64_t length, char* output)

    cdef bytes c_to_bytes(self, uint64_t data_offset, uint64_t length)

    cdef void c_put(self, _MarketDataBuffer* market_data_ptr)

    cdef object c_get(self, uint32_t idx)

    cdef object c_listen(self, bint block=*, double timeout=*)


cdef class MarketDataConcurrentBuffer:
    cdef _ConcurrentBufferHeader* _header
    cdef char* _buffer
    cdef Py_buffer _view
    cdef bint _view_obtained
    cdef uint8_t n_workers
    cdef uint32_t* _worker_header_array
    cdef uint32_t _ptr_capacity
    cdef uint64_t* _ptr_array
    cdef size_t _estimated_entry_size
    cdef uint64_t _data_capacity
    cdef char* _data_array

    cdef uint32_t c_get_worker_head(self, uint32_t worker_id) except -1

    cdef uint32_t c_get_ptr_distance(self, uint32_t ptr_idx)

    cdef uint32_t c_get_ptr_head(self)

    cdef uint64_t c_get_data_head(self)

    cdef bint c_is_worker_empty(self, uint32_t worker_id) except -1

    cdef bint c_is_empty(self)

    cdef bint c_is_full(self)

    cdef void c_write(self, const char* data, uint64_t length)

    cdef void c_read(self, uint64_t data_offset, uint64_t length, char* output)

    cdef bytes c_to_bytes(self, uint64_t data_offset, uint64_t length)

    cdef void c_put(self, _MarketDataBuffer* market_data_ptr)

    cdef object c_get(self, uint32_t idx)

    cdef object c_listen(self, uint32_t worker_id, bint block=*, double timeout=*)
