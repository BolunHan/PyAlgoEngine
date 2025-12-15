from cpython.datetime cimport datetime
from libc.stdint cimport int8_t, uint32_t, uint64_t, uintptr_t

from ..c_heap_allocator cimport heap_allocator_t
from ..c_shm_allocator cimport shm_allocator_ctx, shm_allocator_t


cdef extern from "c_market_data_config.h":
    const bint DEBUG
    const size_t TICKER_SIZE
    const size_t BOOK_SIZE
    const size_t ID_SIZE
    const size_t LONG_ID_SIZE
    const size_t MAX_WORKERS


# Declare external constants
cdef extern from "c_market_data.h":
    const char* internal_dtype_name
    const char* transaction_dtype_name
    const char* order_dtype_name
    const char* tick_lite_dtype_name
    const char* tick_dtype_name
    const char* bar_dtype_name
    const char* report_dtype_name
    const char* instruction_dtype_name
    const char* generic_dtype_name

    const size_t DTYPE_MIN_SIZE
    const size_t DTYPE_MAX_SIZE

    ctypedef enum direction_t:
        DIRECTION_UNKNOWN
        DIRECTION_SHORT
        DIRECTION_LONG
        DIRECTION_NEUTRAL

    ctypedef enum offset_t:
        OFFSET_CANCEL
        OFFSET_ORDER
        OFFSET_OPEN
        OFFSET_CLOSE

    ctypedef enum side_t:
        SIDE_LONG_OPEN
        SIDE_LONG_CLOSE
        SIDE_LONG_CANCEL
        SIDE_SHORT_OPEN
        SIDE_SHORT_CLOSE
        SIDE_SHORT_CANCEL
        SIDE_NEUTRAL_OPEN
        SIDE_NEUTRAL_CLOSE
        SIDE_BID
        SIDE_ASK
        SIDE_CANCEL
        SIDE_UNKNOWN
        SIDE_LONG
        SIDE_SHORT

    ctypedef enum order_type_t:
        ORDER_UNKNOWN
        ORDER_CANCEL
        ORDER_GENERIC
        ORDER_LIMIT
        ORDER_LIMIT_MAKER
        ORDER_MARKET
        ORDER_FOK
        ORDER_FAK
        ORDER_IOC

    ctypedef enum order_state_t:
        STATE_UNKNOWN
        STATE_REJECTED
        STATE_INVALID
        STATE_PENDING
        STATE_SENT
        STATE_PLACED
        STATE_PARTFILLED
        STATE_FILLED
        STATE_CANCELING
        STATE_CANCELED

    ctypedef enum mid_type_t:
        MID_UNKNOWN
        MID_INT
        MID_STRING
        MID_BYTE
        MID_UUID

    ctypedef enum data_type_t:
        DTYPE_UNKNOWN
        DTYPE_INTERNAL
        DTYPE_MARKET_DATA
        DTYPE_TRANSACTION
        DTYPE_ORDER
        DTYPE_TICK_LITE
        DTYPE_TICK
        DTYPE_BAR
        DTYPE_REPORT
        DTYPE_INSTRUCTION

    ctypedef enum filter_mode_t:
        NO_INTERNAL
        NO_CANCEL
        NO_AUCTION
        NO_ORDER
        NO_TRADE
        NO_TICK

    ctypedef struct meta_info_t:
        data_type_t dtype
        const char* ticker
        double timestamp
        shm_allocator_t* shm_allocator
        heap_allocator_t* heap_allocator

    ctypedef struct mid_t:
        mid_type_t id_type
        char data[ID_SIZE]

    ctypedef struct long_mid_t:
        mid_type_t id_type
        char data[LONG_ID_SIZE]

    ctypedef struct internal_t:
        meta_info_t meta_info
        uint32_t code

    ctypedef struct order_book_entry_t:
        double price
        double volume
        uint64_t n_orders

    ctypedef struct order_book_t:
        order_book_entry_t entries[BOOK_SIZE]

    ctypedef struct candlestick_t:
        meta_info_t meta_info
        double bar_span
        double high_price
        double low_price
        double open_price
        double close_price
        double volume
        double notional
        uint64_t trade_count

    ctypedef struct tick_data_lite_t:
        meta_info_t meta_info
        double bid_price
        double bid_volume
        double ask_price
        double ask_volume
        double last_price
        double open_price
        double prev_close
        double total_traded_volume
        double total_traded_notional
        uint64_t total_trade_count

    ctypedef struct tick_data_t:
        tick_data_lite_t lite
        double total_bid_volume
        double total_ask_volume
        double weighted_bid_price
        double weighted_ask_price
        order_book_t bid
        order_book_t ask

    ctypedef struct transaction_data_t:
        meta_info_t meta_info
        double price
        double volume
        side_t side
        double multiplier
        double notional
        mid_t transaction_id
        mid_t buy_id
        mid_t sell_id

    ctypedef struct order_data_t:
        meta_info_t meta_info
        double price
        double volume
        side_t side
        mid_t order_id
        order_type_t order_type

    ctypedef struct trade_report_t:
        meta_info_t meta_info
        double price
        double volume
        side_t side
        double multiplier
        double notional
        double fee
        long_mid_t order_id
        long_mid_t trade_id

    ctypedef struct trade_instruction_t:
        meta_info_t meta_info
        double limit_price
        double volume
        side_t side
        long_mid_t order_id
        order_type_t order_type
        double multiplier
        order_state_t order_state
        double filled_volume
        double filled_notional
        double fee
        double ts_placed
        double ts_canceled
        double ts_finished

    ctypedef union market_data_t:
        meta_info_t meta_info
        internal_t internal
        transaction_data_t transaction_data
        order_data_t order_data
        candlestick_t bar_data
        tick_data_lite_t tick_data_lite
        tick_data_t tick_data_full
        trade_report_t trade_report
        trade_instruction_t trade_instruction

    void c_usleep(unsigned int usec) noexcept nogil
    market_data_t* c_md_new(data_type_t dtype, shm_allocator_ctx* shm_allocator, heap_allocator_t* heap_allocator, int with_lock)
    void c_md_free(market_data_t* market_data, int with_lock)
    double c_md_get_price(const market_data_t* market_data) noexcept nogil
    offset_t c_md_get_offset(side_t side) noexcept nogil
    direction_t c_md_get_direction(side_t side) noexcept nogil
    int8_t c_md_get_sign(direction_t x) noexcept nogil
    size_t c_md_get_size(data_type_t dtype) noexcept nogil
    const char* c_md_dtype_name(data_type_t dtype) noexcept nogil
    size_t c_md_serialized_size(const market_data_t* market_data)
    size_t c_md_serialize(const market_data_t* market_data, char* out)
    market_data_t* c_md_deserialize(const char* src, shm_allocator_ctx* shm_allocator, heap_allocator_t* heap_allocator, int with_lock)
    int c_md_compare_ptr(const void* a, const void* b) noexcept nogil
    int c_md_compare_bid(const void* a, const void* b) noexcept nogil
    int c_md_compare_ask(const void* a, const void* b) noexcept nogil


cdef bint MD_CFG_LOCKED
cdef bint MD_CFG_SHARED
cdef bint MD_CFG_FREELIST


cdef class EnvConfigContext:
    cdef dict overrides
    cdef dict originals

    cdef void c_activate(self)

    cdef void c_deactivate(self)


cdef EnvConfigContext CONFIG_SHARED
cdef EnvConfigContext MD_LOCAL
cdef EnvConfigContext MD_LOCKED
cdef EnvConfigContext MD_UNLOCKED


cdef inline market_data_t* c_init_buffer(data_type_t dtype, const char* ticker, double timestamp)

cdef inline market_data_t* c_deserialize_buffer(const char* src)

cdef inline void c_recycle_buffer(market_data_t* market_data)


# Declare MarketData class
cdef class MarketData:
    cdef dict __dict__
    cdef readonly uintptr_t data_addr

    cdef market_data_t* header
    cdef bint owner

    @staticmethod
    cdef inline object c_from_header(market_data_t* market_data, bint owner=*)

    cdef inline size_t c_get_size(self)

    cdef inline str c_dtype_name(self)

    cdef inline void c_to_bytes(self, char* out)

    @staticmethod
    cdef inline market_data_t* c_from_bytes(bytes data)


cdef class FilterMode:
    cdef public filter_mode_t value

    @staticmethod
    cdef inline bint c_mask_data(market_data_t* market_data, filter_mode_t filter_mode)

    cpdef bint mask_data(self, MarketData market_data)
