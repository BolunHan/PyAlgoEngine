from cpython.unicode cimport PyUnicode_FromString
from libc.stdint cimport int8_t, uint32_t, uint64_t, uintptr_t
from libc.string cimport memcpy

from ..c_heap_allocator cimport heap_allocator, C_ALLOCATOR as HEAP_ALLOCATOR
from ..c_shm_allocator cimport shm_allocator_ctx, shm_allocator, C_ALLOCATOR as SHM_ALLOCATOR
from ..c_intern_string cimport C_POOL as SHM_POOL, C_INTRA_POOL as HEAP_POOL, c_istr, c_istr_synced

from ...profile.c_base cimport C_PROFILE


cdef extern from "c_market_data_config.h":
    const bint DEBUG
    const size_t TICKER_SIZE
    const size_t BOOK_SIZE
    const size_t ID_SIZE
    const size_t LONG_ID_SIZE
    const size_t MAX_WORKERS

    const int MID_ALLOW_INT64
    const int MID_ALLOW_INT128
    const int LONG_MID_ALLOW_INT64
    const int LONG_MID_ALLOW_INT128


# Declare external constants
cdef extern from "c_market_data.h":
    const char* dtype_name_internal
    const char* dtype_name_transaction
    const char* dtype_name_order
    const char* dtype_name_tick_lite
    const char* dtype_name_tick
    const char* dtype_name_bar
    const char* dtype_name_report
    const char* dtype_name_instruction
    const char* dtype_name_generic

    const char* side_name_open
    const char* side_name_close
    const char* side_name_short
    const char* side_name_cover
    const char* side_name_bid
    const char* side_name_ask
    const char* side_name_cancel
    const char* side_name_cancel_bid
    const char* side_name_cancel_ask
    const char* side_name_neutral_open
    const char* side_name_neutral_close
    const char* side_name_unknown

    const char* order_name_unknown
    const char* order_name_cancel
    const char* order_name_generic
    const char* order_name_limit
    const char* order_name_limit_maker
    const char* order_name_market
    const char* order_name_fok
    const char* order_name_fak
    const char* order_name_ioc

    const char* direction_name_short
    const char* direction_name_long
    const char* direction_name_neutral
    const char* direction_name_unknown

    const char* offset_name_cancel
    const char* offset_name_order
    const char* offset_name_open
    const char* offset_name_close
    const char* offset_name_unknown

    const char* state_name_unknown
    const char* state_name_rejected
    const char* state_name_invalid
    const char* state_name_pending
    const char* state_name_sent
    const char* state_name_placed
    const char* state_name_partfilled
    const char* state_name_filled
    const char* state_name_canceling
    const char* state_name_canceled

    const size_t DTYPE_MIN_SIZE
    const size_t DTYPE_MAX_SIZE

    ctypedef unsigned long long uint128_t
    ctypedef long long int128_t
    const int128_t INT128_MIN
    const uint128_t UINT128_MAX

    ctypedef enum md_direction:
        DIRECTION_UNKNOWN
        DIRECTION_SHORT
        DIRECTION_LONG
        DIRECTION_NEUTRAL

    ctypedef enum md_offset:
        OFFSET_CANCEL
        OFFSET_ORDER
        OFFSET_OPEN
        OFFSET_CLOSE

    ctypedef enum md_side:
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

    ctypedef enum md_order_type:
        ORDER_UNKNOWN
        ORDER_CANCEL
        ORDER_GENERIC
        ORDER_LIMIT
        ORDER_LIMIT_MAKER
        ORDER_MARKET
        ORDER_FOK
        ORDER_FAK
        ORDER_IOC

    ctypedef enum md_order_state:
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

    ctypedef enum md_id_type:
        MID_UNKNOWN
        MID_UINT128
        MID_INT128
        MID_UINT64
        MID_INT64
        MID_STRING
        MID_BYTE
        MID_UUID

    ctypedef enum md_data_type:
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

    ctypedef enum md_filter_flag:
        NO_INTERNAL
        NO_CANCEL
        NO_AUCTION
        NO_ORDER
        NO_TRADE
        NO_TICK

    ctypedef struct md_meta:
        md_data_type dtype
        const char* ticker
        double timestamp
        shm_allocator* shm_allocator
        heap_allocator* heap_allocator

    ctypedef struct md_id:
        md_id_type id_type
        char data[ID_SIZE + 1]

    ctypedef struct long_md_id:
        md_id_type id_type
        char data[LONG_ID_SIZE + 1]

    ctypedef struct md_internal:
        md_meta meta_info
        uint32_t code

    ctypedef struct md_orderbook_entry:
        double price
        double volume
        uint64_t n_orders

    ctypedef struct md_orderbook:
        size_t capacity;
        size_t size;
        md_direction direction
        int sorted
        shm_allocator* shm_allocator
        heap_allocator* heap_allocator
        md_orderbook_entry entries[]

    ctypedef struct md_candlestick:
        md_meta meta_info
        double bar_span
        double high_price
        double low_price
        double open_price
        double close_price
        double volume
        double notional
        uint64_t trade_count

    ctypedef struct md_tick_data_lite:
        md_meta meta_info
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

    ctypedef struct md_tick_data:
        md_tick_data_lite lite
        double total_bid_volume
        double total_ask_volume
        double weighted_bid_price
        double weighted_ask_price
        md_orderbook* bid
        md_orderbook* ask

    ctypedef struct md_transaction_data:
        md_meta meta_info
        double price
        double volume
        md_side side
        double multiplier
        double notional
        md_id transaction_id
        md_id buy_id
        md_id sell_id

    ctypedef struct md_order_data:
        md_meta meta_info
        double price
        double volume
        md_side side
        md_id order_id
        md_order_type order_type

    ctypedef struct md_trade_report:
        md_meta meta_info
        double price
        double volume
        md_side side
        double multiplier
        double notional
        double fee
        long_md_id order_id
        long_md_id trade_id

    ctypedef struct md_trade_instruction:
        md_meta meta_info
        double limit_price
        double volume
        md_side side
        long_md_id order_id
        md_order_type order_type
        double multiplier
        md_order_state order_state
        double filled_volume
        double filled_notional
        double fee
        double ts_placed
        double ts_canceled
        double ts_finished

    ctypedef union md_variant:
        md_meta meta_info
        md_internal internal
        md_transaction_data transaction_data
        md_order_data order_data
        md_candlestick bar_data
        md_tick_data_lite tick_data_lite
        md_tick_data tick_data_full
        md_trade_report trade_report
        md_trade_instruction trade_instruction

    void c_usleep(unsigned int usec) noexcept nogil
    md_variant* c_md_new(md_data_type dtype, shm_allocator_ctx* shm_allocator, heap_allocator* heap_allocator, int with_lock)
    void c_md_free(md_variant* market_data, int with_lock)
    double c_md_get_price(const md_variant* market_data) noexcept nogil
    md_offset c_md_side_offset(md_side side) noexcept nogil
    md_direction c_md_side_direction(md_side side) noexcept nogil
    md_side c_md_side_opposite(md_side side) noexcept nogil
    int8_t c_md_side_sign(md_side side) noexcept nogil
    size_t c_md_get_size(md_data_type dtype) noexcept nogil
    const char* c_md_dtype_name(md_data_type dtype) noexcept nogil
    const char* c_md_side_name(md_side side) noexcept nogil
    const char* c_md_order_type_name(md_order_type order_type) noexcept nogil
    const char* c_md_direction_name(md_side side) noexcept nogil
    const char* c_md_offset_name(md_side side) noexcept nogil
    const char* c_md_state_name(md_order_state side) noexcept nogil
    size_t c_md_serialized_size(const md_variant* market_data)
    size_t c_md_serialize(const md_variant* market_data, char* out)
    md_variant* c_md_deserialize(const char* src, shm_allocator_ctx* shm_allocator, heap_allocator* heap_allocator, int with_lock) noexcept nogil
    md_orderbook* c_md_orderbook_new(size_t book_size, shm_allocator_ctx* shm_allocator, heap_allocator* heap_allocator, int with_lock) noexcept nogil
    void c_md_orderbook_free(md_orderbook* orderbook, int with_lock) noexcept nogil
    int c_md_orderbook_sort(md_orderbook* orderbook) noexcept nogil
    int c_md_state_working(md_order_state state) noexcept nogil
    int c_md_state_placed(md_order_state state) noexcept nogil
    int c_md_state_done(md_order_state state) noexcept nogil
    int c_md_compare_ptr(const void* a, const void* b) noexcept nogil
    int c_md_compare_bid(const void* a, const void* b) noexcept nogil
    int c_md_compare_ask(const void* a, const void* b) noexcept nogil
    int c_md_compare_id(const md_id* id1, const md_id* id2) noexcept nogil
    int c_md_compare_long_id(const long_md_id* id1, const long_md_id* id2) noexcept nogil


cdef bint MD_CFG_LOCKED
cdef bint MD_CFG_SHARED
cdef bint MD_CFG_FREELIST
cdef size_t MD_CFG_BOOK_SIZE


cdef class EnvConfigContext:
    cdef dict overrides
    cdef dict originals

    cdef void c_activate(self)

    cdef void c_deactivate(self)


cdef EnvConfigContext MD_SHARED
cdef EnvConfigContext MD_LOCKED
cdef EnvConfigContext MD_FREELIST
cdef EnvConfigContext MD_BOOK5
cdef EnvConfigContext MD_BOOK10
cdef EnvConfigContext MD_BOOK20


cdef inline md_variant* c_init_buffer(md_data_type dtype, const char* ticker, double timestamp):
    cdef md_variant* market_data

    if MD_CFG_SHARED:
        market_data = c_md_new(dtype, SHM_ALLOCATOR, NULL, <int> MD_CFG_LOCKED)
    elif MD_CFG_FREELIST:
        market_data = c_md_new(dtype, NULL, HEAP_ALLOCATOR, <int> MD_CFG_LOCKED)
    else:
        market_data = c_md_new(dtype, NULL, NULL, 0)

    if not market_data:
        raise MemoryError(f'Failed to allocate shared memory for {PyUnicode_FromString(c_md_dtype_name(dtype))}')
    cdef md_meta* meta_data = <md_meta*> market_data

    if MD_CFG_SHARED:
        if MD_CFG_LOCKED:
            meta_data.ticker = c_istr_synced(SHM_POOL, ticker)
        else:
            meta_data.ticker = c_istr(SHM_POOL, ticker)
    else:
        if MD_CFG_LOCKED:
            meta_data.ticker = c_istr_synced(HEAP_POOL, ticker)
        else:
            meta_data.ticker = c_istr(HEAP_POOL, ticker)

    if not meta_data.ticker:
        raise MemoryError('Failed to intern ticker string')

    meta_data.dtype = dtype
    meta_data.timestamp = timestamp
    return market_data


cdef inline md_variant* c_deserialize_buffer(const char* src):
    if MD_CFG_SHARED:
        market_data = c_md_deserialize(src, SHM_ALLOCATOR, NULL, <int> MD_CFG_LOCKED)
    elif MD_CFG_FREELIST:
        market_data = c_md_deserialize(src, NULL, HEAP_ALLOCATOR, <int> MD_CFG_LOCKED)
    else:
        market_data = c_md_deserialize(src, NULL, NULL, 0)

    if not market_data:
        raise MemoryError('Failed to deserialize market data from bytes')
    cdef md_meta* meta_data = <md_meta*> market_data

    cdef const char* ticker = meta_data.ticker
    if not ticker:
        raise ValueError('Deserialized market data has null ticker string')

    if MD_CFG_SHARED:
        if MD_CFG_LOCKED:
            meta_data.ticker = c_istr_synced(SHM_POOL, ticker)
        else:
            meta_data.ticker = c_istr(SHM_POOL, ticker)
    else:
        if MD_CFG_LOCKED:
            meta_data.ticker = c_istr_synced(HEAP_POOL, ticker)
        else:
            meta_data.ticker = c_istr(HEAP_POOL, ticker)

    if not meta_data.ticker:
        raise MemoryError('Failed to intern ticker string')

    return market_data


cdef inline void c_recycle_buffer(md_variant* market_data):
    c_md_free(market_data, <int> MD_CFG_LOCKED)


cdef inline void c_write_uint128(void* data, uint128_t value):
    memcpy(data, &value, 16)


cdef inline uint128_t c_read_uint128(void* data):
    cdef uint128_t value
    memcpy(&value, data, 16)
    return value


cdef inline void c_write_int128(void* data, int128_t value):
    memcpy(data, &value, 16)


cdef inline int128_t c_read_int128(void* data):
    cdef int128_t value
    memcpy(&value, data, 16)
    return value


cdef void c_set_id(md_id* id_ptr, object id_value)
cdef object c_get_id(md_id* id_ptr)
cdef void c_set_long_id(long_md_id* id_ptr, object id_value)
cdef object c_get_long_id(long_md_id* id_ptr)


ctypedef object (*c_from_header_func)(md_variant* market_data, bint owner)

cdef c_from_header_func internal_from_header
cdef c_from_header_func transaction_from_header
cdef c_from_header_func order_from_header
cdef c_from_header_func tick_lite_from_header
cdef c_from_header_func tick_from_header
cdef c_from_header_func bar_from_header
cdef c_from_header_func report_from_header
cdef c_from_header_func instruction_from_header


cdef class MarketData:
    cdef dict __dict__
    cdef readonly uintptr_t data_addr
    cdef readonly bint owner

    cdef md_variant* header

    @staticmethod
    cdef inline object c_from_header(md_variant* market_data, bint owner=?)

    cdef inline size_t c_get_size(self)

    cdef inline str c_dtype_name(self)

    cdef inline void c_to_bytes(self, char* out)

    @staticmethod
    cdef inline md_variant* c_from_bytes(bytes data)


cdef class FilterMode:
    cdef public md_filter_flag value

    @staticmethod
    cdef inline bint c_mask_data(md_variant* market_data, md_filter_flag filter_mode)

    cpdef bint mask_data(self, MarketData market_data)


cdef class ConfigViewer:
    pass
