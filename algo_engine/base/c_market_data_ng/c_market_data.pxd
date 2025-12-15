import uuid

from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.unicode cimport PyUnicode_FromString, PyUnicode_AsUTF8AndSize
from libc.stdint cimport int8_t, uint32_t, UINT64_MAX, INT64_MIN, int64_t, uint64_t, uintptr_t
from libc.string cimport memset, memcpy, strcpy, strlen

from ..c_heap_allocator cimport heap_allocator_t, C_ALLOCATOR as HEAP_ALLOCATOR
from ..c_shm_allocator cimport shm_allocator_ctx, shm_allocator_t, C_ALLOCATOR as SHM_ALLOCATOR
from ..c_intern_string cimport C_POOL as SHM_POOL, C_INTRA_POOL as HEAP_POOL, c_istr, c_istr_synced


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

    const size_t DTYPE_MIN_SIZE
    const size_t DTYPE_MAX_SIZE

    ctypedef unsigned long long uint128_t
    ctypedef long long int128_t
    const int128_t INT128_MIN
    const uint128_t UINT128_MAX

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
        MID_UINT128
        MID_INT128
        MID_UINT64
        MID_INT64
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
        char data[ID_SIZE + 1]

    ctypedef struct long_mid_t:
        mid_type_t id_type
        char data[LONG_ID_SIZE + 1]

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
    offset_t c_md_side_offset(side_t side) noexcept nogil
    direction_t c_md_side_direction(side_t side) noexcept nogil
    side_t c_md_side_opposite(side_t side) noexcept nogil
    int8_t c_md_side_sign(side_t side) noexcept nogil
    size_t c_md_get_size(data_type_t dtype) noexcept nogil
    const char* c_md_dtype_name(data_type_t dtype) noexcept nogil
    const char* c_md_side_name(side_t side) noexcept nogil
    const char* c_md_order_type_name(order_type_t order_type) noexcept nogil
    const char* c_md_direction_name(side_t side) noexcept nogil
    const char* c_md_offset_name(side_t side) noexcept nogil
    size_t c_md_serialized_size(const market_data_t* market_data)
    size_t c_md_serialize(const market_data_t* market_data, char* out)
    market_data_t* c_md_deserialize(const char* src, shm_allocator_ctx* shm_allocator, heap_allocator_t* heap_allocator, int with_lock)
    int c_md_compare_ptr(const void* a, const void* b) noexcept nogil
    int c_md_compare_bid(const void* a, const void* b) noexcept nogil
    int c_md_compare_ask(const void* a, const void* b) noexcept nogil
    int c_md_compare_id(const mid_t* id1, const mid_t* id2) noexcept nogil
    int c_md_compare_long_id(const long_mid_t* id1, const long_mid_t* id2) noexcept nogil


cdef bint MD_CFG_LOCKED
cdef bint MD_CFG_SHARED
cdef bint MD_CFG_FREELIST


cdef class EnvConfigContext:
    cdef dict overrides
    cdef dict originals

    cdef void c_activate(self)

    cdef void c_deactivate(self)


cdef EnvConfigContext MD_SHARED
cdef EnvConfigContext MD_LOCKED
cdef EnvConfigContext MD_FREELIST


cdef inline market_data_t* c_init_buffer(data_type_t dtype, const char* ticker, double timestamp):
    cdef market_data_t* market_data

    if MD_CFG_SHARED:
        market_data = c_md_new(dtype, SHM_ALLOCATOR, NULL, <int> MD_CFG_LOCKED)
    elif MD_CFG_FREELIST:
        market_data = c_md_new(dtype, NULL, HEAP_ALLOCATOR, <int> MD_CFG_LOCKED)
    else:
        market_data = c_md_new(dtype, NULL, NULL, 0)

    if not market_data:
        raise MemoryError(f'Failed to allocate shared memory for {PyUnicode_FromString(c_md_dtype_name(dtype))}')
    cdef meta_info_t* meta_data = <meta_info_t*> market_data

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


cdef inline market_data_t* c_deserialize_buffer(const char* src):
    if MD_CFG_SHARED:
        market_data = c_md_deserialize(src, SHM_ALLOCATOR, NULL, <int> MD_CFG_LOCKED)
    elif MD_CFG_FREELIST:
        market_data = c_md_deserialize(src, NULL, HEAP_ALLOCATOR, <int> MD_CFG_LOCKED)
    else:
        market_data = c_md_deserialize(src, NULL, NULL, 0)

    if not market_data:
        raise MemoryError('Failed to deserialize market data from bytes')
    cdef meta_info_t* meta_data = <meta_info_t*> market_data

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


cdef inline void c_recycle_buffer(market_data_t* market_data):
    c_md_free(market_data, <int> MD_CFG_LOCKED)


cdef inline void c_set_id(mid_t* id_ptr, object id_value):
    cdef bytes id_bytes
    cdef const char* id_chars
    cdef Py_ssize_t id_len

    memset(<void*> id_ptr.data, 0, ID_SIZE + 1)

    if id_value is None:
        # None type
        id_ptr.id_type = mid_type_t.MID_UNKNOWN
    elif isinstance(id_value, int):
        if id_value >= 0:
            if id_value < UINT64_MAX and MID_ALLOW_INT64:
                # uint64_t type
                id_ptr.id_type = mid_type_t.MID_UINT64
                (<uint64_t*> &id_ptr.data[0])[0] = <uint64_t> id_value
                return
            elif id_value < UINT128_MAX and MID_ALLOW_INT128:
                # uint128_t type
                id_ptr.id_type = mid_type_t.MID_UINT128
                (<uint128_t*> &id_ptr.data[0])[0] = <uint128_t> id_value
                return
            else:
                raise ValueError(f'Integer ID {id_value} is too large to fit in the ID buffer.')
        else:
            if id_value > INT64_MIN and MID_ALLOW_INT64:
                id_ptr.id_type = mid_type_t.MID_INT64
                (<int64_t*> &id_ptr.data[0])[0] = <int64_t> id_value
                return
            elif id_value > INT128_MIN and MID_ALLOW_INT128:
                id_ptr.id_type = mid_type_t.MID_INT128
                (<int128_t*> &id_ptr.data[0])[0] = <int128_t> id_value
                return
            else:
                raise ValueError(f'Integer ID {id_value} is too small to fit in the ID buffer.')
    elif isinstance(id_value, str):
        # str type
        id_ptr.id_type = mid_type_t.MID_STRING
        id_chars = PyUnicode_AsUTF8AndSize(id_value, &id_len)
        if id_len <= ID_SIZE:
            memcpy(<void*> id_ptr.data, id_chars, id_len)
            return
        else:
            raise ValueError(f'String ID {id_value} is too long to fit in the ID buffer.')
    elif isinstance(id_value, bytes):
        # bytes type
        id_ptr.id_type = mid_type_t.MID_BYTE
        id_chars = <const char*> id_value
        id_len = len(id_value)
        if id_len <= ID_SIZE:
            memcpy(<void*> id_ptr.data, id_chars, id_len)
            return
        else:
            raise ValueError(f'Byte ID {id_value} is too long to fit in the ID buffer.')
    elif isinstance(id_value, uuid.UUID):
        # uuid type
        id_ptr.id_type = mid_type_t.MID_UUID
        id_bytes = id_value.bytes_le
        if MID_ALLOW_INT128:
            memcpy(<void*> id_ptr.data, <const char*> id_bytes, 16)
            return
        else:
            raise ValueError(f'UUID ID {id_value} is too long to fit in the ID buffer.')


cdef inline object c_get_id(mid_t* id_ptr):
    if id_ptr.id_type == mid_type_t.MID_UNKNOWN:
        return None
    elif id_ptr.id_type == mid_type_t.MID_UINT64:
        return (<uint64_t*> id_ptr.data)[0]
    elif id_ptr.id_type == mid_type_t.MID_INT64:
        return (<int64_t*> id_ptr.data)[0]
    elif id_ptr.id_type == mid_type_t.MID_UINT128:
        return (<uint128_t*> id_ptr.data)[0]
    elif id_ptr.id_type == mid_type_t.MID_INT128:
        return (<int128_t*> id_ptr.data)[0]
    elif id_ptr.id_type == mid_type_t.MID_STRING:
        return PyUnicode_FromString(&id_ptr.data[0])
    elif id_ptr.id_type == mid_type_t.MID_BYTE:
        return PyBytes_FromStringAndSize(&id_ptr.data[0], ID_SIZE).rstrip(b'\0')
    elif id_ptr.id_type == mid_type_t.MID_UUID:
        return uuid.UUID(bytes_le=id_ptr.data[:16])
    raise ValueError(f'Cannot decode the id buffer with type {id_ptr.id_type}.')


cdef inline void c_set_long_id(long_mid_t* id_ptr, object id_value):
    cdef bytes id_bytes
    cdef const char* id_chars
    cdef Py_ssize_t id_len

    memset(<void*> id_ptr.data, 0, LONG_ID_SIZE + 1)

    if id_value is None:
        # None type
        id_ptr.id_type = mid_type_t.MID_UNKNOWN
    elif isinstance(id_value, int):
        if id_value >= 0:
            if id_value < UINT64_MAX and LONG_MID_ALLOW_INT64:
                # uint64_t type
                id_ptr.id_type = mid_type_t.MID_UINT64
                (<uint64_t*> &id_ptr.data[0])[0] = <uint64_t> id_value
                return
            elif id_value < UINT128_MAX and LONG_MID_ALLOW_INT128:
                # uint128_t type
                id_ptr.id_type = mid_type_t.MID_UINT128
                (<uint128_t*> &id_ptr.data[0])[0] = <uint128_t> id_value
                return
            else:
                raise ValueError(f'Integer ID {id_value} is too large to fit in the ID buffer.')
        else:
            if id_value > INT64_MIN and LONG_MID_ALLOW_INT64:
                id_ptr.id_type = mid_type_t.MID_INT64
                (<int64_t*> &id_ptr.data[0])[0] = <int64_t> id_value
                return
            elif id_value > INT128_MIN and LONG_MID_ALLOW_INT128:
                id_ptr.id_type = mid_type_t.MID_INT128
                (<int128_t*> &id_ptr.data[0])[0] = <int128_t> id_value
                return
            else:
                raise ValueError(f'Integer ID {id_value} is too small to fit in the ID buffer.')
    elif isinstance(id_value, str):
        # str type
        id_ptr.id_type = mid_type_t.MID_STRING
        id_chars = PyUnicode_AsUTF8AndSize(id_value, &id_len)
        if id_len <= LONG_ID_SIZE:
            memcpy(<void*> id_ptr.data, id_chars, id_len)
            return
        else:
            raise ValueError(f'String ID {id_value} is too long to fit in the ID buffer.')
    elif isinstance(id_value, bytes):
        # bytes type
        id_ptr.id_type = mid_type_t.MID_BYTE
        id_chars = <const char*> id_value
        id_len = len(id_value)
        if id_len <= LONG_ID_SIZE:
            memcpy(<void*> id_ptr.data, id_chars, id_len)
            return
        else:
            raise ValueError(f'Byte ID {id_value} is too long to fit in the ID buffer.')
    elif isinstance(id_value, uuid.UUID):
        # uuid type
        id_ptr.id_type = mid_type_t.MID_UUID
        id_bytes = id_value.bytes_le
        if LONG_MID_ALLOW_INT128:
            memcpy(<void*> id_ptr.data, <const char*> id_bytes, 16)
            return
        else:
            raise ValueError(f'UUID ID {id_value} is too long to fit in the ID buffer.')


cdef inline object c_get_long_id(long_mid_t* id_ptr):
    if id_ptr.id_type == mid_type_t.MID_UNKNOWN:
        return None
    elif id_ptr.id_type == mid_type_t.MID_UINT64:
        return (<uint64_t*> id_ptr.data)[0]
    elif id_ptr.id_type == mid_type_t.MID_INT64:
        return (<int64_t*> id_ptr.data)[0]
    elif id_ptr.id_type == mid_type_t.MID_UINT128:
        return (<uint128_t*> id_ptr.data)[0]
    elif id_ptr.id_type == mid_type_t.MID_INT128:
        return (<int128_t*> id_ptr.data)[0]
    elif id_ptr.id_type == mid_type_t.MID_STRING:
        return PyUnicode_FromString(&id_ptr.data[0])
    elif id_ptr.id_type == mid_type_t.MID_BYTE:
        return PyBytes_FromStringAndSize(&id_ptr.data[0], LONG_ID_SIZE).rstrip(b'\0')
    elif id_ptr.id_type == mid_type_t.MID_UUID:
        return uuid.UUID(bytes_le=id_ptr.data[:16])
    raise ValueError(f'Cannot decode the id buffer with type {id_ptr.id_type}.')


ctypedef object (*c_from_header_func)(market_data_t* market_data, bint owner)

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

    cdef market_data_t* header

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
