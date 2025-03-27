# cython: language_level=3
from libc.stdint cimport uint8_t, int32_t, uint32_t, uint64_t

# Declare external constants
cdef extern from "market_data_external.c":
    int compare_md(const void* a, const void* b) nogil
    int compare_md_ptr(const void* a, const void* b) nogil
    int compare_entries_bid(const void * a, const void * b) nogil
    int compare_entries_ask(const void * a, const void * b) nogil
    const int TICKER_SIZE
    const int BOOK_SIZE
    const int ID_SIZE

    ctypedef enum Direction:
        DIRECTION_UNKNOWN
        DIRECTION_SHORT
        DIRECTION_LONG

    ctypedef enum Offset:
        OFFSET_CANCEL
        OFFSET_ORDER
        OFFSET_OPEN
        OFFSET_CLOSE

    ctypedef enum Side:
        SIDE_LONG_OPEN
        SIDE_LONG_CLOSE
        SIDE_LONG_CANCEL
        SIDE_SHORT_OPEN
        SIDE_SHORT_CLOSE
        SIDE_SHORT_CANCEL
        SIDE_BID
        SIDE_ASK
        SIDE_CANCEL
        SIDE_UNKNOWN
        SIDE_LONG
        SIDE_SHORT

cdef public enum OrderType:
    ORDER_UNKNOWN = -20
    ORDER_CANCEL = -10
    ORDER_GENERIC = 0
    ORDER_LIMIT = 10
    ORDER_LIMIT_MAKER = 11
    ORDER_MARKET = 20
    ORDER_FOK = 21
    ORDER_FAK = 22
    ORDER_IOC = 23

# Data type mapping
cdef public enum DataType:
    DTYPE_UNKNOWN = 0
    DTYPE_MARKET_DATA = 10
    DTYPE_TRANSACTION = 20
    DTYPE_ORDER = 30
    DTYPE_TICK_LITE = 31
    DTYPE_TICK = 32
    DTYPE_BAR = 40

cdef struct _ID:
    uint8_t id_type
    char data[ID_SIZE]

# Meta info structure
cdef struct _MetaInfo:
    uint8_t dtype
    char ticker[TICKER_SIZE]
    double timestamp

# OrderBookEntry structure
cdef struct _OrderBookEntry:
    double price
    double volume
    uint64_t n_orders

# OrderBookBuffer structure
cdef struct _OrderBookBuffer:
    _OrderBookEntry entries[BOOK_SIZE]

# BarData structure
cdef struct _CandlestickBuffer:
    uint8_t dtype
    char ticker[TICKER_SIZE]
    double timestamp
    double bar_span
    double high_price
    double low_price
    double open_price
    double close_price
    double volume
    double notional
    uint32_t trade_count

# TickDataLite structure
cdef struct _TickDataLiteBuffer:
    uint8_t dtype
    char ticker[TICKER_SIZE]
    double timestamp
    double bid_price
    double bid_volume
    double ask_price
    double ask_volume
    double last_price
    double total_traded_volume
    double total_traded_notional
    uint32_t total_trade_count

# TickData structure
cdef struct _TickDataBuffer:
    _TickDataLiteBuffer lite
    _OrderBookBuffer bid
    _OrderBookBuffer ask

# TransactionData structure
cdef struct _TransactionDataBuffer:
    uint8_t dtype
    char ticker[TICKER_SIZE]
    double timestamp
    double price
    double volume
    int32_t side
    double multiplier
    double notional
    _ID transaction_id
    _ID buy_id
    _ID sell_id

# OrderData structure
cdef struct _OrderDataBuffer:
    uint8_t dtype
    char ticker[TICKER_SIZE]
    double timestamp
    double price
    double volume
    int32_t side
    _ID order_id
    int32_t order_type

# Base MarketData structure as a union
cdef union _MarketDataBuffer:
    _MetaInfo MetaInfo
    _TransactionDataBuffer TransactionData
    _OrderDataBuffer OrderData
    _CandlestickBuffer BarData
    _TickDataLiteBuffer TickDataLite
    _TickDataBuffer TickDataFull

# Declare MarketData class
cdef class MarketData:

    cdef _MarketDataBuffer * _data
    cdef bint _owner
    cdef public dict _additional

    cdef void _set_additional(self, str name, object value)

    @staticmethod
    cdef size_t get_size(uint8_t dtype)

    @staticmethod
    cdef size_t min_size()

    @staticmethod
    cdef size_t max_size()

    cpdef bytes to_bytes(self)

    # the property of MarketData
    # cdef str ticker
    # cdef datetime market_time
    # cdef float timestamp
    # cdef float market_price
    # cdef int dtype
