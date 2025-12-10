# cython: language_level=3
from cpython.datetime cimport datetime
from libc.stdint cimport int8_t, uint8_t, int32_t, uint32_t, uint64_t, uintptr_t


cdef extern from "c_market_data_config.h":
    const bint DEBUG
    const size_t TICKER_SIZE
    const size_t BOOK_SIZE
    const size_t ID_SIZE
    const size_t LONG_ID_SIZE
    const size_t MAX_WORKERS


# Declare external constants
cdef extern from "c_market_data.h":

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

    ctypedef enum id_type_t:
        ID_UNKNOWN
        ID_INT
        ID_STRING
        ID_BYTE
        ID_UUID

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
        char* ticker
        double timestamp

    ctypedef struct id_t:
        id_type_t id_type
        char data[ID_SIZE]

    ctypedef struct long_id_t:
        id_type_t id_type
        char data[LONG_ID_SIZE]

    ctypedef struct internal_buffer_t:
        meta_info_t meta_info
        uint32_t code

    ctypedef struct order_book_entry_t:
        double price
        double volume
        uint64_t n_orders

    ctypedef struct order_book_buffer_t:
        order_book_entry_t entries[BOOK_SIZE]

    ctypedef struct candlestick_buffer_t:
        meta_info_t meta_info
        double bar_span
        double high_price
        double low_price
        double open_price
        double close_price
        double volume
        double notional
        uint64_t trade_count

    ctypedef struct tick_data_lite_buffer_t:
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

    ctypedef struct tick_data_buffer_t:
        tick_data_lite_buffer_t lite
        double total_bid_volume
        double total_ask_volume
        double weighted_bid_price
        double weighted_ask_price
        order_book_buffer_t bid
        order_book_buffer_t ask

    ctypedef struct transaction_data_buffer_t:
        meta_info_t meta_info
        double price
        double volume
        side_t side
        double multiplier
        double notional
        id_t transaction_id
        id_t buy_id
        id_t sell_id

    ctypedef struct order_data_buffer_t:
        meta_info_t meta_info
        double price
        double volume
        side_t side
        id_t order_id
        order_type_t order_type

    ctypedef struct trade_report_buffer_t:
        meta_info_t meta_info
        double price
        double volume
        side_t side
        double multiplier
        double notional
        double fee
        long_id_t order_id
        long_id_t trade_id

    ctypedef struct trade_instruction_buffer_t:
        meta_info_t meta_info
        double limit_price
        double volume
        side_t side
        long_id_t order_id
        order_type_t order_type
        double multiplier
        order_state_t order_state
        double filled_volume
        double filled_notional
        double fee
        double ts_placed
        double ts_canceled
        double ts_finished

    ctypedef union market_data_buffer_t:
        meta_info_t meta_info
        internal_buffer_t internal
        transaction_data_buffer_t transaction_data
        order_data_buffer_t order_data
        candlestick_buffer_t bar_data
        tick_data_lite_buffer_t tick_data_lite
        tick_data_buffer_t tick_data_full
        trade_report_buffer_t trade_report
        trade_instruction_buffer_t trade_instruction

    int8_t direction_to_sign(direction_t x) noexcept nogil
    void platform_usleep(unsigned int usec) noexcept nogil
    int compare_md_ptr(const void* a, const void* b) noexcept nogil
    int compare_entries_bid(const void* a, const void* b) noexcept nogil
    int compare_entries_ask(const void* a, const void* b) noexcept nogil


# Declare MarketData class
cdef class _MarketDataVirtualBase:
    cdef dict __dict__
    cdef market_data_buffer_t* _data_ptr

    @staticmethod
    cdef inline size_t c_get_size(uint8_t dtype)

    @staticmethod
    cdef inline str c_dtype_name(uint8_t dtype)

    @staticmethod
    cdef inline object c_ptr_to_data(market_data_buffer_t* data_ptr)

    @staticmethod
    cdef inline bytes c_ptr_to_bytes(market_data_buffer_t* data_ptr)

    @staticmethod
    cdef inline size_t c_max_size()

    @staticmethod
    cdef inline size_t c_min_size()

    @staticmethod
    cdef inline datetime c_to_dt(double timestamp)


cdef class FilterMode:
    cdef public uint32_t value

    @staticmethod
    cdef inline bint c_mask_data(meta_info_t* data_addr, uint32_t filter_mode)
