import enum
from collections import namedtuple

from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.unicode cimport PyUnicode_FromString
from libc.string cimport memcpy

from ..c_shm_allocator cimport C_ALLOCATOR
from ...profile.c_base cimport C_PROFILE


class DataType(enum.IntEnum):
    DTYPE_UNKNOWN = data_type_t.DTYPE_UNKNOWN
    DTYPE_INTERNAL = data_type_t.DTYPE_INTERNAL
    DTYPE_MARKET_DATA = data_type_t.DTYPE_MARKET_DATA
    DTYPE_TRANSACTION = data_type_t.DTYPE_TRANSACTION
    DTYPE_ORDER = data_type_t.DTYPE_ORDER
    DTYPE_TICK_LITE = data_type_t.DTYPE_TICK_LITE
    DTYPE_TICK = data_type_t.DTYPE_TICK
    DTYPE_BAR = data_type_t.DTYPE_BAR
    DTYPE_REPORT = data_type_t.DTYPE_REPORT
    DTYPE_INSTRUCTION = data_type_t.DTYPE_INSTRUCTION


cdef inline market_data_t* c_init_buffer(data_type_t dtype, bint with_lock):
    cdef market_data_t* market_data = c_md_new(dtype, C_ALLOCATOR, <int> with_lock)
    if not market_data:
        raise MemoryError(f'Failed to allocate shared memory for {PyUnicode_FromString(c_md_dtype_name(dtype))}')
    return market_data


cdef inline void c_recycle_buffer(market_data_t* market_data, bint with_lock):
    c_md_free(market_data, C_ALLOCATOR, <int> with_lock)


cdef class MarketData:

    @staticmethod
    cdef inline object c_from_header(market_data_t* market_data):
        cdef data_type_t dtype = market_data.meta_info.dtype
        cdef size_t length = c_md_get_size(dtype)
        cdef MarketData instance

        if dtype == data_type_t.DTYPE_INTERNAL:
            instance = <MarketData> InternalData.__new__(InternalData)
        elif dtype == data_type_t.DTYPE_TRANSACTION:
            instance = <MarketData> TransactionData.__new__(TransactionData)
        elif dtype == data_type_t.DTYPE_ORDER:
            instance = <MarketData> OrderData.__new__(OrderData)
        elif dtype == data_type_t.DTYPE_TICK_LITE:
            instance = <MarketData> TickDataLite.__new__(TickDataLite)
        elif dtype == data_type_t.DTYPE_TICK:
            instance = <MarketData> TickData.__new__(TickData)
        elif dtype == data_type_t.DTYPE_BAR:
            bar_data = BarData.__new__(BarData)
        elif dtype == data_type_t.DTYPE_REPORT:
            trade_report = TradeReport.__new__(TradeReport)
        elif dtype == data_type_t.DTYPE_INSTRUCTION:
            trade_order = TradeInstruction.__new__(TradeInstruction)
        else:
            raise ValueError(f'Unknown data type {dtype}')

        memcpy(<void*> instance.header, <const void*> market_data, length)
        if dtype == data_type_t.DTYPE_TICK:
            (<TickData> instance).c_init_order_book()
        return instance

    cdef inline size_t c_get_size(self):
        cdef size_t size = c_md_get_size(self.header.meta_info.dtype)
        if not size:
            raise ValueError(f'Unknown data type {dtype}')
        return size

    cdef inline str c_dtype_name(self):
        cdef const char* dtype_name = c_md_dtype_name(self.header.meta_info.dtype)
        if not dtype_name:
            raise ValueError(f'Unknown data type {dtype}')
        return PyUnicode_FromString(dtype_name)

    cdef inline bytes c_to_bytes(self):
        cdef size_t size = c_md_serialized_size(self.header)
        cdef bytes data = PyBytes_FromStringAndSize(NULL, size)
        c_md_serialize(self.header, <char*> data)
        return data

    property ticker:
        def __get__(self):
            return PyUnicode_FromString(self.header.meta_info.ticker)

    property timestamp:
        def __get__(self):
            return self.header.meta_info.timestamp

    property dtype:
        def __get__(self):
            return DataType(self.header.meta_info.dtype)

    property topic:
        def __get__(self):
            return f'{self.ticker}.{self.c_dtype_name()}'

    property market_time:
        def __get__(self):
            return datetime.fromtimestamp(timestamp, tz=C_PROFILE.time_zone)

    property market_price:
        def __get__(self):
            return c_md_get_price(self.header)


cdef class FilterMode:
    # Class-level constants for flags
    NO_INTERNAL = FilterMode.__new__(FilterMode, filter_mode_t.NO_INTERNAL)
    NO_CANCEL = FilterMode.__new__(FilterMode, filter_mode_t.NO_CANCEL)
    NO_AUCTION = FilterMode.__new__(FilterMode, filter_mode_t.NO_AUCTION)
    NO_ORDER = FilterMode.__new__(FilterMode, filter_mode_t.NO_ORDER)
    NO_TRADE = FilterMode.__new__(FilterMode, filter_mode_t.NO_TRADE)
    NO_TICK = FilterMode.__new__(FilterMode, filter_mode_t.NO_TICK)

    def __cinit__(self, filter_mode_t value=0):
        self.value = value

    def __or__(self, FilterMode other):
        return FilterMode.__new__(FilterMode, self.value | other.value)

    def __and__(self, FilterMode other):
        return FilterMode.__new__(FilterMode, self.value & other.value)

    def __invert__(self):
        # Invert all bits except those beyond our known flags
        inverted_value = ~self.value & (
            filter_mode_t.NO_INTERNAL |
            filter_mode_t.NO_CANCEL |
            filter_mode_t.NO_AUCTION |
            filter_mode_t.NO_ORDER |
            filter_mode_t.NO_TRADE |
            filter_mode_t.NO_TICK
        )
        return FilterMode.__new__(FilterMode, inverted_value)

    def __contains__(self, FilterMode other):
        return (self.value & other.value) == other.value

    def __repr__(self):
        flags = []
        if filter_mode_t.NO_INTERNAL & self.value: flags.append("NO_INTERNAL")
        if filter_mode_t.NO_CANCEL & self.value: flags.append("NO_CANCEL")
        if filter_mode_t.NO_AUCTION & self.value: flags.append("NO_AUCTION")
        if filter_mode_t.NO_ORDER & self.value: flags.append("NO_ORDER")
        if filter_mode_t.NO_TRADE & self.value: flags.append("NO_TRADE")
        if filter_mode_t.NO_TICK & self.value: flags.append("NO_TICK")
        return f"<FilterMode {self.value:#0x}: {' | '.join(flags) or 'None'}>"

    @staticmethod
    cdef inline bint c_mask_data(market_data_t* market_data, filter_mode_t filter_mode):
        cdef data_type_t dtype = market_data.meta_info.dtype
        cdef side_t side

        if filter_mode_t.NO_INTERNAL & filter_mode:
            if dtype == data_type_t.DTYPE_INTERNAL:
                return False

        if filter_mode_t.NO_CANCEL & filter_mode:
            if dtype == data_type_t.DTYPE_TRANSACTION:
                side = (<transaction_data_t*> market_data).side
                if c_md_get_offset(side) == offset_t.OFFSET_CANCEL:
                    return False

        cdef double timestamp = market_data.meta_info.timestamp

        if filter_mode_t.NO_AUCTION & filter_mode:
            if not C_PROFILE.c_timestamp_in_market_session(timestamp):
                return False

        if filter_mode_t.NO_ORDER & filter_mode:
            if dtype == data_type_t.DTYPE_ORDER:
                return False

        if filter_mode_t.NO_TRADE & filter_mode:
            if dtype == data_type_t.DTYPE_TRANSACTION:
                return False

        if filter_mode_t.NO_TICK & filter_mode:
            if dtype == data_type_t.DTYPE_TICK or dtype == data_type_t.DTYPE_TICK_LITE:
                return False

        return True

    @classmethod
    def all(cls):
        return FilterMode.__new__(
            FilterMode,
            filter_mode_t.NO_INTERNAL |
            filter_mode_t.NO_CANCEL |
            filter_mode_t.NO_AUCTION |
            filter_mode_t.NO_ORDER |
            filter_mode_t.NO_TRADE |
            filter_mode_t.NO_TICK
        )

    cpdef bint mask_data(self, MarketData market_data):
        return FilterMode.c_mask_data(market_data.header, self.value)


from .c_internal cimport InternalData
from .c_tick cimport TickData, TickDataLite
from .c_transaction cimport TransactionData, OrderData, TransactionHelper
from .c_candlestick cimport BarData
from .c_trade_utils cimport TradeReport, TradeInstruction

C_CONFIG = namedtuple('CONFIG', ['TICKER_SIZE', 'BOOK_SIZE', 'ID_SIZE', 'MAX_WORKERS'])(
    DEBUG=DEBUG,
    TICKER_SIZE=TICKER_SIZE,
    BOOK_SIZE=BOOK_SIZE,
    ID_SIZE=ID_SIZE,
    LONG_ID_SIZE=LONG_ID_SIZE,
    MAX_WORKERS=MAX_WORKERS
)
