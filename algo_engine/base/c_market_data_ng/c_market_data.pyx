import enum
from collections import namedtuple

from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.unicode cimport PyUnicode_FromString
from libc.string cimport memcpy

from ..c_intern_string cimport C_POOL as SHM_POOL, C_INTRA_POOL as HEAP_POOL, c_istr, c_istr_synced
from ..c_heap_allocator cimport C_ALLOCATOR as HEAP_ALLOCATOR
from ..c_shm_allocator cimport C_ALLOCATOR as SHM_ALLOCATOR
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


cdef bint MD_CFG_LOCKED = False
cdef bint MD_CFG_SHARED = True
cdef bint MD_CFG_FREELIST = True


cdef class EnvConfigContext:
    def __cinit__(self, **kwargs):
        self.overrides = kwargs
        self.originals = {}

    cdef void c_activate(self):
        if 'locked' in self.overrides:
            global MD_CFG_LOCKED
            MD_CFG_LOCKED = self.overrides['locked']
            self.originals['locked'] = MD_CFG_LOCKED

        if 'shared' in self.overrides:
            global MD_CFG_SHARED
            MD_CFG_SHARED = self.overrides['shared']
            self.originals['shared'] = MD_CFG_SHARED

        if 'freelist' in self.overrides:
            global MD_CFG_FREELIST
            MD_CFG_FREELIST = self.overrides['freelist']
            self.originals['freelist'] = MD_CFG_FREELIST

    cdef void c_deactivate(self):
        if 'locked' in self.originals:
            global MD_CFG_LOCKED
            MD_CFG_LOCKED = self.originals.pop('locked')

        if 'shared' in self.originals:
            global MD_CFG_SHARED
            MD_CFG_SHARED = self.originals.pop('shared')

        if 'freelist' in self.originals:
            global MD_CFG_FREELIST
            MD_CFG_FREELIST = self.originals.pop('freelist')

    def __repr__(self):
        return f'{self.__class__.__name__}({self.overrides!r})'

    def __enter__(self):
        self.c_activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.c_deactivate()
        return exc_type is None

    def __or__(self, EnvConfigContext other):
        if not isinstance(other, EnvConfigContext):
            return NotImplemented
        merged_overrides = self.overrides | other.overrides
        return EnvConfigContext(**merged_overrides)

    def __invert__(self):
        return EnvConfigContext.__new__(
            EnvConfigContext,
            **{k: not v for k, v in self.overrides.items()}
        )

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            self.c_activate()
            ret = func(*args, **kwargs)
            self.c_deactivate()
            return ret
        return wrapper


cdef EnvConfigContext MD_SHARED = EnvConfigContext(shared=True)
cdef EnvConfigContext MD_LOCKED = EnvConfigContext(locked=True)
cdef EnvConfigContext MD_FREELIST = EnvConfigContext(freelist=True)


globals()['MD_SHARED'] = MD_SHARED
globals()['MD_LOCKED'] = MD_LOCKED
globals()['MD_FREELIST'] = MD_FREELIST


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


cdef class MarketData:

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            c_recycle_buffer(self.header)

    def __reduce__(self):
        return MarketData.from_bytes, (self.to_bytes(),), self.__dict__

    def __setstate__(self, state):
        if state:
            self.__dict__.update(state)

    def __copy__(self):
        cdef object cls = self.__class__
        cdef MarketData instance = <MarketData> cls.__new__(cls)
        cdef data_type_t dtype = self.header.meta_info.dtype
        cdef market_data_t* header = c_md_new(dtype, NULL, NULL, 0)
        cdef size_t size = c_md_get_size(self.header.meta_info.dtype)
        memcpy(<void*> instance.header, <const char*> self.header, size)
        instance.__dict__.update(self.__dict__)
        return instance

    @staticmethod
    cdef inline object c_from_header(market_data_t* market_data, bint owner=False):
        cdef data_type_t dtype = market_data.meta_info.dtype
        cdef MarketData instance

        if dtype == data_type_t.DTYPE_INTERNAL:
            instance = <MarketData> InternalData.__new__(InternalData)
            instance.header = market_data
            instance.owner = owner
        # elif dtype == data_type_t.DTYPE_TRANSACTION:
        #     instance = <MarketData> TransactionData.__new__(TransactionData)
        # elif dtype == data_type_t.DTYPE_ORDER:
        #     instance = <MarketData> OrderData.__new__(OrderData)
        # elif dtype == data_type_t.DTYPE_TICK_LITE:
        #     instance = <MarketData> TickDataLite.__new__(TickDataLite)
        # elif dtype == data_type_t.DTYPE_TICK:
        #     instance = <MarketData> TickData.__new__(TickData)
        # elif dtype == data_type_t.DTYPE_BAR:
        #     bar_data = BarData.__new__(BarData)
        # elif dtype == data_type_t.DTYPE_REPORT:
        #     trade_report = TradeReport.__new__(TradeReport)
        # elif dtype == data_type_t.DTYPE_INSTRUCTION:
        #     trade_order = TradeInstruction.__new__(TradeInstruction)
        else:
            raise ValueError(f'Unknown data type {dtype}')

        # if dtype == data_type_t.DTYPE_TICK:
        #     (<TickData> instance).c_init_order_book()
        return instance

    cdef inline size_t c_get_size(self):
        cdef data_type_t dtype = self.header.meta_info.dtype
        cdef size_t size = c_md_get_size(dtype)
        if not size:
            raise ValueError(f'Unknown data type {dtype}')
        return size

    cdef inline str c_dtype_name(self):
        cdef data_type_t dtype = self.header.meta_info.dtype
        cdef const char* dtype_name = c_md_dtype_name(dtype)
        if not dtype_name:
            raise ValueError(f'Unknown data type {dtype}')
        return PyUnicode_FromString(dtype_name)

    cdef inline void c_to_bytes(self, char* out):
        c_md_serialize(self.header, out)

    @staticmethod
    cdef inline market_data_t* c_from_bytes(bytes data):
        return c_deserialize_buffer(<const char*> data)

    def to_bytes(self) -> bytes:
        cdef size_t size = c_md_serialized_size(self.header)
        cdef bytes data = PyBytes_FromStringAndSize(NULL, size)
        c_md_serialize(self.header, <char*> data)
        return data

    @classmethod
    def from_bytes(cls, bytes data):
        cdef market_data_t* header = MarketData.c_from_bytes(data)
        return MarketData.c_from_header(header, True)

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
            return datetime.fromtimestamp(self.header.meta_info.timestamp, tz=C_PROFILE.time_zone)

    property market_price:
        def __get__(self):
            return c_md_get_price(self.header)

    property price:
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

    def __cinit__(self, filter_mode_t value):
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
# from .c_tick cimport TickData, TickDataLite
# from .c_transaction cimport TransactionData, OrderData, TransactionHelper
# from .c_candlestick cimport BarData
# from .c_trade_utils cimport TradeReport, TradeInstruction

C_CONFIG = namedtuple('CONFIG', ['TICKER_SIZE', 'BOOK_SIZE', 'ID_SIZE', 'MAX_WORKERS'])(
    DEBUG=DEBUG,
    TICKER_SIZE=TICKER_SIZE,
    BOOK_SIZE=BOOK_SIZE,
    ID_SIZE=ID_SIZE,
    LONG_ID_SIZE=LONG_ID_SIZE,
    MAX_WORKERS=MAX_WORKERS
)
