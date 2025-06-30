# cython: language_level=3
cimport cython
import abc

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.string cimport memcpy

from algo_engine.profile.c_base cimport C_PROFILE


cdef class _MarketDataVirtualBase:
    @staticmethod
    cdef size_t c_get_size(uint8_t dtype):
        if dtype == DataType.DTYPE_INTERNAL:
            return sizeof(_InternalBuffer)
        elif dtype == DataType.DTYPE_TRANSACTION:
            return sizeof(_TransactionDataBuffer)
        elif dtype == DataType.DTYPE_ORDER:
            return sizeof(_OrderDataBuffer)
        elif dtype == DataType.DTYPE_TICK_LITE:
            return sizeof(_TickDataLiteBuffer)
        elif dtype == DataType.DTYPE_TICK:
            return sizeof(_TickDataBuffer)
        elif dtype == DataType.DTYPE_BAR:
            return sizeof(_CandlestickBuffer)
        elif dtype == DataType.DTYPE_REPORT:
            return sizeof(_TradeReportBuffer)
        elif dtype == DataType.DTYPE_TRANSACTION:
            return sizeof(_TradeInstructionBuffer)
        elif dtype == DataType.DTYPE_MARKET_DATA or dtype == DataType.DTYPE_UNKNOWN:
            return sizeof(_MarketDataBuffer)
        else:
            raise ValueError(f'Unknown data type {dtype}')

    @staticmethod
    cdef str c_dtype_name(uint8_t dtype):
        if dtype == DataType.DTYPE_INTERNAL:
            return 'InternalData'
        elif dtype == DataType.DTYPE_TRANSACTION:
            return 'TransactionData'
        elif dtype == DataType.DTYPE_ORDER:
            return 'OrderData'
        elif dtype == DataType.DTYPE_TICK_LITE:
            return 'TickDataLite'
        elif dtype == DataType.DTYPE_TICK:
            return 'TickData'
        elif dtype == DataType.DTYPE_BAR:
            return 'BarData'
        elif dtype == DataType.DTYPE_REPORT:
            return 'TradeReport'
        elif dtype == DataType.DTYPE_TRANSACTION:
            return 'TradeInstruction'
        elif dtype == DataType.DTYPE_MARKET_DATA or dtype == DataType.DTYPE_UNKNOWN:
            return 'GenericMarketData'
        else:
            raise ValueError(f'Unknown data type {dtype}')

    @staticmethod
    cdef object c_ptr_to_data(_MarketDataBuffer* data_ptr):
        cdef _MetaInfo* meta_info = <_MetaInfo*> data_ptr
        cdef uint8_t dtype = meta_info.dtype
        cdef size_t length = _MarketDataVirtualBase.c_get_size(dtype)

        cdef InternalData internal_data
        cdef TransactionData transaction_data
        cdef OrderData order_data
        cdef TickDataLite tick_data_lite
        cdef TickData tick_data
        cdef BarData bar_data
        cdef TradeReport trade_report
        cdef TradeInstruction trade_order

        if dtype == DataType.DTYPE_INTERNAL:
            internal_data = InternalData.__new__(InternalData)
            memcpy(<char*> internal_data._data_ptr, <const char*> data_ptr, length)
            return internal_data
        elif dtype == DataType.DTYPE_TRANSACTION:
            transaction_data = TransactionData.__new__(TransactionData)
            memcpy(<char*> transaction_data._data_ptr, <const char*> data_ptr, length)
            return transaction_data
        elif dtype == DataType.DTYPE_ORDER:
            order_data = OrderData.__new__(OrderData)
            memcpy(<char*> order_data._data_ptr, <const char*> data_ptr, length)
            return order_data
        elif dtype == DataType.DTYPE_TICK_LITE:
            tick_data_lite = TickDataLite.__new__(TickDataLite)
            memcpy(<char*> tick_data_lite._data_ptr, <const char*> data_ptr, length)
            return tick_data_lite
        elif dtype == DataType.DTYPE_TICK:
            tick_data = TickData.__new__(TickData)
            memcpy(<char*> tick_data._data_ptr, <const char*> data_ptr, length)
            tick_data._init_order_book()
            return tick_data
        elif dtype == DataType.DTYPE_BAR:
            bar_data = BarData.__new__(BarData)
            memcpy(<char*> bar_data._data_ptr, <const char*> data_ptr, length)
            return bar_data
        elif dtype == DataType.DTYPE_REPORT:
            trade_report = TradeReport.__new__(TradeReport)
            memcpy(<char*> trade_report._data_ptr, <const char*> data_ptr, length)
            return trade_report
        elif dtype == DataType.DTYPE_INSTRUCTION:
            trade_order = TradeInstruction.__new__(TradeInstruction)
            memcpy(<char*> trade_order._data_ptr, <const char*> data_ptr, length)
            return trade_order
        else:
            raise ValueError(f'Unknown data type {dtype}')

    @staticmethod
    cdef bytes c_ptr_to_bytes(_MarketDataBuffer* data_ptr):
        cdef _MetaInfo* meta_info = <_MetaInfo*> data_ptr
        cdef uint8_t dtype = meta_info.dtype
        cdef size_t length = _MarketDataVirtualBase.c_get_size(dtype)
        cdef bytes data = PyBytes_FromStringAndSize(<const char*> data_ptr, length)
        return data

    @staticmethod
    cdef size_t c_max_size():
        return max(sizeof(_TransactionDataBuffer), sizeof(_TickDataBuffer), sizeof(_CandlestickBuffer))

    @staticmethod
    cdef size_t c_min_size():
        return min(sizeof(_OrderDataBuffer), sizeof(_TickDataLiteBuffer), sizeof(_CandlestickBuffer))

    @staticmethod
    cdef datetime c_to_dt(double timestamp):
        return datetime.fromtimestamp(timestamp, tz=C_PROFILE.time_zone)


class MarketData(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __reduce__(self):
        ...

    @abc.abstractmethod
    def __setstate__(self, state):
        ...

    @abc.abstractmethod
    def __copy__(self):
        ...

    @classmethod
    def buffer_size(cls) -> int:
        return _MarketDataVirtualBase.c_max_size()

    @classmethod
    @abc.abstractmethod
    def from_bytes(cls, data: bytes):
        ...

    @abc.abstractmethod
    def to_bytes(self) -> bytes:
        ...

    @property
    @abc.abstractmethod
    def ticker(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def timestamp(self) -> float:
        ...

    @property
    @abc.abstractmethod
    def dtype(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def topic(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def market_time(self) -> datetime:
        ...

    @property
    @abc.abstractmethod
    def market_price(self) -> float:
        ...


@cython.freelist(128)
cdef class InternalData:
    def __cinit__(self):
        self._data_ptr = <_MarketDataBuffer*> &self._data
        self._data_addr = <uintptr_t> self._data_ptr

    def __init__(self, *, ticker: str, double timestamp, uint32_t code, **kwargs):
        # Initialize base class fields
        cdef bytes ticker_bytes = ticker.encode('utf-8')
        cdef size_t ticker_len = min(len(ticker_bytes), TICKER_SIZE - 1)
        memcpy(<void*> &self._data.ticker, <const char*> ticker_bytes, ticker_len)
        self._data.timestamp = timestamp
        self._data.dtype = DataType.DTYPE_INTERNAL
        if kwargs: self.__dict__.update(kwargs)

        # Initialize internal-specific fields
        self._data.code = code

    def __repr__(self) -> str:
        return f"<InternalData>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker}, code={self.code})"

    def __reduce__(self):
        return self.__class__.from_bytes, (self.to_bytes(),), self.__dict__

    def __setstate__(self, state):
        if state:
            self.__dict__.update(state)

    def __copy__(self):
        cdef TransactionData instance = TransactionData.__new__(TransactionData)
        memcpy(<void*> &instance._data, <const char*> &self._data, sizeof(_InternalBuffer))
        return instance

    @classmethod
    def buffer_size(cls):
        return sizeof(_InternalBuffer)

    cdef bytes c_to_bytes(self):
        return PyBytes_FromStringAndSize(<char*> &self._data, sizeof(self._data))

    @staticmethod
    cdef InternalData c_from_bytes(bytes data):
        cdef InternalData instance = InternalData.__new__(InternalData)
        memcpy(<void*> &instance._data, <const char*> data, sizeof(_InternalBuffer))
        return instance

    def to_bytes(self) -> bytes:
        return self.c_to_bytes()

    @classmethod
    def from_bytes(cls, bytes data):
        return InternalData.c_from_bytes(data)

    @property
    def ticker(self) -> str:
        return self._data.ticker.decode('utf-8')

    @property
    def timestamp(self) -> float:
        return self._data.timestamp

    @property
    def dtype(self) -> int:
        return self._data.dtype

    @property
    def topic(self) -> str:
        ticker_str = self._data.ticker.decode('utf-8')
        return f'{ticker_str}.{self.__class__.__name__}'

    @property
    def market_time(self) :
        return _MarketDataVirtualBase.c_to_dt(self._data.timestamp)

    @property
    def price(self) -> float:
        return 0.

    @property
    def code(self) -> int:
        return self._data.code


cdef class FilterMode:
    # Class-level constants for flags
    NO_INTERNAL = FilterMode.__new__(FilterMode, _FilterMode.NO_INTERNAL)
    NO_CANCEL = FilterMode.__new__(FilterMode, _FilterMode.NO_CANCEL)
    NO_AUCTION = FilterMode.__new__(FilterMode, _FilterMode.NO_AUCTION)
    NO_ORDER = FilterMode.__new__(FilterMode, _FilterMode.NO_ORDER)
    NO_TRADE = FilterMode.__new__(FilterMode, _FilterMode.NO_TRADE)
    NO_TICK = FilterMode.__new__(FilterMode, _FilterMode.NO_TICK)

    def __cinit__(self, uint32_t value=0):
        self.value = value

    @classmethod
    def all(cls):
        return FilterMode.__new__(
            FilterMode,
            _FilterMode.NO_INTERNAL |
            _FilterMode.NO_CANCEL |
            _FilterMode.NO_AUCTION |
            _FilterMode.NO_ORDER |
            _FilterMode.NO_TRADE |
            _FilterMode.NO_TICK
        )

    def __or__(self, FilterMode other):
        return FilterMode.__new__(FilterMode, self.value | other.value)

    def __and__(self, FilterMode other):
        return FilterMode.__new__(FilterMode, self.value & other.value)

    def __contains__(self, FilterMode other):
        return (self.value & other.value) == other.value

    def __repr__(self):
        flags = []
        if _FilterMode.NO_INTERNAL & self.value: flags.append("NO_INTERNAL")
        if _FilterMode.NO_CANCEL & self.value: flags.append("NO_CANCEL")
        if _FilterMode.NO_AUCTION & self.value: flags.append("NO_AUCTION")
        if _FilterMode.NO_ORDER & self.value: flags.append("NO_ORDER")
        if _FilterMode.NO_TRADE & self.value: flags.append("NO_TRADE")
        if _FilterMode.NO_TICK & self.value: flags.append("NO_TICK")
        return f"<FilterMode {self.value:#0x}: {' | '.join(flags) or 'None'}>"

    @staticmethod
    cdef inline bint c_mask_data(uintptr_t data_addr, uint32_t filter_mode):
        cdef _MarketDataBuffer* market_data_ptr = <_MarketDataBuffer*> data_addr
        cdef uint8_t dtype = market_data_ptr.MetaInfo.dtype
        cdef double timestamp = market_data_ptr.MetaInfo.timestamp
        cdef _TransactionDataBuffer* trade_data

        if _FilterMode.NO_INTERNAL & filter_mode:
            if dtype == DataType.DTYPE_INTERNAL:
                return False

        if _FilterMode.NO_CANCEL & filter_mode:
            if dtype == DataType.DTYPE_TRANSACTION:
                trade_data = <_TransactionDataBuffer*> data_addr
                side = trade_data.side
                if TransactionHelper.get_offset(side) == Offset.OFFSET_CANCEL:
                    return False

        if _FilterMode.NO_AUCTION & filter_mode:
            if not C_PROFILE.c_timestamp_in_market_session(t=timestamp):
                return False

        if _FilterMode.NO_ORDER & filter_mode:
            if dtype == DataType.DTYPE_ORDER:
                return False

        if _FilterMode.NO_TRADE & filter_mode:
            if dtype == DataType.DTYPE_TRANSACTION:
                return False

        if _FilterMode.NO_TICK & filter_mode:
            if dtype == DataType.DTYPE_TICK or dtype == DataType.DTYPE_TICK_LITE:
                return False

        return True

    def mask_data(self, market_data: object) -> bool:
        cdef uintptr_t data_addr = <uintptr_t> market_data._data_addr
        return FilterMode.c_mask_data(data_addr, self.value)


from .c_tick cimport TickData, TickDataLite
from .c_transaction cimport TransactionData, OrderData, TransactionHelper
from .c_candlestick cimport BarData
from .c_trade_utils cimport TradeReport, TradeInstruction

MarketData.register(InternalData)
MarketData.register(TickData)
MarketData.register(TickDataLite)
MarketData.register(TransactionData)
MarketData.register(OrderData)
MarketData.register(BarData)
