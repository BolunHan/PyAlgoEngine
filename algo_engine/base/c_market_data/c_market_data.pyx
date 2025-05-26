# cython: language_level=3
import abc
from cpython.datetime cimport datetime
from libc.stdint cimport uint8_t

from algo_engine.profile import PROFILE


cdef class _MarketDataVirtualBase:
    @staticmethod
    cdef size_t c_get_size(uint8_t dtype):
        if dtype == DataType.DTYPE_TRANSACTION:
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
    cdef size_t c_max_size():
        return max(sizeof(_TransactionDataBuffer), sizeof(_TickDataBuffer), sizeof(_CandlestickBuffer))

    @staticmethod
    cdef size_t c_min_size():
        return min(sizeof(_OrderDataBuffer), sizeof(_TickDataLiteBuffer), sizeof(_CandlestickBuffer))

    @staticmethod
    cdef datetime c_to_dt(double timestamp):
        return datetime.fromtimestamp(timestamp, tz=PROFILE.time_zone)


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


from .c_tick cimport TickData, TickDataLite
from .c_transaction cimport TransactionData, OrderData
from .c_candlestick cimport BarData

MarketData.register(TickData)
MarketData.register(TickDataLite)
MarketData.register(TransactionData)
MarketData.register(OrderData)
MarketData.register(BarData)