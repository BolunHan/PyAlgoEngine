# cython: language_level=3
from cpython.datetime cimport datetime
from libc.stdint cimport uint8_t
from libc.string cimport memset, memcpy

from algo_engine.profile import PROFILE


# Base MarketData class
cdef class MarketData:
    """
    Base class for all market data types.
    """

    def __init__(self, *, str ticker, double timestamp, uint8_t dtype, **kwargs):
        """
        Initialize the market data with values.
        This method should only be called after memory has been allocated by a child class.
        """
        if self._data_ptr is NULL:
            raise ValueError("Memory not allocated. This is a abstract class and should not be instantiated directly.")

        cdef bytes ticker_bytes = ticker.encode('utf-8')
        cdef int ticker_len = min(len(ticker_bytes), TICKER_SIZE - 1)

        memset(&self._data_ptr.MetaInfo.ticker, 0, TICKER_SIZE)
        memcpy(&self._data_ptr.MetaInfo.ticker, <const char*> ticker_bytes, ticker_len)

        self._data_ptr.MetaInfo.timestamp = timestamp
        self._data_ptr.MetaInfo.dtype = dtype

        if kwargs:
            self.__dict__.update(kwargs)

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

    # --- python interface ---

    def __reduce__(self):
        """Support for pickle serialization"""
        return self.__class__.from_bytes, (self.to_bytes(),), self.__dict__

    def __setstate__(self, state):
        """Restore state from pickle"""
        if state:
            self.__dict__.update(state)

    def __copy__(self):
        return self.__class__.from_bytes(self.to_bytes())

    @classmethod
    def buffer_size(cls) -> int:
        return sizeof(_MarketDataBuffer)

    @classmethod
    def from_bytes(cls, data: bytes):
        ...

    def to_bytes(self) -> bytes:
        ...

    @property
    def ticker(self) -> str:
        return self._data_ptr.MetaInfo.ticker.decode('utf-8')

    @property
    def timestamp(self) -> float:
        return self._data_ptr.MetaInfo.timestamp

    @property
    def dtype(self) -> int:
        return self._data_ptr.MetaInfo.dtype

    @property
    def topic(self) -> str:
        ticker_str = self._data_ptr.MetaInfo.ticker.decode('utf-8').rstrip('\0')
        return f'{ticker_str}.{self.__class__.__name__}'

    @property
    def market_time(self) -> datetime:
        return datetime.fromtimestamp(self._data_ptr.MetaInfo.timestamp, tz=PROFILE.time_zone)
