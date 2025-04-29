# cython: language_level=3
import enum

from cpython.buffer cimport PyBuffer_FillInfo
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.datetime cimport datetime
from cpython.mem cimport PyMem_Free
from libc.stdint cimport uint8_t
from libc.string cimport memcpy, memset

from ..profile import PROFILE


# Base MarketData class
cdef class MarketData:
    """
    Base class for all market data types.
    """
    _dtype = DataType.DTYPE_MARKET_DATA

    def __cinit__(self):
        """
        Initialize the class but don't allocate memory.
        Child classes will allocate the appropriate memory.
        """
        self._data = NULL
        self._owner = False

    def __dealloc__(self):
        """
        Free allocated memory if this instance owns it.
        """
        if self._data is not NULL and self._owner:
            PyMem_Free(self._data)
            self._data = NULL

    def __init__(self, str ticker, double timestamp, **kwargs):
        """
        Initialize the market data with values.
        This method should only be called after memory has been allocated by a child class.
        """
        if self._data == NULL:
            raise ValueError("Memory not allocated. This class should not be instantiated directly.")

        cdef bytes ticker_bytes = ticker.encode('utf-8')
        cdef int ticker_len = min(len(ticker_bytes), TICKER_SIZE - 1)

        memset(&self._data.MetaInfo.ticker, 0, TICKER_SIZE)
        memcpy(&self._data.MetaInfo.ticker, <char*>ticker_bytes, ticker_len)

        self._data.MetaInfo.timestamp = timestamp
        self._data.MetaInfo.dtype = self._dtype

        if kwargs:
            self._additional = kwargs.copy()

    def __reduce__(self):
        """Support for pickle serialization"""
        return self.__class__.from_bytes, (self.to_bytes(),), self._additional

    def __setstate__(self, state):
        """Restore state from pickle"""
        if state:
            self._additional = state.copy()

    def __copy__(self):
        return self.__class__.from_bytes(self.to_bytes())

    def __setattr__(self, key, value):
        if hasattr(self, key):
            raise AttributeError(f'{self.__class__.__name__} is readonly.')

        self._set_additional(name=key, value=value)

    def __getattr__(self, key):
        if self._additional is None:
            raise AttributeError(f'Can not find attribute {key}.')

        if key in self._additional:
            return self._additional[key]

        raise AttributeError(f'Can not find attribute {key}.')

    cdef void _set_additional(self, str name, object value):
        if self._additional is None:
            self._additional = {name: value}
        else:
            self._additional[name] = value

    @staticmethod
    cdef size_t get_size(uint8_t dtype):
        """
        Get the size of an entry based on its dtype.
        """
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
            raise ValueError(f'Unknown data type {dtype}.')

    @classmethod
    def buffer_size(cls) -> int:
        return MarketData.get_size(cls._dtype)

    @staticmethod
    cdef size_t min_size():
        return min(sizeof(_OrderDataBuffer), sizeof(_TickDataLiteBuffer), sizeof(_CandlestickBuffer))

    @staticmethod
    cdef size_t max_size():
        return sizeof(_MarketDataBuffer)

    @classmethod
    def from_buffer(cls, const unsigned char[:] buffer):
        ...

    @classmethod
    def from_bytes(cls, bytes data):
        ...

    cpdef bytes to_bytes(self):
        """
        Convert the market data to bytes.
        Uses the meta info to determine the data type and size.
        """
        if self._data == NULL:
            raise ValueError("Cannot convert uninitialized data to bytes")

        cdef uint8_t dtype = self._data.MetaInfo.dtype
        cdef size_t size = MarketData.get_size(dtype)
        return PyBytes_FromStringAndSize(<char*>self._data, size)

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        """
        Implement the buffer protocol for read-only access.
        """
        if self._data == NULL:
            raise ValueError("Cannot get buffer from uninitialized data")

        cdef uint8_t dtype = self._data.MetaInfo.dtype
        cdef size_t size = MarketData.get_size(dtype)
        PyBuffer_FillInfo(buffer, self, <void*>self._data, size, 1, flags)

    def __releasebuffer__(self, Py_buffer* buffer):
        """
        Release the buffer.
        """
        pass

    @property
    def ticker(self) -> str:
        """
        Get the ticker symbol.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.MetaInfo.ticker.decode('utf-8').rstrip('\0')

    @property
    def timestamp(self) -> float:
        """
        Get the timestamp.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.MetaInfo.timestamp

    @property
    def dtype(self) -> int:
        """
        Get the data type.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.MetaInfo.dtype

    @property
    def topic(self) -> str:
        if self._data == NULL:
            raise ValueError("Data not initialized")
        ticker_str = self._data.MetaInfo.ticker.decode('utf-8').rstrip('\0')
        return f'{ticker_str}.{self.__class__.__name__}'

    @property
    def market_time(self) -> datetime:
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return datetime.fromtimestamp(self._data.MetaInfo.timestamp, tz=PROFILE.time_zone)
