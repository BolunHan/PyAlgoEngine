# cython: language_level=3
from market_data cimport MarketData, _MarketDataBuffer, _CandlestickBuffer, DataType

from libc.string cimport memcpy, memset
from libc.stdint cimport uint32_t
from cpython.buffer cimport PyBuffer_FillInfo
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.mem cimport PyMem_Malloc

cdef class BarData(MarketData):
    """
    Represents a single bar of market data for a specific ticker within a given time frame.
    """
    def __cinit__(self):
        """
        Allocate memory for the bar data structure but don't initialize it.
        """
        self._dtype = DataType.DTYPE_BAR
        self._owner = True

    def __init__(self, str ticker, double timestamp, double high_price, double low_price, double open_price, double close_price, double volume=0.0, double notional=0.0, uint32_t trade_count=0, double bar_span=0.0):
        """
        Initialize the bar data with values.
        """
        if not self._owner:
            raise MemoryError(f"Can not initialize a view of {self.__class__.__name__}.")

        self._data = <_MarketDataBuffer*>PyMem_Malloc(sizeof(_CandlestickBuffer))
        memset(self._data, 0, sizeof(_CandlestickBuffer))

        # Initialize base class fields
        MarketData.__init__(self, ticker, timestamp)

        # Set data type for BarData
        self._data.MetaInfo.dtype = DataType.DTYPE_BAR

        # Initialize bar-specific fields
        self._data.BarData.high_price = high_price
        self._data.BarData.low_price = low_price
        self._data.BarData.open_price = open_price
        self._data.BarData.close_price = close_price
        self._data.BarData.volume = volume
        self._data.BarData.notional = notional
        self._data.BarData.trade_count = trade_count
        self._data.BarData.bar_span = bar_span

    @classmethod
    def from_buffer(cls, const unsigned char[:] buffer):
        """
        Create a new instance from a buffer.
        """
        cdef BarData instance = cls.__new__(cls)

        # Point to the buffer data
        instance._data = <_MarketDataBuffer*>&buffer[0]
        instance._owner = False

        return instance

    @classmethod
    def from_bytes(cls, bytes data):
        """
        Create a new instance from bytes.
        Creates a copy of the data, so the instance owns the memory.
        """
        cdef BarData instance = cls.__new__(cls)
        cdef const char* data_ptr = <const char*>data

        instance._owner = True
        instance._data = <_MarketDataBuffer*>PyMem_Malloc(sizeof(_CandlestickBuffer))

        if instance._data == NULL:
            raise MemoryError("Failed to allocate memory for BarData")

        memcpy(instance._data, data_ptr, sizeof(_CandlestickBuffer))

        return instance

    def to_bytes(self):
        """
        Convert the bar data to bytes.
        """
        if self._data == NULL:
            raise ValueError("Cannot convert uninitialized data to bytes")

        return PyBytes_FromStringAndSize(<char*>self._data, sizeof(_CandlestickBuffer))

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        """
        Implement the buffer protocol for read-only access.
        """
        if self._data == NULL:
            raise ValueError("Cannot get buffer from uninitialized data")
        PyBuffer_FillInfo(buffer, self, <_CandlestickBuffer*>self._data, sizeof(_CandlestickBuffer), 1, flags)

    def __copy__(self):
        """
        Create a copy of this instance.
        """
        if self._data == NULL:
            raise ValueError("Cannot copy uninitialized data")

        cdef BarData new_bar = BarData.__new__(BarData)
        # Allocate memory specifically for a BarData buffer
        new_bar._data = <_MarketDataBuffer*>PyMem_Malloc(sizeof(_CandlestickBuffer))
        if new_bar._data == NULL:
            raise MemoryError("Failed to allocate memory for copy")

        new_bar._owner = True
        memcpy(new_bar._data, self._data, sizeof(_CandlestickBuffer))

        return new_bar

    @property
    def high_price(self) -> float:
        """
        Get the highest price during the bar.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.BarData.high_price

    @property
    def low_price(self) -> float:
        """
        Get the lowest price during the bar.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.BarData.low_price

    @property
    def open_price(self) -> float:
        """
        Get the opening price of the bar.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.BarData.open_price

    @property
    def close_price(self) -> float:
        """
        Get the closing price of the bar.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.BarData.close_price

    @property
    def volume(self) -> float:
        """
        Get the total volume of trades during the bar.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.BarData.volume

    @property
    def notional(self) -> float:
        """
        Get the total notional value of trades during the bar.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.BarData.notional

    @property
    def trade_count(self) -> int:
        """
        Get the number of trades that occurred during the bar.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.BarData.trade_count

    @property
    def start_timestamp(self) -> float:
        """
        Get the start time of the bar.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.BarData.timestamp - self._data.BarData.bar_span

    @property
    def bar_span(self) -> float:
        """
        Get the duration of the bar.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.BarData.bar_span

    @property
    def vwap(self) -> float:
        """
        Get the volume-weighted average price for the bar.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        if self._data.BarData.volume <= 0:
            return 0.0
        return self._data.BarData.notional / self._data.BarData.volume

    def __repr__(self) -> str:
        """
        String representation of the bar data.
        """
        if self._data == NULL:
            return "BarData(uninitialized)"
        return f"BarData(ticker='{self.ticker}', timestamp={self.timestamp}, open={self.open_price}, high={self.high_price}, low={self.low_price}, close={self.close_price}, volume={self.volume})"
