# cython: language_level=3
from typing import Literal

from cpython.buffer cimport PyBuffer_FillInfo
from cpython.datetime cimport datetime, date, timedelta
from cpython.mem cimport PyMem_Malloc
from libc.math cimport NAN
from libc.string cimport memcpy, memset
from libc.stdint cimport uint32_t

from .market_data cimport MarketData, _MarketDataBuffer, _CandlestickBuffer, DataType
from ..profile import PROFILE

cdef class BarData(MarketData):
    """
    Represents a single bar of market data for a specific ticker within a given time frame.
    """
    _dtype = DataType.DTYPE_BAR

    def __cinit__(self):
        """
        Allocate memory for the bar data structure but don't initialize it.
        """
        self._owner = True

    def __init__(self, str ticker, double timestamp, double high_price, double low_price, double open_price, double close_price, double volume=0.0, double notional=0.0, uint32_t trade_count=0, double start_timestamp=0., object bar_span=None, **kwargs):
        """
        Initialize the bar data with values.
        """
        if not self._owner:
            raise MemoryError(f"Can not initialize a view of {self.__class__.__name__}.")

        self._data = <_MarketDataBuffer*>PyMem_Malloc(sizeof(_CandlestickBuffer))
        memset(self._data, 0, sizeof(_CandlestickBuffer))

        # Initialize base class fields
        MarketData.__init__(self, ticker=ticker, timestamp=timestamp, **kwargs)

        # Set data type for BarData
        # self._data.MetaInfo.dtype = DataType.DTYPE_BAR

        if bar_span is None:
            if not start_timestamp:
                raise ValueError('Must assign either start_timestamp or bar_span or both.')
            else:
                bar_span = timestamp - start_timestamp
        else:
            if isinstance(bar_span, timedelta):
                bar_span = bar_span.total_seconds()
            else:
                bar_span = float(bar_span)

        # Initialize bar-specific fields
        self._data.BarData.high_price = high_price
        self._data.BarData.low_price = low_price
        self._data.BarData.open_price = open_price
        self._data.BarData.close_price = close_price
        self._data.BarData.volume = volume
        self._data.BarData.notional = notional
        self._data.BarData.trade_count = trade_count
        self._data.BarData.bar_span = bar_span

    def __repr__(self) -> str:
        """
        String representation of the bar data.
        """
        if self._data == NULL:
            return "<BarData>(uninitialized)"
        return f"<BarData>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker}, open={self.open_price}, high={self.high_price}, low={self.low_price}, close={self.close_price}, volume={self.volume})"

    def __setitem__(self, key: str, value: float | int):
        if key == 'high_price':
            self._data.BarData.high_price = value
        elif key == 'low_price':
            self._data.BarData.low_price = value
        elif key == 'open_price':
            self._data.BarData.open_price = value
        elif key == 'close_price':
            self._data.BarData.close_price = value
        elif key == 'volume':
            self._data.BarData.volume = value
        elif key == 'notional':
            self._data.BarData.notional = value
        elif key == 'trade_count':
            self._data.BarData.trade_count = value
        else:
            raise KeyError(f'Can not set {key} to the value {value}.')

    def __getitem__(self, key: str) -> float | int:
        if key == 'high_price':
            return self._data.BarData.high_price
        elif key == 'low_price':
            return self._data.BarData.low_price
        elif key == 'open_price':
            return self._data.BarData.open_price
        elif key == 'close_price':
            return self._data.BarData.close_price
        elif key == 'volume':
            return self._data.BarData.volume
        elif key == 'notional':
            return self._data.BarData.notional
        elif key == 'trade_count':
            return self._data.BarData.trade_count
        else:
            raise KeyError(f'Can not get {key} value.')

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
    def bar_span_seconds(self) -> float:
        """
        Get the duration of the bar.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.BarData.bar_span

    @property
    def bar_span(self) -> timedelta:
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return timedelta(seconds=self._data.BarData.bar_span)

    @property
    def vwap(self) -> float:
        """
        Get the volume-weighted average price for the bar.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        if self._data.BarData.volume <= 0:
            return NAN
        return self._data.BarData.notional / self._data.BarData.volume

    @property
    def bar_type(self) -> Literal['Hourly-Plus', 'Hourly', 'Minute-Plus', 'Minute', 'Sub-Minute']:
        """
        Determines the type of the bar based on its span.

        Returns:
            Literal['Hourly-Plus', 'Hourly', 'Minute-Plus', 'Minute', 'Sub-Minute']: The type of the bar.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        bar_span = self._data.BarData.bar_span

        if bar_span > 3600:
            return 'Hourly-Plus'
        elif bar_span == 3600:
            return 'Hourly'
        elif bar_span > 60:
            return 'Minute-Plus'
        elif bar_span == 60:
            return 'Minute'
        else:
            return 'Sub-Minute'

    @property
    def bar_end_time(self) -> datetime:
        """
        The end time of the bar.

        Returns:
            datetime | date: The end time of the bar.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return datetime.fromtimestamp(self.timestamp, tz=PROFILE.time_zone)

    @property
    def bar_start_time(self) -> datetime:
        """
        The start time of the bar.

        Returns:
            datetime: The start time of the bar.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return datetime.fromtimestamp(self.start_timestamp, tz=PROFILE.time_zone)

    @property
    def market_price(self) -> float:
        return self.close_price


class DailyBar(BarData):
    def __init__(self, str ticker, date market_date, double high_price, double low_price, double open_price, double close_price, double volume=0.0, double notional=0.0, uint32_t trade_count=0, int bar_span=1, **kwargs):
        timestamp = 10000 * market_date.year + 100 * market_date.month + market_date.day

        super().__init__(ticker=ticker, timestamp=timestamp, high_price=high_price, low_price=low_price, open_price=open_price, close_price=close_price, volume=volume, notional=notional, trade_count=trade_count, bar_span=bar_span, **kwargs)

    def __repr__(self) -> str:
        if (bar_span := super().bar_span_seconds) == 1:
            return f"<DailyBar>([{self.market_date}] {self.ticker}, open={self.open_price}, high={self.high_price}, low={self.low_price}, close={self.close_price}, volume={self.volume})"
        else:
            return f"<DailyBar>([{self.market_date}] {self.ticker}, span={bar_span}d, open={self.open_price}, high={self.high_price}, low={self.low_price}, close={self.close_price}, volume={self.volume})"

    @property
    def market_date(self) -> date:
        """
        The market date of the bar.

        Returns:
            date: The market date of the bar.
        """

        int_date = int(self.timestamp)
        y, _m = divmod(int_date, 10000)
        m, d = divmod(_m, 100)

        return date(year=y, month=m, day=d)

    @property
    def market_time(self) -> date:
        """
        The market date of the bar (same as `market_date`).

        Returns:
            date: The market date of the bar.
        """
        return self.market_date

    @property
    def bar_start_time(self) -> date:
        """
        The start date of the bar period.

        Returns:
            date: The start date of the bar.
        """
        return self.market_date - self.bar_span

    @property
    def bar_end_time(self) -> date:
        """
        The end date of the bar period.

        Returns:
            date: The end date of the bar.
        """
        return self.market_date

    @property
    def bar_span(self) -> timedelta:
        return timedelta(days=super().bar_span_seconds)

    @property
    def bar_type(self) -> Literal['Daily', 'Daily-Plus']:
        """
        Determines the type of the bar based on its span.

        Returns:
            Literal['Daily', 'Daily-Plus']: The type of the bar.

        Raises:
            ValueError: If `bar_span` is not valid for a daily bar.
        """
        if (bar_span := super().bar_span_seconds) == 1:
            return 'Daily'
        elif bar_span > 1:
            return 'Daily-Plus'
        else:
            raise ValueError(f'Invalid bar_span for {self.__class__.__name__}! Expect an int greater or equal to 1, got {super().bar_span}.')
