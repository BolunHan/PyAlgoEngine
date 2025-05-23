# cython: language_level=3
from typing import Literal

cimport cython
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.datetime cimport datetime, date, timedelta
from libc.math cimport NAN
from libc.string cimport memcpy
from libc.stdint cimport uint32_t

from .c_market_data cimport _MarketDataVirtualBase, TICKER_SIZE, _MarketDataBuffer, _CandlestickBuffer, DataType


@cython.freelist(128)
cdef class BarData:
    def __cinit__(self):
        self._data_ptr = <_MarketDataBuffer*> &self._data
        self._data_addr = <uintptr_t> self._data_ptr

    def __init__(self, *, str ticker, double timestamp, double high_price, double low_price, double open_price, double close_price, double volume=0.0, double notional=0.0, uint32_t trade_count=0, double start_timestamp=0., object bar_span=None, **kwargs):
        # Initialize base class fields
        cdef bytes ticker_bytes = ticker.encode('utf-8')
        cdef size_t ticker_len = min(len(ticker_bytes), TICKER_SIZE - 1)
        memcpy(<void*> &self._data.ticker, <const char*> ticker_bytes, ticker_len)
        self._data.timestamp = timestamp
        self._data.dtype = DataType.DTYPE_BAR
        if kwargs: self.__dict__.update(kwargs)
        
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
        self._data.high_price = high_price
        self._data.low_price = low_price
        self._data.open_price = open_price
        self._data.close_price = close_price
        self._data.volume = volume
        self._data.notional = notional
        self._data.trade_count = trade_count
        self._data.bar_span = bar_span

    def __repr__(self) -> str:
        return f"<BarData>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker}, open={self.open_price}, high={self.high_price}, low={self.low_price}, close={self.close_price}, volume={self.volume})"

    def __reduce__(self):
        return self.__class__.from_bytes, (self.to_bytes(),), self.__dict__

    def __setstate__(self, state):
        if state:
            self.__dict__.update(state)

    def __copy__(self):
        cdef BarData instance = BarData.__new__(BarData)
        memcpy(<void*> &instance._data, <const char*> &self._data, sizeof(_CandlestickBuffer))
        return instance

    def __setitem__(self, key: str, value: float | int):
        if key == 'high_price':
            self._data.high_price = value
        elif key == 'low_price':
            self._data.low_price = value
        elif key == 'open_price':
            self._data.open_price = value
        elif key == 'close_price':
            self._data.close_price = value
        elif key == 'volume':
            self._data.volume = value
        elif key == 'notional':
            self._data.notional = value
        elif key == 'trade_count':
            self._data.trade_count = value
        else:
            raise KeyError(f'Can not set {key} to the value {value}.')

    def __getitem__(self, key: str) -> float | int:
        if key == 'high_price':
            return self._data.high_price
        elif key == 'low_price':
            return self._data.low_price
        elif key == 'open_price':
            return self._data.open_price
        elif key == 'close_price':
            return self._data.close_price
        elif key == 'volume':
            return self._data.volume
        elif key == 'notional':
            return self._data.notional
        elif key == 'trade_count':
            return self._data.trade_count
        else:
            raise KeyError(f'Can not get {key} value.')

    @classmethod
    def buffer_size(cls):
        return sizeof(_CandlestickBuffer)

    cdef bytes c_to_bytes(self):
        return PyBytes_FromStringAndSize(<char*> &self._data, sizeof(self._data))

    @staticmethod
    cdef BarData c_from_bytes(bytes data):
        cdef BarData instance = BarData.__new__(BarData)
        memcpy(<void*> &instance._data, <const char*> data, sizeof(_CandlestickBuffer))
        return instance

    def to_bytes(self) -> bytes:
        return self.c_to_bytes()

    @classmethod
    def from_bytes(cls, bytes data):
        return BarData.c_from_bytes(data)

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
    def high_price(self) -> float:
        return self._data.high_price

    @property
    def low_price(self) -> float:
        return self._data.low_price

    @property
    def open_price(self) -> float:
        return self._data.open_price

    @property
    def close_price(self) -> float:
        return self._data.close_price

    @property
    def volume(self) -> float:
        return self._data.volume

    @property
    def notional(self) -> float:
        return self._data.notional

    @property
    def trade_count(self) -> int:
        return self._data.trade_count

    @property
    def start_timestamp(self) -> float:
        return self._data.timestamp - self._data.bar_span

    @property
    def bar_span_seconds(self) -> float:
        return self._data.bar_span

    @property
    def bar_span(self) -> timedelta:
        return timedelta(seconds=self._data.bar_span)

    @property
    def vwap(self) -> float:
        if self._data.volume <= 0:
            return NAN
        return self._data.notional / self._data.volume

    @property
    def bar_type(self) -> Literal['Hourly-Plus', 'Hourly', 'Minute-Plus', 'Minute', 'Sub-Minute']:
        bar_span = self._data.bar_span

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
        return _MarketDataVirtualBase.c_to_dt(self.timestamp)

    @property
    def bar_start_time(self) -> datetime:
        return _MarketDataVirtualBase.c_to_dt(self.start_timestamp)

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
