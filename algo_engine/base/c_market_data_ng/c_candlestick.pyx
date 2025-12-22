from cpython.datetime cimport datetime, date, timedelta
from cpython.unicode cimport PyUnicode_AsUTF8
from libc.math cimport NAN
from libc.stdint cimport uint64_t, uintptr_t

from .c_market_data cimport C_PROFILE, md_data_type, c_init_buffer


cdef class BarData(MarketData):
    def __init__(
            self,
            *,
            str ticker,
            double timestamp,
            double high_price,
            double low_price,
            double open_price,
            double close_price,
            double volume=0.0,
            double notional=0.0,
            uint64_t trade_count=0,
            double start_timestamp=0.,
            object bar_span=None,
            **kwargs
    ):
        self.header = c_init_buffer(
            md_data_type.DTYPE_BAR,
            PyUnicode_AsUTF8(ticker),
            timestamp
        )

        cdef double bar_span_seconds
        if bar_span is None:
            if not start_timestamp:
                raise ValueError('Must assign either start_timestamp or bar_span or both.')
            else:
                bar_span_seconds = timestamp - start_timestamp
        else:
            if isinstance(bar_span, timedelta):
                bar_span_seconds = bar_span.total_seconds()
            else:
                bar_span_seconds = <double> bar_span

        # Initialize bar-specific fields
        self.header.bar_data.high_price = high_price
        self.header.bar_data.low_price = low_price
        self.header.bar_data.open_price = open_price
        self.header.bar_data.close_price = close_price
        self.header.bar_data.volume = volume
        self.header.bar_data.notional = notional
        self.header.bar_data.trade_count = trade_count
        self.header.bar_data.bar_span = bar_span_seconds

        self.data_addr = <uintptr_t> self.header
        self.owner = True

        if kwargs:
            self.__dict__.update(kwargs)

    def __repr__(self):
        if not self.header:
            return f"<{self.__class__.__name__}>(Uninitialized)"
        return f"<{self.__class__.__name__}>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker}, open={self.open_price}, high={self.high_price}, low={self.low_price}, close={self.close_price}, volume={self.volume})"

    def __setitem__(self, str key, object value):
        if key == 'high_price':
            self.header.bar_data.high_price = value
        elif key == 'low_price':
            self.header.bar_data.low_price = value
        elif key == 'open_price':
            self.header.bar_data.open_price = value
        elif key == 'close_price':
            self.header.bar_data.close_price = value
        elif key == 'volume':
            self.header.bar_data.volume = value
        elif key == 'notional':
            self.header.bar_data.notional = value
        elif key == 'trade_count':
            self.header.bar_data.trade_count = value
        else:
            raise KeyError(f'Can not set {key} to the value {value}.')

    def __getitem__(self, str key):
        if key == 'high_price':
            return self.header.bar_data.high_price
        elif key == 'low_price':
            return self.header.bar_data.low_price
        elif key == 'open_price':
            return self.header.bar_data.open_price
        elif key == 'close_price':
            return self.header.bar_data.close_price
        elif key == 'volume':
            return self.header.bar_data.volume
        elif key == 'notional':
            return self.header.bar_data.notional
        elif key == 'trade_count':
            return self.header.bar_data.trade_count
        else:
            raise KeyError(f'Can not get {key} value.')

    property high_price:
        def __get__(self):
            return self.header.bar_data.high_price

    property low_price:
        def __get__(self):
            return self.header.bar_data.low_price

    property open_price:
        def __get__(self):
            return self.header.bar_data.open_price

    property close_price:
        def __get__(self):
            return self.header.bar_data.close_price

    property volume:
        def __get__(self):
            return self.header.bar_data.volume

    property notional:
        def __get__(self):
            return self.header.bar_data.notional

    property trade_count:
        def __get__(self):
            return self.header.bar_data.trade_count

    property start_timestamp:
        def __get__(self):
            return self.header.bar_data.meta_info.timestamp - self.header.bar_data.bar_span

    property bar_span_seconds:
        def __get__(self):
            return self.header.bar_data.bar_span

    property bar_span:
        def __get__(self):
            return timedelta(seconds=self.header.bar_data.bar_span)

    property vwap:
        def __get__(self):
            if self.header.bar_data.volume <= 0:
                return NAN
            return self.header.bar_data.notional / self.header.bar_data.volume

    property bar_type:
        def __get__(self):
            cdef double bar_span = self.header.bar_data.bar_span
            if bar_span > 3600.0:
                return 'Hourly-Plus'
            elif bar_span == 3600.0:
                return 'Hourly'
            elif bar_span > 60.0:
                return 'Minute-Plus'
            elif bar_span == 60.0:
                return 'Minute'
            else:
                return 'Sub-Minute'

    property bar_end_time:
        def __get__(self):
            return datetime.fromtimestamp(self.header.meta_info.timestamp, tz=C_PROFILE.time_zone)

    property bar_start_time:
        def __get__(self):
            return datetime.fromtimestamp(self.header.meta_info.timestamp, tz=C_PROFILE.time_zone)


cdef class DailyBar(BarData):
    def __init__(
            self,
            *,
            str ticker,
            date market_date,
            double high_price,
            double low_price,
            double open_price,
            double close_price,
            double volume=0.0,
            double notional=0.0,
            uint64_t trade_count=0,
            int bar_span=1,
            **kwargs
    ):
        cdef double timestamp = 10000 * market_date.year + 100 * market_date.month + market_date.day

        super().__init__(
            ticker=ticker,
            timestamp=timestamp,
            high_price=high_price,
            low_price=low_price,
            open_price=open_price,
            close_price=close_price,
            volume=volume,
            notional=notional,
            trade_count=trade_count,
            bar_span=bar_span,
            **kwargs
        )

    def __repr__(self) -> str:
        if not self.header:
            return f"<{self.__class__.__name__}>(Uninitialized)"
        cdef double bar_span_seconds = self.header.bar_data.bar_span
        cdef uint64_t bar_span = <uint64_t> bar_span_seconds
        if bar_span == 1:
            return f"<{self.__class__.__name__}>([{self.market_date}] {self.ticker}, open={self.open_price}, high={self.high_price}, low={self.low_price}, close={self.close_price}, volume={self.volume})"
        else:
            return f"<{self.__class__.__name__}>([{self.market_date}] {self.ticker}, span={bar_span}d, open={self.open_price}, high={self.high_price}, low={self.low_price}, close={self.close_price}, volume={self.volume})"

    property market_date:
        def __get__(self):
            cdef size_t int_date = int(self.timestamp)
            cdef size_t y, m, d, _m
            y, _m = divmod(int_date, 10000)
            m, d = divmod(_m, 100)
            return date(year=y, month=m, day=d)

    property market_time:
        def __get__(self):
            return self.market_date

    property bar_start_time:
        def __get__(self):
            return self.market_date - self.bar_span

    property bar_end_time:
        def __get__(self):
            return self.market_date

    property bar_span:
        def __get__(self):
            cdef double bar_span_seconds = self.header.bar_data.bar_span
            return timedelta(days=bar_span_seconds)

    property bar_type:
        def __get__(self):
            cdef double bar_span_seconds = self.header.bar_data.bar_span
            cdef uint64_t bar_span = <uint64_t> bar_span_seconds
            if bar_span_seconds == 1.0:
                return 'Daily'
            elif bar_span_seconds > 1.0:
                return 'Daily-Plus'
            else:
                raise ValueError(f'Invalid bar_span for {self.__class__.__name__}! Expect an int greater or equal to 1, got {bar_span_seconds}.')


from . cimport c_market_data

c_market_data.bar_from_header = bar_from_header
