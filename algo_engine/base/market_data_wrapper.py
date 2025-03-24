import abc
import datetime
import enum
import logging
from typing import Self, Literal

import candlestick as candlestick_cython
import market_data as market_data_cython
import transaction as transaction_cython

LOGGER = logging.getLogger(__name__)
from algo_engine.profile import PROFILE


class Direction(enum.IntEnum):
    DIRECTION_UNKNOWN = 1
    DIRECTION_SHORT = 0
    DIRECTION_LONG = 2


class Offset(enum.IntEnum):
    OFFSET_CANCEL = 0
    OFFSET_ORDER = 4
    OFFSET_OPEN = 8
    OFFSET_CLOSE = 16


class TransactionSide(enum.IntEnum):
    # Long Side
    SIDE_LONG_OPEN = Direction.DIRECTION_LONG + Offset.OFFSET_OPEN  # 2 + 8 = 10
    SIDE_LONG_CLOSE = Direction.DIRECTION_LONG + Offset.OFFSET_CLOSE  # 2 + 16 = 18
    SIDE_LONG_CANCEL = Direction.DIRECTION_LONG + Offset.OFFSET_CANCEL  # 2 + 0 = 2

    # Short Side
    SIDE_SHORT_OPEN = Direction.DIRECTION_SHORT + Offset.OFFSET_OPEN  # 0 + 8 = 8
    SIDE_SHORT_CLOSE = Direction.DIRECTION_SHORT + Offset.OFFSET_CLOSE  # 0 + 16 = 16
    SIDE_SHORT_CANCEL = Direction.DIRECTION_SHORT + Offset.OFFSET_CANCEL  # 0 + 0 = 0

    # Order
    SIDE_BID = Direction.DIRECTION_LONG + Offset.OFFSET_ORDER  # 2 + 4 = 6
    SIDE_ASK = Direction.DIRECTION_SHORT + Offset.OFFSET_ORDER  # 0 + 4 = 4

    # Generic Cancel
    SIDE_CANCEL = Direction.DIRECTION_UNKNOWN + Offset.OFFSET_CANCEL  # 1 + 0 = 1

    # Alias
    SIDE_UNKNOWN = SIDE_CANCEL  # 1
    SIDE_LONG = SIDE_LONG_OPEN  # 10
    SIDE_SHORT = SIDE_SHORT_OPEN  # 8

    # Backward compatibility
    ShortOrder = AskOrder = Ask = SIDE_ASK
    LongOrder = BidOrder = Bid = SIDE_BID

    ShortFilled = Unwind = Sell = SIDE_SHORT_CLOSE
    LongFilled = LongOpen = Buy = SIDE_LONG_OPEN

    ShortOpen = Short = SIDE_SHORT_OPEN
    Cover = SIDE_LONG_CLOSE

    UNKNOWN = CANCEL = SIDE_CANCEL

    def __neg__(self) -> Self:
        return self.__class__([market_data_cython.TransactionHelper.pyget_opposite(self.value)])

    @classmethod
    def _missing_(cls, value: str | int):
        """
        Handle missing values in the enumeration.

        Args:
            value (str | int): The value to resolve.

        Returns:
            TransactionSide: The resolved transaction side, or UNKNOWN if not recognized.
        """
        side_str = str(value).lower()

        match side_str:
            case 'long' | 'buy' | 'b':
                trade_side = cls.SIDE_LONG_OPEN
            case 'short' | 'sell' | 's':
                trade_side = cls.SIDE_SHORT_CLOSE
            case 'short' | 'ss':
                trade_side = cls.SIDE_SHORT_OPEN
            case 'cover' | 'bc':
                trade_side = cls.SIDE_LONG_CLOSE
            case 'ask':
                trade_side = cls.SIDE_ASK
            case 'bid':
                trade_side = cls.SIDE_BID
            case _:
                try:
                    trade_side = cls.__getitem__(value)
                except Exception as _:
                    trade_side = cls.SIDE_UNKNOWN
                    LOGGER.warning('{} is not recognized, return TransactionSide.UNKNOWN'.format(value))

        return trade_side

    @property
    def sign(self) -> int:
        return market_data_cython.TransactionHelper.pyget_sign(self.value)

    @property
    def offset(self) -> Offset:
        """
        Get the offset of the transaction side.

        Returns:
            int: The offset value, equivalent to the sign.
        """
        return Offset(market_data_cython.TransactionHelper.pyget_offset(self.value))

    @property
    def direction(self) -> Direction:
        """
        Get the offset of the transaction side.

        Returns:
            int: The offset value, equivalent to the sign.
        """
        return Direction(market_data_cython.TransactionHelper.pyget_direction(self.value))

    @property
    def side_name(self) -> str:
        return market_data_cython.TransactionHelper.pyget_side_name(self.value)

    @property
    def offset_name(self) -> str:
        return market_data_cython.TransactionHelper.pyget_offset_name(self.value)

    @property
    def direction_name(self) -> str:
        return market_data_cython.TransactionHelper.pyget_direction_name(self.value)


class OrderType(enum.IntEnum):
    UNKNOWN = -20
    CancelOrder = -10
    Generic = 0
    LimitOrder = 10
    LimitMarketMaking = 11
    MarketOrder = 2
    FOK = 21
    FAK = 22
    IOC = 23


class MarketData(market_data_cython.MarketData, metaclass=abc.ABCMeta):
    """
    Python wrapper for MarketData Cython class.
    Provides pickle serialization support.
    """

    @classmethod
    def from_buffer(cls, buffer, **kwargs):
        self = super().from_buffer(buffer)
        self.__dict__.update(kwargs)
        return self

    @abc.abstractmethod
    def __reduce__(self):
        ...

    def __copy__(self):
        return self.__class__.from_buffer(memoryview(self), **self.__dict__)

    @property
    def topic(self) -> str:
        return f'{self.ticker}.{self.__class__.__name__}'

    @property
    def market_time(self) -> datetime.datetime | datetime.date:
        return datetime.datetime.fromtimestamp(self.timestamp, tz=PROFILE.time_zone)


class TransactionData(transaction_cython.TransactionData, MarketData):
    def __init__(
            self,
            ticker: str,
            timestamp: float,
            price: float,
            volume: float,
            side: int | TransactionSide,
            multiplier: float = 1.0,
            notional: float = 0.0,
            transaction_id: int | str | bytes = None,
            buy_id: int | str | bytes = None,
            sell_id: int | str | bytes = None,
            **kwargs
    ):
        super().__init__(ticker=ticker, timestamp=timestamp, price=price, volume=volume, side=side, multiplier=multiplier, notional=notional, transaction_id=transaction_id, buy_id=buy_id, sell_id=sell_id)
        self.__dict__.update(kwargs)

    def __reduce__(self):
        """Support for pickle serialization"""
        return self.__class__.from_bytes, (self.to_bytes(),), self.__dict__

    def __setstate__(self, state):
        """Restore state from pickle"""
        self.__dict__.update(state)

    @property
    def side_int(self) -> int:
        return super().side

    @property
    def side(self) -> TransactionSide:
        return TransactionSide(super().side)


class BarData(candlestick_cython.BarData, MarketData):
    def __init__(
            self,
            ticker: str,
            timestamp: float,
            high_price: float,
            low_price: float,
            open_price: float,
            close_price: float,
            volume: float = 0.0,
            notional: float = 0.0,
            trade_count: int = 0,
            start_timestamp: float = None,
            bar_span: float = None,
            **kwargs: object
    ) -> None:
        if bar_span is None:
            if start_timestamp is None:
                raise ValueError('Must assign either start_timestamp or bar_span or both.')
            else:
                bar_span = timestamp - start_timestamp
        else:
            if isinstance(bar_span, datetime.timedelta):
                bar_span = bar_span.total_seconds()
            else:
                bar_span = float(bar_span)

        super().__init__(ticker=ticker, timestamp=timestamp, high_price=high_price, low_price=low_price, open_price=open_price, close_price=close_price, volume=volume, notional=notional, trade_count=trade_count, bar_span=bar_span)
        self.__dict__.update(kwargs)

    def __reduce__(self):
        """Support for pickle serialization"""
        return self.__class__.from_bytes, (self.to_bytes(),), self.__dict__

    def __setstate__(self, state):
        """Restore state from pickle"""
        self.__dict__.update(state)

    @property
    def bar_type(self) -> Literal['Hourly-Plus', 'Hourly', 'Minute-Plus', 'Minute', 'Sub-Minute']:
        """
        Determines the type of the bar based on its span.

        Returns:
            Literal['Hourly-Plus', 'Hourly', 'Minute-Plus', 'Minute', 'Sub-Minute']: The type of the bar.
        """
        bar_span = super().bar_span

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
    def bar_end_time(self) -> datetime.datetime | datetime.date:
        """
        The end time of the bar.

        Returns:
            datetime.datetime | datetime.date: The end time of the bar.
        """
        return self.market_time

    @property
    def bar_start_time(self) -> datetime.datetime:
        """
        The start time of the bar.

        Returns:
            datetime.datetime: The start time of the bar.
        """
        return datetime.datetime.fromtimestamp(super().start_timestamp, tz=PROFILE.time_zone)

    @property
    def bar_span(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=super().bar_span)


class DailyBar(candlestick_cython.BarData):
    def __init__(
            self,
            ticker: str,
            market_date: datetime.date,  # The market date of the bar, if with 1D data, or the END date of the bar.
            high_price: float,
            low_price: float,
            open_price: float,
            close_price: float,
            volume: float = 0.0,
            notional: float = 0.0,
            trade_count: int = 0,
            bar_span: datetime.timedelta | int = None,  # Expect to be a timedelta for several days, or the number of days
            start_date: datetime.date = None,
            **kwargs
    ):
        if bar_span is None:
            if start_date is None:
                bar_span = 1
            else:
                bar_span = (market_date - start_date).days
        else:
            if isinstance(bar_span, datetime.timedelta):
                bar_span = bar_span.days
            else:
                bar_span = float(bar_span)

        timestamp = 10000 * market_date.year + 100 * market_date.month + market_date.day

        super().__init__(ticker=ticker, timestamp=timestamp, high_price=high_price, low_price=low_price, open_price=open_price, close_price=close_price, volume=volume, notional=notional, trade_count=trade_count, bar_span=bar_span)
        self.__dict__.update(kwargs)

    def __repr__(self) -> str:
        if (bar_span := super().bar_span) == 1:
            return f"DailyBar(ticker='{self.ticker}', date={self.market_date}, open={self.open_price}, high={self.high_price}, low={self.low_price}, close={self.close_price}, volume={self.volume})"
        else:
            return f"DailyBar(ticker='{self.ticker}', date={self.market_date}, span={bar_span}d, open={self.open_price}, high={self.high_price}, low={self.low_price}, close={self.close_price}, volume={self.volume})"

    def __reduce__(self):
        """Support for pickle serialization"""
        return self.__class__.from_bytes, (self.to_bytes(),), self.__dict__

    def __setstate__(self, state):
        """Restore state from pickle"""
        self.__dict__.update(state)

    @property
    def market_date(self) -> datetime.date:
        """
        The market date of the bar.

        Returns:
            datetime.date: The market date of the bar.
        """

        int_date = int(self.timestamp)
        y, _m = divmod(int_date, 10000)
        m, d = divmod(_m, 100)

        return datetime.date(year=y, month=m, day=d)

    @property
    def market_time(self) -> datetime.date:
        """
        The market date of the bar (same as `market_date`).

        Returns:
            datetime.date: The market date of the bar.
        """
        return self.market_date

    @property
    def bar_start_time(self) -> datetime.date:
        """
        The start date of the bar period.

        Returns:
            datetime.date: The start date of the bar.
        """
        return self.market_date - self.bar_span

    @property
    def bar_end_time(self) -> datetime.date:
        """
        The end date of the bar period.

        Returns:
            datetime.date: The end date of the bar.
        """
        return self.market_date

    @property
    def bar_span(self) -> datetime.timedelta:
        return datetime.timedelta(days=super().bar_span)

    @property
    def bar_type(self) -> Literal['Daily', 'Daily-Plus']:
        """
        Determines the type of the bar based on its span.

        Returns:
            Literal['Daily', 'Daily-Plus']: The type of the bar.

        Raises:
            ValueError: If `bar_span` is not valid for a daily bar.
        """
        if super().bar_span == 1:
            return 'Daily'
        elif super().bar_span > 1:
            return 'Daily-Plus'
        else:
            raise ValueError(f'Invalid bar_span for {self.__class__.__name__}! Expect an int greater or equal to 1, got {super().bar_span}.')


class TickDataLite(market_data_cython.TickDataLite, MarketData):
    """
    Python wrapper for TickDataLite Cython class.
    Represents tick data for a specific ticker without the order_book field.
    """

    def __init__(
            self,
            ticker: str,
            timestamp: float,
            last_price: float,
            bid_price: float = float('nan'),
            bid_volume: float = float('nan'),
            ask_price: float = float('nan'),
            ask_volume: float = float('nan'),
            total_traded_volume: float = 0.0,
            total_traded_notional: float = 0.0,
            total_trade_count: int = 0,
            **kwargs
    ):
        """
        Initialize a new instance of TickDataLite.

        Args:
            ticker (str): The ticker symbol for the market data.
            timestamp (float): The timestamp of the tick data.
            last_price (float): The last traded price.
            bid_price (float, optional): The bid price. Defaults to None.
            bid_volume (float, optional): The bid volume. Defaults to None.
            ask_price (float, optional): The ask price. Defaults to None.
            ask_volume (float, optional): The ask volume. Defaults to None.
            total_traded_volume (float, optional): The total traded volume. Defaults to 0.0.
            total_traded_notional (float, optional): The total traded notional value. Defaults to 0.0.
            total_trade_count (int, optional): The total number of trades. Defaults to 0.
            **kwargs: Additional keyword arguments.
        """
        # Create the Cython object
        super().__init__(
            ticker=ticker,
            timestamp=timestamp,
            last_price=last_price,
            bid_price=bid_price,
            bid_volume=bid_volume,
            ask_price=ask_price,
            ask_volume=ask_volume,
            total_traded_volume=total_traded_volume,
            total_traded_notional=total_traded_notional,
            total_trade_count=total_trade_count
        )
        self.__dict__.update(kwargs)

    def __reduce__(self):
        """Support for pickle serialization"""
        return self.__class__.from_bytes, (self.to_bytes(),), self.__dict__

    def __setstate__(self, state):
        """Restore state from pickle"""
        self.__dict__.update(state)

    def __repr__(self) -> str:
        """
        Returns a string representation of the TickDataLite instance.

        Returns:
            str: A string representation of the TickDataLite instance.
        """
        return f'<TickDataLite>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker}, bid={self.bid_price}, ask={self.ask_price})'

    @classmethod
    def from_buffer(cls, buffer, **kwargs):
        self = super().from_buffer(buffer)
        self.__dict__.update(kwargs)
        return self
