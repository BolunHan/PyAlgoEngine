import abc
import datetime
import enum
import inspect
from typing import TypeVar, Any, Self, Literal
import logging
import market_data as market_data_cython

LOGGER = logging.getLogger(__name__)
from algo_engine.profile import PROFILE


class TransactionSide(enum.IntEnum):
    ShortOrder = AskOrder = Offer_to_Short = -3
    ShortOpen = Sell_to_Short = -2
    ShortFilled = LongClose = Sell_to_Unwind = ask = -1
    UNKNOWN = CANCEL = 0
    LongFilled = LongOpen = Buy_to_Long = bid = 1
    ShortClose = Buy_to_Cover = 2
    LongOrder = BidOrder = Bid_to_Long = 3

    def __neg__(self) -> Self:
        """
        Get the opposite transaction side.

        Returns:
            TransactionSide: The opposite transaction side.
        """
        if self is self.LongOpen:
            return self.LongClose
        elif self is self.LongClose:
            return self.LongOpen
        elif self is self.ShortOpen:
            return self.ShortClose
        elif self is self.ShortClose:
            return self.ShortOpen
        elif self is self.BidOrder:
            return self.AskOrder
        elif self is self.AskOrder:
            return self.BidOrder
        else:
            LOGGER.warning('No valid registered opposite trade side for {}'.format(self))
            return self.UNKNOWN

    @classmethod
    def from_offset(cls, direction: str, offset: str) -> Self:
        """
        Determine the transaction side from direction and offset.

        Args:
            direction (str): The trade direction (e.g., 'buy', 'sell').
            offset (str): The trade offset (e.g., 'open', 'close').

        Returns:
            TransactionSide: The corresponding transaction side.

        Raises:
            ValueError: If the direction or offset is not recognized.
        """
        direction = direction.lower()
        offset = offset.lower()

        if direction in ['buy', 'long', 'b']:
            if offset in ['open', 'wind']:
                return cls.LongOpen
            elif offset in ['close', 'cover', 'unwind']:
                return cls.ShortOpen
            else:
                raise ValueError(f'Not recognized {direction} {offset}')
        elif direction in ['sell', 'short', 's']:
            if offset in ['open', 'wind']:
                return cls.ShortOpen
            elif offset in ['close', 'cover', 'unwind']:
                return cls.LongClose
            else:
                raise ValueError(f'Not recognized {direction} {offset}')
        else:
            raise ValueError(f'Not recognized {direction} {offset}')

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
                trade_side = cls.LongOpen
            case 'short' | 'sell' | 's':
                trade_side = cls.LongClose
            case 'short' | 'ss':
                trade_side = cls.ShortOpen
            case 'cover' | 'bc':
                trade_side = cls.ShortClose
            case 'ask':
                trade_side = cls.AskOrder
            case 'bid':
                trade_side = cls.BidOrder
            case _:
                try:
                    trade_side = cls.__getitem__(value)
                except Exception as _:
                    trade_side = cls.UNKNOWN
                    LOGGER.warning('{} is not recognized, return TransactionSide.UNKNOWN'.format(value))

        return trade_side

    @property
    def sign(self) -> int:
        """
        Get the sign of the transaction side.

        Returns:
            int: 1 for buy/long, -1 for sell/short, 0 for unknown.
        """
        if self.value == self.Buy_to_Long.value or self.value == self.Buy_to_Cover.value:
            return 1
        elif self.value == self.Sell_to_Unwind.value or self.value == self.Sell_to_Short.value:
            return -1
        elif self.value == 0:
            return 0
        else:
            frame = inspect.currentframe()
            caller = inspect.getframeinfo(frame.f_back)
            LOGGER.warning(f"Requesting .sign of {self.name} is not recommended, use .order_sign instead.\nCalled from {caller.filename}, line {caller.lineno}.")
            return self.order_sign

    @property
    def order_sign(self) -> int:
        """
        Get the order sign of the transaction side.

        Returns:
            int: 1 for long orders, -1 for short orders, 0 for unknown.
        """
        if self.value == self.LongOrder.value:
            return 1
        elif self.value == self.ShortOrder.value:
            return -1
        elif self.value == 0:
            return 0
        else:
            LOGGER.warning(f'Requesting .order_sign of {self.name} is not recommended, use .sign instead')
            return self.sign

    @property
    def offset(self) -> int:
        """
        Get the offset of the transaction side.

        Returns:
            int: The offset value, equivalent to the sign.
        """
        return self.sign

    @property
    def side_name(self) -> str:
        """
        Get the name of the transaction side.

        Returns:
            str: 'Long', 'Short', 'ask', 'bid', or 'Unknown'.
        """
        if self.value == self.Buy_to_Long.value or self.value == self.Buy_to_Cover.value:
            return 'Long'
        elif self.value == self.Sell_to_Unwind.value or self.value == self.Sell_to_Short.value:
            return 'Short'
        elif self.value == self.Offer_to_Short.value:
            return 'ask'
        elif self.value == self.Bid_to_Long.value:
            return 'bid'
        else:
            return 'Unknown'

    @property
    def offset_name(self) -> str:
        """
        Get the offset name of the transaction side.

        Returns:
            str: 'Open', 'Close', 'ask', 'bid', or 'Unknown'.
        """
        if self.value == self.Buy_to_Long.value or self.value == self.Sell_to_Short.value:
            return 'Open'
        elif self.value == self.Buy_to_Cover.value or self.value == self.Sell_to_Unwind.value:
            return 'Close'
        elif self.value == self.Offer_to_Short.value or self.value == self.Bid_to_Long.value:
            LOGGER.warning(f'Requesting offset of {self.name} is not supported, returns {self.side_name}')
            return self.side_name
        else:
            return 'Unknown'


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


class TransactionData(market_data_cython.TransactionData, MarketData):
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


class BarData(market_data_cython.BarData, MarketData):
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


class DailyBar(market_data_cython.BarData):
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
                bar_span = bar_span.days()
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


class OrderBook(market_data_cython.OrderBook, MarketData):
    """
    Python wrapper for OrderBook Cython class.
    Provides pickle serialization support and additional functionality.
    """

    def __init__(
            self,
            ticker: str,
            timestamp: float,
            bid: list[list[float | int]] = None,
            ask: list[list[float | int]] = None,
            **kwargs
    ):
        """
        Initialize the order book with values.

        Args:
            ticker (str): The ticker symbol for the market data.
            timestamp (float): The timestamp of the order book.
            bid (list, optional): List of bid entries [price, volume, n_orders].
            ask (list, optional): List of ask entries [price, volume, n_orders].
            **kwargs: Additional keyword arguments.
        """
        super().__init__(ticker=ticker, timestamp=timestamp, bid=bid, ask=ask)
        self.__dict__.update(kwargs)

    def __reduce__(self):
        """Support for pickle serialization"""
        return self.__class__.from_bytes, (self.to_bytes(),), self.__dict__

    def __setstate__(self, state):
        """Restore state from pickle"""
        self.__dict__.update(state)

    def update(self, bid=None, ask=None):
        """
        Update the order book with new bid and ask data.

        Args:
            bid (list, optional): List of bid entries [price, volume, n_orders].
            ask (list, optional): List of ask entries [price, volume, n_orders].

        Returns:
            OrderBook: Self for method chaining.
        """
        if bid is not None:
            self.update_bid(bid)
        if ask is not None:
            self.update_ask(ask)
        return self
