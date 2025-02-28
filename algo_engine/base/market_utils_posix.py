import abc
import ctypes
import datetime
import enum
import inspect
import json
import math
import re
import warnings
from collections import namedtuple
from collections.abc import Iterable
from multiprocessing import RawValue, RawArray, Condition
from typing import overload, Literal, Self

import numpy as np

from . import LOGGER, PROFILE

LOGGER = LOGGER.getChild('MarketUtils')
__all__ = ['TransactionSide', 'OrderType',
           'MarketData', 'OrderBook', 'BarData', 'DailyBar', 'CandleStick', 'TickData', 'TransactionData', 'TradeData', 'OrderData',
           'MarketDataBuffer', 'MarketDataRingBuffer']
__cache__ = {}

# TICKER_SIZE: int = 16
# ID_SIZE: int = 16
# BOOK_SIZE: int = 10

Contexts = namedtuple(
    typename='Contexts',
    field_names=['TICKER_SIZE', 'ID_SIZE', 'BOOK_SIZE'],
    defaults=[16, 16, 10],
)()

DTYPE_MAPPING: dict[str, int] = {
    'OrderBook': 10,
    'BarData': 20,
    'DailyBar': 21,
    'TickData': 30,
    'TransactionData': 40,
    'TradeData': 41,
    'OrderData': 50
}


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
        capital_str = str(value).capitalize()

        if capital_str == 'Long' or capital_str == 'Buy' or capital_str == 'B':
            trade_side = cls.LongOpen
        elif capital_str == 'Short' or capital_str == 'Ss':
            trade_side = cls.ShortOpen
        elif capital_str == 'Close' or capital_str == 'Sell' or capital_str == 'S':
            trade_side = cls.LongClose
        elif capital_str == 'Cover' or capital_str == 'Bc':
            trade_side = cls.ShortClose
        elif capital_str == 'Ask':
            trade_side = cls.AskOrder
        elif capital_str == 'Bid':
            trade_side = cls.BidOrder
        else:
            # noinspection PyBroadException
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
            LOGGER.warning(
                f"Requesting .sign of {self.name} is not recommended, use .order_sign instead. "
                f"Called from {caller.filename}, line {caller.lineno}."
            )
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


class MarketData(object, metaclass=abc.ABCMeta):
    def __init__(self, buffer: ctypes.Structure, encoding='utf-8', **kwargs):
        self._buffer = buffer
        self._encoding = encoding

        self._buffer.dtype = DTYPE_MAPPING[self.__class__.__name__]

        if kwargs:
            self._additional = dict(kwargs)

    def __copy__(self):
        new_md = self.__class__(
            buffer=self._buffer.__class__.from_buffer_copy(self._buffer)
        )

        if hasattr(self, '_additional'):
            new_md._additional = self._additional.copy()

        return new_md

    def __setitem__(self, key: str, value):
        setattr(self._buffer, key, value)

    def __getitem__(self, key):
        warnings.warn(f'getitem from {self.__class__.__name__} deprecated!', DeprecationWarning, stacklevel=2)
        LOGGER.warning(f'getitem from {self.__class__.__name__} deprecated!')
        return getattr(self._buffer, key)

    def __reduce__(self):
        return self.__class__.from_bytes, (bytes(self._buffer),)

    def copy(self):
        return self.__copy__()

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            dtype=self.__class__.__name__,
            **{name: value[:] if isinstance(value := getattr(self._buffer, name), ctypes.Array) else value for name in self.fields}
        )

        if hasattr(self, '_additional'):
            data_dict.update(self._additional)

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, expected "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        dtype = json_dict.pop('dtype', None)

        if dtype == 'BarData':
            return BarData.from_json(json_dict)
        elif dtype == 'DailyBar':
            return DailyBar.from_json(json_dict)
        elif dtype == 'TickData':
            return TickData.from_json(json_dict)
        elif dtype == 'TransactionData':
            return TransactionData.from_json(json_dict)
        elif dtype == 'TradeData':
            return TradeData.from_json(json_dict)
        elif dtype == 'OrderBook':
            return OrderBook.from_json(json_dict)
        elif dtype == 'OrderData':
            return OrderData.from_json(json_dict)
        else:
            raise TypeError(f'Invalid dtype {dtype}')

    @classmethod
    def parse_buffer(cls, buffer: ctypes.Union) -> ctypes.Structure:
        dtype_int = buffer.dtype

        if dtype_int == 10:
            return buffer.OrderBook
        elif dtype_int == 20:
            return buffer.BarData
        elif dtype_int == 21:
            return buffer.BarData
        elif dtype_int == 30:
            return buffer.TickData
        elif dtype_int == 40:
            return buffer.TransactionData
        elif dtype_int == 41:
            return buffer.TransactionData
        elif dtype_int == 50:
            return buffer.OrderData
        else:
            raise ValueError(f'Invalid buffer type {dtype_int}!')

    @classmethod
    def cast_buffer(cls, buffer: ctypes.Structure | ctypes.Union | memoryview) -> Self:
        dtype_int = buffer.dtype

        if dtype_int == 10:
            buffer = _BUFFER_CONSTRUCTOR.new_orderbook_buffer().from_buffer(buffer)
            return OrderBook.from_buffer(buffer=buffer)
        elif dtype_int == 20:
            buffer = _BUFFER_CONSTRUCTOR.new_candlestick_buffer().from_buffer(buffer)
            return BarData.from_buffer(buffer=buffer)
        elif dtype_int == 21:
            buffer = _BUFFER_CONSTRUCTOR.new_candlestick_buffer().from_buffer(buffer)
            return DailyBar.from_buffer(buffer=buffer)
        elif dtype_int == 30:
            buffer = _BUFFER_CONSTRUCTOR.new_tick_buffer().from_buffer(buffer)
            return TickData.from_buffer(buffer=buffer)
        elif dtype_int == 40:
            buffer = _BUFFER_CONSTRUCTOR.new_transaction_buffer().from_buffer(buffer)
            return TransactionData.from_buffer(buffer=buffer)
        elif dtype_int == 41:
            buffer = _BUFFER_CONSTRUCTOR.new_transaction_buffer().from_buffer(buffer)
            return TradeData.from_buffer(buffer=buffer)
        elif dtype_int == 50:
            buffer = _BUFFER_CONSTRUCTOR.new_order_buffer().from_buffer(buffer)
            return OrderData.from_buffer(buffer=buffer)
        else:
            raise ValueError(f'Invalid buffer type {dtype_int}!')

    @classmethod
    def from_buffer(cls, buffer: ctypes.Structure) -> Self:
        dtype_int = buffer.dtype

        if dtype_int == 10:
            return OrderBook.from_buffer(buffer=buffer)
        elif dtype_int == 20:
            return BarData.from_buffer(buffer=buffer)
        elif dtype_int == 21:
            return DailyBar.from_buffer(buffer=buffer)
        elif dtype_int == 30:
            return TickData.from_buffer(buffer=buffer)
        elif dtype_int == 40:
            return TransactionData.from_buffer(buffer=buffer)
        elif dtype_int == 41:
            return TradeData.from_buffer(buffer=buffer)
        elif dtype_int == 50:
            return OrderData.from_buffer(buffer=buffer)
        else:
            raise ValueError(f'Invalid buffer type {dtype_int}!')

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        dtype_int = data[0]

        if dtype_int == 10:
            buffer = _BUFFER_CONSTRUCTOR.new_orderbook_buffer().from_buffer_copy(data)
            return OrderBook.from_buffer(buffer=buffer)
        elif dtype_int == 20:
            buffer = _BUFFER_CONSTRUCTOR.new_candlestick_buffer().from_buffer_copy(data)
            return BarData.from_buffer(buffer=buffer)
        elif dtype_int == 21:
            buffer = _BUFFER_CONSTRUCTOR.new_candlestick_buffer().from_buffer_copy(data)
            return DailyBar.from_buffer(buffer=buffer)
        elif dtype_int == 30:
            buffer = _BUFFER_CONSTRUCTOR.new_tick_buffer().from_buffer_copy(data)
            return TickData.from_buffer(buffer=buffer)
        elif dtype_int == 40:
            buffer = _BUFFER_CONSTRUCTOR.new_transaction_buffer().from_buffer_copy(data)
            return TransactionData.from_buffer(buffer=buffer)
        elif dtype_int == 41:
            buffer = _BUFFER_CONSTRUCTOR.new_transaction_buffer().from_buffer_copy(data)
            return TradeData.from_buffer(buffer=buffer)
        elif dtype_int == 50:
            buffer = _BUFFER_CONSTRUCTOR.new_order_buffer().from_buffer_copy(data)
            return OrderData.from_buffer(buffer=buffer)
        else:
            raise ValueError(f'Invalid buffer type {dtype_int}!')

    @property
    def ticker(self) -> str:
        ticker_bytes = self._buffer.ticker
        return ticker_bytes.decode(self._encoding)

    @property
    def timestamp(self) -> float:
        return self._buffer.timestamp

    @property
    def additional(self) -> dict:
        if hasattr(self, '_additional'):
            return self._additional

    @property
    def topic(self) -> str:
        return f'{self.ticker}.{self.__class__.__name__}'

    @property
    def market_time(self) -> datetime.datetime | datetime.date:
        return datetime.datetime.fromtimestamp(self.timestamp, tz=PROFILE.time_zone)

    @property
    def fields(self) -> Iterable[str]:
        return tuple(name for name, *_ in getattr(self._buffer, '_fields_'))

    @property
    def byte_size(self) -> int:
        return ctypes.sizeof(self._buffer)

    @property
    @abc.abstractmethod
    def market_price(self) -> float:
        ...


class BufferConstructor(object):

    def __init__(self, **kwargs):
        self._cache = kwargs.get('cache', __cache__)

        if 'MarketData' in self._cache:
            self._md_cache = self._cache['MarketData']
        else:
            self._md_cache = self._cache['MarketData'] = dict()

        if 'OrderBook' in self._cache:
            self._orderbook_cache = self._cache['OrderBook']
        else:
            self._orderbook_cache = self._cache['OrderBook'] = dict()

        if 'BarData' in self._cache:
            self._candlestick_cache = self._cache['BarData']
        else:
            self._candlestick_cache = self._cache['BarData'] = dict()

        if 'TickData' in self._cache:
            self._tick_cache = self._cache['TickData']
        else:
            self._tick_cache = self._cache['TickData'] = dict()

        if 'TransactionData' in self._cache:
            self._trade_cache = self._cache['TransactionData']
        else:
            self._trade_cache = self._cache['TransactionData'] = dict()

        if 'OrderData' in self._cache:
            self._order_cache = self._cache['OrderData']
        else:
            self._order_cache = self._cache['OrderData'] = dict()

    def __call__(self, dtype: 'str') -> type[ctypes.Structure]:
        match dtype:
            case 'MarketData':
                return self.new_market_data_buffer()
            case 'OrderBook':
                return self.new_orderbook_buffer()
            case 'BarData':
                return self.new_candlestick_buffer()
            case 'TickData':
                return self.new_tick_buffer()
            case 'TradeData' | 'TransactionData':
                return self.new_transaction_buffer()
            case 'OrderData':
                return self.new_order_buffer()
            case _:
                raise ValueError(f'Invalid dtype {dtype}')

    def new_id_buffer(self):
        class IntID(ctypes.Structure):
            id_size = Contexts.ID_SIZE

            _fields_ = [
                ('id_type', ctypes.c_int),
                ('data', ctypes.c_byte * id_size),
            ]

        class StrID(ctypes.Structure):
            id_size = Contexts.ID_SIZE

            _fields_ = [
                ('id_type', ctypes.c_int),
                ('data', ctypes.c_char * id_size),
            ]

        class UnionID(ctypes.Union):
            id_size = Contexts.ID_SIZE

            _fields_ = [
                ('id_type', ctypes.c_int),
                ('id_int', IntID),
                ('id_str', StrID),
            ]

        return UnionID

    def new_orderbook_buffer(self) -> type[ctypes.Structure]:
        if (key := (Contexts.TICKER_SIZE, Contexts.BOOK_SIZE)) in self._orderbook_cache:
            return self._orderbook_cache[key]

        class _Buffer(ctypes.Structure):
            ticker_size = Contexts.TICKER_SIZE
            book_size = Contexts.BOOK_SIZE

            _fields_ = [
                ("dtype", ctypes.c_uint8),
                ("ticker", ctypes.c_char * ticker_size),
                ("timestamp", ctypes.c_double),
                ('bid_price', ctypes.c_double * book_size),
                ('ask_price', ctypes.c_double * book_size),
                ('bid_volume', ctypes.c_double * book_size),
                ('ask_volume', ctypes.c_double * book_size),
                ('bid_n_orders', ctypes.c_uint * book_size),
                ('ask_n_orders', ctypes.c_uint * book_size)
            ]

        self._orderbook_cache[key] = _Buffer
        return _Buffer

    def new_candlestick_buffer(self) -> type[ctypes.Structure]:
        if (key := Contexts.TICKER_SIZE) in self._candlestick_cache:
            return self._candlestick_cache[key]

        class _Buffer(ctypes.Structure):
            ticker_size = Contexts.TICKER_SIZE

            _fields_ = [
                ("dtype", ctypes.c_uint8),
                ("ticker", ctypes.c_char * ticker_size),
                ("timestamp", ctypes.c_double),
                ('start_timestamp', ctypes.c_double),
                ('bar_span', ctypes.c_double),
                ('high_price', ctypes.c_double),
                ('low_price', ctypes.c_double),
                ('open_price', ctypes.c_double),
                ('close_price', ctypes.c_double),
                ('volume', ctypes.c_double),
                ('notional', ctypes.c_double),
                ('trade_count', ctypes.c_uint),
            ]

        self._candlestick_cache[key] = _Buffer
        return _Buffer

    def new_tick_buffer(self) -> type[ctypes.Structure]:
        if (key := Contexts.TICKER_SIZE) in self._tick_cache:
            return self._tick_cache[key]

        class _Buffer(ctypes.Structure):
            ticker_size = Contexts.TICKER_SIZE

            _fields_ = [
                ("dtype", ctypes.c_uint8),
                ("ticker", ctypes.c_char * ticker_size),
                ("timestamp", ctypes.c_double),
                ('order_book', self.new_orderbook_buffer()),
                ('bid_price', ctypes.c_double),
                ('bid_volume', ctypes.c_double),
                ('ask_price', ctypes.c_double),
                ('ask_volume', ctypes.c_double),
                ('last_price', ctypes.c_double),
                ('total_traded_volume', ctypes.c_double),
                ('total_traded_notional', ctypes.c_double),
                ('total_trade_count', ctypes.c_uint),
            ]

        self._tick_cache[key] = _Buffer
        return _Buffer

    def new_transaction_buffer(self) -> type[ctypes.Structure]:
        if (key := (Contexts.TICKER_SIZE, Contexts.ID_SIZE)) in self._trade_cache:
            return self._trade_cache[key]

        TransactionID = self.new_id_buffer()

        class _Buffer(ctypes.Structure):
            ticker_size = Contexts.TICKER_SIZE
            id_size = Contexts.ID_SIZE

            _fields_ = [
                ("dtype", ctypes.c_uint8),
                ("ticker", ctypes.c_char * ticker_size),  # Dynamic size based on TICKER_LEN
                ("timestamp", ctypes.c_double),
                ("price", ctypes.c_double),
                ("volume", ctypes.c_double),
                ("side", ctypes.c_int),
                ("multiplier", ctypes.c_double),
                ("notional", ctypes.c_double),
                ("transaction_id", TransactionID),
                ("buy_id", TransactionID),
                ("sell_id", TransactionID)
            ]

        self._trade_cache[key] = _Buffer
        return _Buffer

    def new_order_buffer(self) -> type[ctypes.Structure]:
        if (key := (Contexts.TICKER_SIZE, Contexts.ID_SIZE)) in self._order_cache:
            return self._trade_cache[key]

        OrderID = self.new_id_buffer()

        class _Buffer(ctypes.Structure):
            ticker_size = Contexts.TICKER_SIZE
            id_size = Contexts.ID_SIZE

            _fields_ = [
                ("dtype", ctypes.c_uint8),
                ("ticker", ctypes.c_char * ticker_size),  # Dynamic size based on TICKER_LEN
                ("timestamp", ctypes.c_double),
                ("price", ctypes.c_double),
                ("volume", ctypes.c_double),
                ("side", ctypes.c_int),
                ("order_id", OrderID),
                ("order_type", ctypes.c_int),
            ]

        self._order_cache[key] = _Buffer
        return _Buffer

    def new_market_data_buffer(self) -> type[ctypes.Union]:
        if (key := Contexts) in self._md_cache:
            return self._md_cache[key]

        class _Buffer(ctypes.Union):
            _fields_ = [
                ("dtype", ctypes.c_uint8),
                ("OrderBook", self.new_orderbook_buffer()),
                ("BarData", self.new_candlestick_buffer()),
                ("TickData", self.new_tick_buffer()),
                ("TransactionData", self.new_transaction_buffer()),
                ('OrderData', self.new_order_buffer())
            ]

        self._md_cache[key] = _Buffer
        return _Buffer


_BUFFER_CONSTRUCTOR = BufferConstructor()


class OrderBook(MarketData):
    """
    Class representing an order book, which tracks bid and ask orders for a financial instrument.

    Nested Classes:
        Book: Represents a side of the order book (either bid or ask), with methods for managing entries.
    """

    class Book(object):
        """
        Class representing a side of the order book (either bid or ask).

        Attributes:
            side (int): Indicates the side of the book; positive for bid, negative for ask.
            _book (list[tuple[float, float, ...]]): A list of tuples representing (price, volume, order).
            _dict (dict[float, tuple[float, float, ...]]): A dictionary mapping prices to order book entries.
            sorted (bool): Indicates whether the book is sorted.
        """

        def __init__(self, side: int):
            """
            Initialize the order book for a specific side.

            Args:
                side (int): Side of the book; positive for bid, negative for ask.
            """
            self.side: int = side
            # store the entry in order of (price, volume, order, etc...)
            self._book: list[tuple[float, float, ...]] = []
            self._dict: dict[float, tuple[float, float, ...]] = {}
            self.sorted = False

        def __iter__(self):
            """
            Iterate over the sorted order book.

            Returns:
                iterator: An iterator over the sorted book entries.
            """
            self.sort()
            return self._book.__iter__()

        def __getitem__(self, item):
            """
            Retrieve an entry by price or level.

            Args:
                item (int | float): Level number (int) or price (float).

            Returns:
                tuple[float, float, ...]: The order book entry at the specified level or price.

            Raises:
                KeyError: If the index value is ambiguous.
            """
            if isinstance(item, int) and item not in self._dict:
                return self.at_level(item)
            elif isinstance(item, float):
                return self.at_price(item)
            else:
                raise KeyError(f'Ambiguous index value {item}, please use at_price or at_level specifically')

        def __contains__(self, price: float):
            """
            Check if a price exists in the order book.

            Args:
                price (float): The price to check.

            Returns:
                bool: True if the price exists, False otherwise.
            """
            return self._dict.__contains__(price)

        def __len__(self):
            """
            Get the number of entries in the order book.

            Returns:
                int: The number of entries.
            """
            return self._book.__len__()

        def __repr__(self):
            """
            Get a string representation of the book.

            Returns:
                str: A string indicating whether the book is for bids or asks.
            """
            return f'<OrderBook.Book.{"Bid" if self.side > 0 else "Ask"}>'

        def __bool__(self):
            """
            Check if the order book has any entries.

            Returns:
                bool: True if the book is not empty, False otherwise.
            """
            return bool(self._book)

        def __sub__(self, other: Self) -> dict[float, float]:
            """
            Subtract another order book from this one to find the differences in volumes at matching prices.

            Args:
                other (OrderBook.Book): The other book to compare against.

            Returns:
                dict[float, float]: A dictionary of price differences.

            Raises:
                TypeError: If the other object is not of type OrderBook.Book.
                ValueError: If the sides of the books do not match.
            """
            if not isinstance(other, self.__class__):
                raise TypeError(f'Expect type {self.__class__.__name__}, got {type(other)}')

            if self.side != other.side:
                raise ValueError(f'Expect side {self.side}, got {other.side}')

            diff = {}

            # bid book
            if (not self._dict) or (not other._dict):
                pass
            elif self.side > 0:
                limit_0 = min(self._dict)
                limit_1 = min(other._dict)
                limit = max(limit_0, limit_1)
                contain_limit = limit_0 == limit_1

                for entry in self._book:
                    price, volume, *_ = entry

                    if price > limit or (price >= limit and contain_limit):
                        diff[price] = volume

                for entry in other._book:
                    price, volume, *_ = entry

                    if price > limit or (price >= limit and contain_limit):
                        diff[price] = diff.get(price, 0.) - volume
            # ask book
            else:
                limit_0 = max(self._dict)
                limit_1 = max(other._dict)
                limit = min(limit_0, limit_1)
                contain_limit = limit_0 == limit_1

                for entry in self._book:
                    price, volume, *_ = entry

                    if price < limit or (price <= limit and contain_limit):
                        diff[price] = volume

                for entry in other._book:
                    price, volume, *_ = entry

                    if price < limit or (price <= limit and contain_limit):
                        diff[price] = diff.get(price, 0.) - volume

            return diff

        def get(self, item=None, **kwargs) -> tuple[float, float, ...] | None:
            """
            Retrieve an entry by price or level, with flexibility for keyword arguments.

            Args:
                item (int | float, optional): The level (int) or price (float) to retrieve.
                **kwargs: Additional arguments for price or level.

            Returns:
                tuple[float, float, ...] | None: The entry at the specified price or level, or None if not found.

            Raises:
                ValueError: If both price and level are not provided or both are provided.
            """
            if item is None:
                price = kwargs.pop('price', None)
                level = kwargs.pop('level', None)
            else:
                if isinstance(item, int):
                    price = None
                    level = item
                elif isinstance(item, float):
                    price = item
                    level = None
                else:
                    raise ValueError(f'Invalid type {type(item)}, must be int or float')

            if price is None and level is None:
                raise ValueError('Must assign either price or level in kwargs')
            elif price is None:
                try:
                    return self.at_level(level=level)
                except IndexError:
                    return None
            elif level is None:
                try:
                    return self.at_price(price=price)
                except KeyError:
                    return None
            else:
                raise ValueError('Must NOT assign both price and level in kwargs')

        def pop(self, price: float):
            """
            Remove and return an entry at the specified price.

            Args:
                price (float): The price of the entry to remove.

            Returns:
                tuple[float, float, ...]: The removed entry.

            Raises:
                KeyError: If the price does not exist in the order book.
            """
            entry = self._dict.pop(price, None)
            if entry is not None:
                self._book.remove(entry)
            else:
                raise KeyError(f'Price {price} does not exist in the order book')
            return entry

        def remove(self, entry: tuple[float, float, ...]):
            """
            Remove a specific entry from the order book.

            Args:
                entry (tuple[float, float, ...]): The entry to remove.

            Raises:
                ValueError: If the entry does not exist in the order book.
            """
            try:
                self._book.remove(entry)
                self._dict.pop(entry[0])
            except ValueError:
                raise ValueError(f'Entry {entry} does not exist in the order book')

        def at_price(self, price: float):
            """
            Get the entry at a specific price.

            Args:
                price (float): The price to search for.

            Returns:
                tuple[float, float, ...]: The entry at the given price, or None if not found.
            """
            if price in self._dict:
                return self._dict.__getitem__(price)
            else:
                return None

        def at_level(self, level: int):
            """
            Get the entry at a specific level.

            Args:
                level (int): The level to search for.

            Returns:
                tuple[float, float, ...]: The entry at the given level.
            """
            return self._book.__getitem__(level)

        def update(self, price: float, volume: float, order: int = None):
            """
            Update or add an entry in the order book.

            Args:
                price (float): The price of the entry to update.
                volume (float): The new volume for the entry.
                order (int, optional): The order number. Defaults to None.

            Raises:
                ValueError: If the volume is invalid.
            """
            if price in self._dict:
                if volume == 0:
                    self.pop(price=price)
                elif volume < 0:
                    LOGGER.warning(f'Invalid volume {volume}, expect a positive float.')
                    self.pop(price=price)
                else:
                    entry = self._dict[price]
                    new_entry = list(entry)
                    new_entry[1] = volume
                    self._dict[price] = tuple(new_entry)
                    self._book[self._book.index(entry)] = tuple(new_entry)
            else:
                self.add(price=price, volume=volume, order=order)

        def add(self, price: float, volume: float, order: int = None):
            """
            Add a new entry to the order book.

            Args:
                price (float): The price of the new entry.
                volume (float): The volume of the new entry.
                order (int, optional): The order number. Defaults to None.
            """
            entry = (price, volume, order if order else 0)
            self._dict[price] = entry
            self._book.append(entry)

        def loc_volume(self, p0: float, p1: float) -> float:
            """
            Calculate the total volume between two price levels. Inclusive of the 2 given price.

            Args:
                p0 (float): The first price level.
                p1 (float): The second price level.

            Returns:
                float: The total volume between the two prices.
            """
            volume = 0.0
            p_min = min(p0, p1)
            p_max = max(p0, p1)

            for entry in self._book:
                price, vol, *_ = entry
                if p_min <= price <= p_max:
                    volume += vol

            return volume

        def sort(self):
            """
            Sort the order book by price in the appropriate order (descending for bids, ascending for asks).
            """
            if self.side > 0:  # bid
                self._book.sort(reverse=True, key=lambda x: x[0])
            else:  # ask
                self._book.sort(key=lambda x: x[0])
            self.sorted = True

        @property
        def price(self) -> list[float]:
            """
            Get a sorted list of all prices in the order book.

            Returns:
                list[float]: A list of all prices.
            """
            if not self.sorted:
                self.sort()

            return [entry[0] for entry in self._book]

        @property
        def volume(self) -> list[float]:
            """
            Get a sorted list of all volumes in the order book.

            Returns:
                list[float]: A list of all volumes.
            """
            if not self.sorted:
                self.sort()

            return [entry[1] for entry in self._book]

    def __init__(self, *, ticker: str, timestamp: float, bid: list[list[float | int]] = None, ask: list[list[float | int]] = None, **kwargs):
        """
        Initialize an OrderBook instance with market data.

        Args:
            ticker (str): The ticker symbol of the financial instrument.
            timestamp (float): The timestamp of the market data.
            bid (list[list[float | int]], optional): A list of bid data, where each sublist contains price, volume, and optionally, order numbers. Defaults to None.
            ask (list[list[float | int]], optional): A list of ask data. Defaults to None.
            **kwargs: Additional key-value pairs for parsing extra data fields.
        """

        if 'buffer' in kwargs:
            super().__init__(buffer=kwargs['buffer'])
            return

        buffer_constructor = _BUFFER_CONSTRUCTOR.new_orderbook_buffer()
        book_size = buffer_constructor.book_size

        bid_price, bid_volume, *bid_n_orders = zip(*bid)
        ask_price, ask_volume, *ask_n_orders = zip(*ask)

        buffer = buffer_constructor(
            ticker=ticker.encode('utf-8'),
            timestamp=timestamp
        )

        super().__init__(buffer=buffer, **kwargs)

        if bid_price:
            buffer.bid_price = (ctypes.c_double * book_size)(*bid_price)

        if bid_volume:
            buffer.bid_volume = (ctypes.c_double * book_size)(*bid_volume)

        if bid_n_orders:
            buffer.bid_n_orders = (ctypes.c_uint * book_size)(*bid_n_orders)

        if ask_price:
            buffer.ask_price = (ctypes.c_double * book_size)(*ask_price)

        if ask_volume:
            buffer.ask_volume = (ctypes.c_double * book_size)(*ask_volume)

        if ask_n_orders:
            buffer.ask_n_orders = (ctypes.c_uint * book_size)(*ask_n_orders)

        self.parse(**kwargs)

    def __getattr__(self, item: str):
        """
        Dynamically retrieve attributes like bid_price_X or ask_volume_Y.

        Args:
            item (str): The name of the attribute to retrieve.

        Returns:
            The value of the requested attribute.

        Raises:
            AttributeError: If the attribute is not found or the query level exceeds the maximum level.
        """
        if re.match('^((bid_)|(ask_))((price_)|(volume_))[0-9]+$', item):
            side, key, level, *_ = item.split('_')
            level = int(level)
            book: OrderBook.Book = self.__getattribute__(f'{side}')
            if 0 < level <= len(book):
                return book[level - 1].__getattribute__(key)
            else:
                raise AttributeError(f'query level [{level}] exceed max level [{len(book)}]')
        else:
            raise AttributeError(f'{item} not found in {self.__class__}')

    def __setattr__(self, key, value):
        """
        Dynamically set attributes like bid_price_X or ask_volume_Y.

        Args:
            key (str): The name of the attribute to set.
            value: The value to set for the attribute.
        """
        if re.match('^((bid_)|(ask_))((price_)|(volume_))[0-9]+$', key):
            self.update({key: value})
        else:
            super().__setattr__(key, value)

    def __repr__(self):
        """
        String representation of the OrderBook instance.

        Returns:
            str: A string describing the OrderBook.
        """
        return f'<OrderBook>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker}, bid={self.best_bid_price}, ask={self.best_ask_price})'

    def __str__(self):
        """
        String representation for print output.

        Returns:
            str: A detailed string representation of the OrderBook.
        """
        return f'<OrderBook>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker} {{Bid: {self.best_bid_price, self.best_bid_volume}, Ask: {self.best_ask_price, self.best_ask_volume}, Level: {self.max_level}}})'

    def __bool__(self):
        """
        Boolean value of the OrderBook instance.

        Returns:
            bool: True if both bid and ask sides have entries, False otherwise.
        """
        return bool(self.bid) and bool(self.ask)

    @classmethod
    def _parse_entry_name(cls, name: str, validate: bool = False) -> tuple[str, str, int]:
        """
        Parse an entry name like bid_price_X into its components.

        Args:
            name (str): The entry name to parse.
            validate (bool, optional): Whether to validate the entry name. Defaults to False.

        Returns:
            tuple[str, str, int]: The parsed side, key, and level.

        Raises:
            ValueError: If validation fails and the name is not parsable.
        """
        if validate:
            if not re.match('^((bid_)|(ask_))((price_)|(volume_)|(order_))[0-9]+$', name):
                raise ValueError(f'Cannot parse kwargs {name}.')

        side, key, level = name.split('_')
        level = int(level)

        return side, key, level

    @overload
    def parse(self, data: dict[str, float] = None, /, bid_price_1: float = math.nan, bid_volume_1: float = math.nan, ask_price_1: float = math.nan, ask_volume_1: float = math.nan, **kwargs: float):
        ...

    def parse(self, data: dict[str, float] = None, validate: bool = False, **kwargs):
        """
        Parse bid and ask data into the OrderBook.

        Args:
            data (dict[str, float], optional): A dictionary of data entries to parse. Defaults to None.
            validate (bool, optional): Whether to validate the entry names. Defaults to False.
            **kwargs: Additional key-value pairs for parsing into the OrderBook.
        """
        if not data:
            data = {}

        data.update(kwargs)

        for name, value in data.items():
            side, key, level = self._parse_entry_name(name, validate)
            self._buffer.__getattribute__(f'{side}_{key}')[level - 1] = value

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        """
        Create an OrderBook instance from a JSON message.

        Args:
            json_message (str | bytes | bytearray | dict): The JSON message to parse.

        Returns:
            OrderBook: An instance of the OrderBook class.

        Raises:
            TypeError: If the dtype in the JSON message does not match the class name.
        """
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        dtype = json_dict.pop('dtype', None)
        if dtype is not None and dtype != cls.__name__:
            raise TypeError(f'dtype mismatch, expect {cls.__name__}, got {dtype}.')

        bid_price = json_dict.pop('bid_price', [])
        bid_volume = json_dict.pop('bid_volume', [])
        bid_n_orders = json_dict.pop('bid_n_orders', [])
        ask_price = json_dict.pop('ask_price', [])
        ask_volume = json_dict.pop('ask_volume', [])
        ask_n_orders = json_dict.pop('ask_n_orders', [])

        self = cls(**json_dict)
        book_size = self._buffer.book_size

        if bid_price:
            self._buffer.bid_price = (ctypes.c_double * book_size)(*bid_price)

        if bid_volume:
            self._buffer.bid_volume = (ctypes.c_double * book_size)(*bid_volume)

        if bid_n_orders:
            self._buffer.bid_n_orders = (ctypes.c_uint * book_size)(*bid_n_orders)

        if ask_price:
            self._buffer.ask_price = (ctypes.c_double * book_size)(*ask_price)

        if ask_volume:
            self._buffer.ask_volume = (ctypes.c_double * book_size)(*ask_volume)

        if ask_n_orders:
            self._buffer.ask_n_orders = (ctypes.c_uint * book_size)(*ask_n_orders)

        return self

    @classmethod
    def from_buffer(cls, buffer: ctypes.Structure) -> Self:
        self = cls(
            ticker='',
            timestamp=0,
            buffer=buffer
        )
        return self

    @property
    def market_price(self):
        """
        Get the mid price of the order book.

        Returns:
            float: The mid price, or NaN if not available.
        """
        return self.mid_price

    @property
    def mid_price(self):
        """
        Calculate the mid price of the order book.

        Returns:
            float: The mid price, or NaN if not available.
        """
        if math.isfinite(self.best_bid_price) and math.isfinite(self.best_ask_price):
            return (self.best_bid_price + self.best_ask_price) / 2
        else:
            return math.nan

    @property
    def spread(self):
        """
        Calculate the bid-ask spread.

        Returns:
            float: The spread, or NaN if not available.
        """
        if math.isfinite(self.best_bid_price) and math.isfinite(self.best_ask_price):
            return self.best_ask_price - self.best_bid_price
        else:
            return math.nan

    @property
    def spread_pct(self):
        """
        Calculate the bid-ask spread as a percentage of the mid price.

        Returns:
            float: The spread percentage, or infinity if mid price is zero.
        """
        if self.mid_price != 0:
            return self.spread / self.mid_price
        else:
            return np.inf

    @property
    def bid(self) -> Book:
        """
        Get the bid side of the order book.

        Returns:
            Book: An instance of the Book class representing the bid side.
        """
        book = self.Book(side=1)
        for price, volume, n_orders in zip(self._buffer.bid_price, self._buffer.bid_volume, self._buffer.bid_n_orders):
            if not volume:
                continue
            book.add(price=price, volume=volume, order=n_orders)
        book.sort()
        return book

    @property
    def ask(self) -> Book:
        """
        Get the ask side of the order book.

        Returns:
            Book: An instance of the Book class representing the ask side.
        """
        book = self.Book(side=-1)
        for price, volume, n_orders in zip(self._buffer.ask_price, self._buffer.ask_volume, self._buffer.ask_n_orders):
            if not volume:
                continue
            book.add(price=price, volume=volume, order=n_orders)
        book.sort()
        return book

    @property
    def best_bid_price(self):
        """
        Get the best bid price in the order book.

        Returns:
            float: The best bid price, or NaN if not available.
        """
        if book := self.bid:
            return book.at_level(0)[0]
        else:
            return math.nan

    @property
    def best_ask_price(self):
        """
        Get the best ask price in the order book.

        Returns:
            float: The best ask price, or NaN if not available.
        """
        if book := self.ask:
            return book.at_level(0)[0]
        else:
            return math.nan

    @property
    def best_bid_volume(self):
        """
        Get the best bid volume in the order book.

        Returns:
            float: The best bid volume, or NaN if not available.
        """
        if book := self.bid:
            return book.at_level(0)[1]
        else:
            return math.nan

    @property
    def best_ask_volume(self):
        """
        Get the best ask volume in the order book.

        Returns:
            float: The best ask volume, or NaN if not available.
        """
        if book := self.ask:
            return book.at_level(0)[1]
        else:
            return math.nan


class BarData(MarketData):
    """
    Represents a single bar of market data for a specific ticker within a given time frame.

    This class extends the `MarketData` class and includes attributes and methods relevant to a market bar,
    such as price, volume, and duration. It also provides functionality for data serialization and validation.

    Methods:
        from_json(json_message: str | bytes | bytearray | dict) -> BarData:
            Creates a `BarData` instance from a JSON-encoded message or dictionary.

        to_list() -> list[float | int | str | bool]:
            Converts the `BarData` instance to a list of its attributes.

        from_list(data_list: list[float | int | str | bool]) -> BarData:
            Creates a `BarData` instance from a list of attributes.

        is_valid(verbose=False) -> bool:
            Validates the `BarData` instance to ensure all required fields are set correctly.

    Properties:
        high_price (float): The highest price during the bar.
        low_price (float): The lowest price during the bar.
        open_price (float): The opening price of the bar.
        close_price (float): The closing price of the bar.
        bar_span (datetime.timedelta): The duration of the bar.
        volume (float): The total volume of trades during the bar.
        notional (float): The total notional value of trades during the bar.
        trade_count (int): The number of trades that occurred during the bar.
        bar_start_time (datetime.datetime): The start time of the bar.
        vwap (float): The volume-weighted average price for the bar.
        market_price (float): The closing price of the bar.
        bar_type (Literal['Hourly-Plus', 'Hourly', 'Minute-Plus', 'Minute', 'Sub-Minute']): The type of the bar based on its span.
        bar_end_time (datetime.datetime | datetime.date): The end time of the bar.
    """

    def __init__(
            self, *,
            ticker: str,
            timestamp: float,  # The bar end timestamp
            start_timestamp: float = None,
            bar_span: datetime.timedelta | int | float = None,
            high_price: float = math.nan,
            low_price: float = math.nan,
            open_price: float = math.nan,
            close_price: float = math.nan,
            volume: float = 0.,
            notional: float = 0.,
            trade_count: int = 0,
            **kwargs
    ):
        """
        Initializes a new instance of `BarData`.

        Args:
            ticker (str): The ticker symbol for the market data.
            timestamp (float): The timestamp marking the end of the bar.
            start_timestamp (float, optional): The timestamp marking the start of the bar. Required if `bar_span` is not provided.
            bar_span (datetime.timedelta | int | float, optional): The duration of the bar. Either this or `start_timestamp` must be provided.
            high_price (float, optional): The highest price during the bar. Defaults to NaN.
            low_price (float, optional): The lowest price during the bar. Defaults to NaN.
            open_price (float, optional): The opening price of the bar. Defaults to NaN.
            close_price (float, optional): The closing price of the bar. Defaults to NaN.
            volume (float, optional): The total volume of trades during the bar. Defaults to 0.0.
            notional (float, optional): The total notional value of trades during the bar. Defaults to 0.0.
            trade_count (int, optional): The number of trades that occurred during the bar. Defaults to 0.
            **kwargs: Additional keyword arguments passed to the parent `MarketData` class.

        Raises:
            ValueError: If neither `start_timestamp` nor `bar_span` is provided or if `bar_span` is of invalid type.
        """

        if 'buffer' in kwargs:
            super().__init__(buffer=kwargs['buffer'])
            return

        buffer_constructor = _BUFFER_CONSTRUCTOR.new_candlestick_buffer()

        buffer = buffer_constructor(
            ticker=ticker.encode('utf-8'),
            timestamp=timestamp,
            high_price=high_price,
            low_price=low_price,
            open_price=open_price,
            close_price=close_price,
            volume=volume,
            notional=notional,
            trade_count=trade_count,
        )

        super().__init__(buffer=buffer, **kwargs)

        if bar_span is None and start_timestamp is None:
            raise ValueError('Must assign either start_timestamp or bar_span or both.')
        elif start_timestamp is None:
            # self['start_timestamp'] = timestamp - bar_span.total_seconds()
            if isinstance(bar_span, datetime.timedelta):
                buffer.bar_span = bar_span.total_seconds()
            elif isinstance(bar_span, (int, float)):
                buffer.bar_span = bar_span
            else:
                raise ValueError(f'Invalid bar_span {bar_span}! Expected a int, float or timedelta!')
        elif bar_span is None:
            self.start_timestamp = start_timestamp
        else:
            self.start_timestamp = start_timestamp

            if isinstance(bar_span, datetime.timedelta):
                buffer.bar_span = bar_span.total_seconds()
            elif isinstance(bar_span, (int, float)):
                buffer.bar_span = bar_span
            else:
                raise ValueError(f'Invalid bar_span {bar_span}! Expected a int, float or timedelta!')

    def __repr__(self):
        """
        Returns a string representation of the `BarData` instance.

        The string representation includes the class name, market time, ticker symbol, and key price attributes.

        Returns:
            str: A string representation of the `BarData` instance.
        """
        return f'<{self.__class__.__name__}>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker}, open={self.open_price}, close={self.close_price}, high={self.high_price}, low={self.low_price})'

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        """
        Creates a `BarData` instance from a JSON-encoded message or dictionary.

        Args:
            json_message (str | bytes | bytearray | dict): The JSON-encoded message or dictionary containing `BarData` attributes.

        Returns:
            BarData: A `BarData` instance initialized with the data from the JSON message.

        Raises:
            TypeError: If the dtype in the JSON does not match the class name.
        """
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        dtype = json_dict.pop('dtype', None)
        if dtype is not None and dtype != cls.__name__:
            raise TypeError(f'dtype mismatch, expect {cls.__name__}, got {dtype}.')

        self = cls(**json_dict)
        return self

    @classmethod
    def from_buffer(cls, buffer: ctypes.Structure | memoryview) -> Self:
        self = cls(
            ticker='',
            timestamp=0,
            buffer=buffer
        )
        return self

    @property
    def high_price(self) -> float:
        """
        The highest price during the bar.

        Returns:
            float: The highest price during the bar.
        """
        return self._buffer.high_price

    @property
    def low_price(self) -> float:
        """
        The lowest price during the bar.

        Returns:
            float: The lowest price during the bar.
        """
        return self._buffer.low_price

    @property
    def open_price(self) -> float:
        """
        The opening price of the bar.

        Returns:
            float: The opening price of the bar.
        """
        return self._buffer.open_price

    @property
    def close_price(self) -> float:
        """
        The closing price of the bar.

        Returns:
            float: The closing price of the bar.
        """
        return self._buffer.close_price

    @property
    def bar_span(self) -> datetime.timedelta:
        """
        The duration of the bar.

        Returns:
            datetime.timedelta: The duration of the bar.
        """

        if bar_span := self._buffer.bar_span:
            return datetime.timedelta(seconds=bar_span)
        else:
            return datetime.timedelta(seconds=self._buffer.timestamp - self._buffer.start_timestamp)

    @property
    def volume(self) -> float:
        """
        The total volume of trades during the bar.

        Returns:
            float: The total volume of trades during the bar.
        """
        return self._buffer.volume

    @property
    def notional(self) -> float:
        """
        The total notional value of trades during the bar.

        Returns:
            float: The total notional value of trades during the bar.
        """
        return self._buffer.notional

    @property
    def trade_count(self) -> int:
        """
        The number of trades that occurred during the bar.

        Returns:
            int: The number of trades that occurred during the bar.
        """
        return self._buffer.trade_count

    @property
    def bar_start_time(self) -> datetime.datetime:
        """
        The start time of the bar.

        Returns:
            datetime.datetime: The start time of the bar.
        """
        if start_timestamp := self._buffer.start_timestamp:
            return datetime.datetime.fromtimestamp(start_timestamp, tz=PROFILE.time_zone)
        else:
            return datetime.datetime.fromtimestamp(self._buffer.timestamp - self._buffer.bar_span, tz=PROFILE.time_zone)

    @property
    def vwap(self) -> float:
        """
        The volume-weighted average price for the bar.

        Returns:
            float: The VWAP for the bar. Defaults to the closing price if volume is zero.
        """
        if self.volume != 0:
            return self.notional / self.volume
        else:
            LOGGER.warning(f'[{self.market_time}] {self.ticker} Volume data not available, using close_price as default VWAP value')
            return self.close_price

    @property
    def is_valid(self, verbose=False) -> bool:
        """
        Validates the `BarData` instance to ensure all required fields are set correctly.

        Args:
            verbose (bool, optional): If True, logs detailed validation errors. Defaults to False.

        Returns:
            bool: True if the `BarData` instance is valid, False otherwise.
        """
        try:
            assert type(self.ticker) is str, f'{self} Invalid ticker'
            assert math.isfinite(self.high_price), f'{self} Invalid high_price'
            assert math.isfinite(self.low_price), f'{self} Invalid low_price'
            assert math.isfinite(self.open_price), f'{self} Invalid open_price'
            assert math.isfinite(self.close_price), f'{self} Invalid close_price'
            assert math.isfinite(self.volume), f'{self} Invalid volume'
            assert math.isfinite(self.notional), f'{self} Invalid notional'
            assert math.isfinite(self.trade_count), f'{self} Invalid trade_count'
            assert isinstance(self.bar_start_time, (datetime.datetime, datetime.date)), f'{self} Invalid bar_start_time'
            assert isinstance(self.bar_span, datetime.timedelta), f'{self} Invalid bar_span'

            return True
        except AssertionError as e:
            if verbose:
                LOGGER.warning(str(e))
            return False

    @property
    def market_price(self) -> float:
        """
        The closing price for the `BarData`.

        Returns:
            float: The closing price of the bar.
        """
        return self.close_price

    @property
    def bar_type(self) -> Literal['Hourly-Plus', 'Hourly', 'Minute-Plus', 'Minute', 'Sub-Minute']:
        """
        Determines the type of the bar based on its span.

        Returns:
            Literal['Hourly-Plus', 'Hourly', 'Minute-Plus', 'Minute', 'Sub-Minute']: The type of the bar.
        """
        bar_span = self.bar_span.total_seconds()

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


class DailyBar(BarData):
    """
    Represents a daily bar of market data for a specific ticker.

    This class extends the `BarData` class and focuses on daily bar data, which includes attributes and methods
    specific to daily market bars. It supports various ways to define the bar span and manage the market date.

    Attributes:
        ...

    Methods:
        __repr__() -> str:
            Returns a string representation of the `DailyBar` instance.

        to_json(fmt='str', **kwargs) -> str | dict:
            Converts the `DailyBar` instance to a JSON string or dictionary.

        to_list() -> list[float | int | str | bool]:
            Converts the `DailyBar` instance to a list of its attributes.

        from_list(data_list: list[float | int | str | bool]) -> DailyBar:
            Creates a `DailyBar` instance from a list of attributes.

    Properties:
        ticker (str): The ticker symbol for the market data.
        timestamp (float): The timestamp marking the end of the bar.
        start_date (datetime.date, optional): The start date of the bar period. Required if `bar_span` is not provided.
        bar_span (datetime.timedelta | int, optional): The duration of the bar in days. Either this or `start_date` must be provided.
        high_price (float): The highest price during the bar.
        low_price (float): The lowest price during the bar.
        open_price (float): The opening price of the bar.
        close_price (float): The closing price of the bar.
        volume (float): The total volume of trades during the bar.
        notional (float): The total notional value of trades during the bar.
        trade_count (int): The number of trades that occurred during the bar.
        bar_span (datetime.timedelta): The duration of the bar in days.

        market_date (datetime.date): The market date of the bar.
        market_time (datetime.date): The market date of the bar (same as `market_date`).
        bar_start_time (datetime.date): The start date of the bar period.
        bar_end_time (datetime.date): The end date of the bar period.
        bar_type (Literal['Daily', 'Daily-Plus']): The type of the bar based on its span.
    """

    def __init__(
            self, *,
            ticker: str,
            market_date: datetime.date | str,  # The market date of the bar, if with 1D data, or the END date of the bar.
            timestamp: float = None,
            start_date: datetime.date = None,
            bar_span: datetime.timedelta | int = None,  # Expect to be a timedelta for several days, or the number of days
            high_price: float = math.nan,
            low_price: float = math.nan,
            open_price: float = math.nan,
            close_price: float = math.nan,
            volume: float = 0.,
            notional: float = 0.,
            trade_count: int = 0,
            **kwargs
    ):
        """
        Initializes a new instance of `DailyBar`.

        Args:
            ticker (str): The ticker symbol for the market data.
            market_date (datetime.date | str): The market date of the bar or the end date of the bar.
            timestamp (float, optional): repurposed to marking the end of the bar. Defaults to None.
            start_date (datetime.date, optional): The start date of the bar period. Required if `bar_span` is not provided.
            bar_span (datetime.timedelta | int, optional): The duration of the bar in days. Either this or `start_date` must be provided.
            high_price (float, optional): The highest price during the bar. Defaults to NaN.
            low_price (float, optional): The lowest price during the bar. Defaults to NaN.
            open_price (float, optional): The opening price of the bar. Defaults to NaN.
            close_price (float, optional): The closing price of the bar. Defaults to NaN.
            volume (float, optional): The total volume of trades during the bar. Defaults to 0.0.
            notional (float, optional): The total notional value of trades during the bar. Defaults to 0.0.
            trade_count (int, optional): The number of trades that occurred during the bar. Defaults to 0.
            **kwargs: Additional keyword arguments passed to the parent `BarData` class.

        Raises:
            ValueError: If neither `start_date` nor `bar_span` is provided or if `bar_span` is of invalid type.
        """

        if 'buffer' in kwargs:
            super().__init__(ticker='', timestamp=0, buffer=kwargs['buffer'])
            return

        if isinstance(market_date, str):
            market_date = datetime.date.fromisoformat(market_date)

        if bar_span is None and start_date is None:
            raise ValueError('Must assign either datetime.date or bar_span or both.')
        elif start_date is None:
            if isinstance(bar_span, datetime.timedelta):
                bar_span = bar_span.days
            elif isinstance(bar_span, int):
                pass
            else:
                raise ValueError(f'Invalid bar_span, expect int, float or timedelta, got {bar_span}')
        elif bar_span is None:
            bar_span = (market_date - start_date).days
        else:
            assert (market_date - start_date).days == bar_span.days

        if timestamp is not None:
            LOGGER.warning(f'Timestamp of {self.__class__.__name__} should not be provided.')

        timestamp = 10000 * market_date.year + 100 * market_date.month + market_date.day

        super().__init__(
            ticker=ticker,
            timestamp=timestamp,
            bar_span=bar_span,
            high_price=high_price,
            low_price=low_price,
            open_price=open_price,
            close_price=close_price,
            volume=volume,
            notional=notional,
            trade_count=trade_count,
            **kwargs
        )

    def __repr__(self) -> str:
        """
        Returns a string representation of the `DailyBar` instance.

        The string representation includes the class name, market date, ticker symbol, and key price attributes.

        Returns:
            str: A string representation of the `DailyBar` instance.
        """
        return f'<{self.__class__.__name__}>([{self.market_time:%Y-%m-%d}] {self.ticker}, open={self.open_price}, close={self.close_price}, high={self.high_price}, low={self.low_price})'

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        """
        Converts the `DailyBar` instance to a JSON string or dictionary.

        Args:
            fmt (str, optional): The format for the JSON output. Either 'dict' or 'str'. Defaults to 'str'.
            **kwargs: Additional keyword arguments passed to `json.dumps()` if `fmt='str'`.

        Returns:
            str | dict: The JSON-encoded representation of the `DailyBar` instance, in the specified format.

        Raises:
            ValueError: If an invalid format is specified.
        """
        data_dict = super().to_json(fmt='dict', **kwargs)
        data_dict['market_date'] = self.market_date.isoformat()

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, expected "dict" or "str".')

    @classmethod
    def from_list(cls, data_list: list[float | int | str | bool]) -> Self:
        """
        Creates a `DailyBar` instance from a list of attributes.

        Args:
            data_list (list[float | int | str | bool]): A list of attributes representing a `DailyBar` instance.

        Returns:
            DailyBar: A `DailyBar` instance initialized with the data from the list.

        Raises:
            TypeError: If the dtype in the list does not match the class name.
        """
        (dtype, ticker, market_date, timestamp, high_price, low_price, open_price, close_price,
         bar_span, volume, notional, trade_count) = data_list

        if dtype != cls.__name__:
            raise TypeError(f'dtype mismatch, expect {cls.__name__}, got {dtype}.')

        return cls(
            ticker=ticker,
            market_date=market_date,
            timestamp=timestamp,
            high_price=high_price,
            low_price=low_price,
            open_price=open_price,
            close_price=close_price,
            bar_span=datetime.timedelta(days=bar_span) if bar_span else None,
            volume=volume,
            notional=notional,
            trade_count=trade_count
        )

    @property
    def bar_span(self) -> datetime.timedelta:
        """
        The duration of the bar in days.

        Returns:
            datetime.timedelta: The duration of the bar.
        """
        return datetime.timedelta(days=self._buffer.bar_span)

    @property
    def market_date(self) -> datetime.date:
        """
        The market date of the bar.

        Returns:
            datetime.date: The market date of the bar.
        """

        int_date = int(self._buffer.timestamp)
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
    def bar_type(self) -> Literal['Daily', 'Daily-Plus']:
        """
        Determines the type of the bar based on its span.

        Returns:
            Literal['Daily', 'Daily-Plus']: The type of the bar.

        Raises:
            ValueError: If `bar_span` is not valid for a daily bar.
        """
        if self._buffer.bar_span == 1:
            return 'Daily'
        elif self._buffer.bar_span > 1:
            return 'Daily-Plus'
        else:
            raise ValueError(f'Invalid bar_span for {self.__class__.__name__}! Expect an int greater or equal to 1, got {self._buffer.bar_span}')


class TickData(MarketData):
    """
    Represents tick data for a specific ticker.

    This class extends the `MarketData` class and focuses on tick-level market data, including last price, bid/ask prices,
    bid/ask volumes, and order book details.

    Attributes:
        ...

    Methods:
        __repr__() -> str:
            Returns a string representation of the `TickData` instance.

        from_json(json_message: str | bytes | bytearray | dict) -> TickData:
            Creates a `TickData` instance from a JSON message.

        to_list() -> list[float | int | str | bool]:
            Converts the `TickData` instance to a list of its attributes, excluding order book information.

        from_list(data_list: list[float | int | str | bool]) -> TickData:
            Creates a `TickData` instance from a list of attributes.

    Properties:
    ticker (str): The ticker symbol for the market data.
        timestamp (float): The timestamp of the tick data.
        bid (list[list[float | int]] | None): A list of bid prices and volumes. Optional, used to build the order book.
        ask (list[list[float | int]] | None): A list of ask prices and volumes. Optional, used to build the order book.
        level_2 (OrderBook | None): The level 2 order book created from the bid and ask data.
        order_book (OrderBook | None): Alias for `level_2`.
        last_price (float): The last traded price.
        bid_price (float | None): The bid price.
        ask_price (float | None): The ask price.
        bid_volume (float | None): The bid volume.
        ask_volume (float | None): The ask volume.
        total_traded_volume (float): The total traded volume.
        total_traded_notional (float): The total traded notional value.
        total_trade_count (float): The total number of trades.
        mid_price (float): The midpoint price calculated as the average of bid and ask prices.
        market_price (float): The last traded price.
    """

    def __init__(
            self, *,
            ticker: str,
            timestamp: float,
            last_price: float,
            bid_price: float = None,
            bid_volume: float = None,
            ask_price: float = None,
            ask_volume: float = None,
            order_book: OrderBook = None,
            bid: Iterable[list[float | int]] = None,
            ask: Iterable[list[float | int]] = None,
            total_traded_volume: float = 0.,
            total_traded_notional: float = 0.,
            total_trade_count: int = 0,
            **kwargs
    ):
        """
        Initializes a new instance of `TickData`.

        Args:
            ticker (str): The ticker symbol for the market data.
            timestamp (float): The timestamp of the tick data.
            last_price (float): The last traded price.
            bid_price (float, optional): The bid price. Defaults to None.
            bid_volume (float, optional): The bid volume. Defaults to None.
            ask_price (float, optional): The ask price. Defaults to None.
            ask_volume (float, optional): The ask volume. Defaults to None.
            order_book (OrderBook, optional): The order book containing bid and ask data. Defaults to None.
            bid (Iterable[list[float | int]], optional): A list of bid prices and volumes. Defaults to None.
            ask (Iterable[list[float | int]], optional): A list of ask prices and volumes. Defaults to None.
            total_traded_volume (float, optional): The total traded volume. Defaults to 0.0.
            total_traded_notional (float, optional): The total traded notional value. Defaults to 0.0.
            total_trade_count (int, optional): The total number of trades. Defaults to 0.
            **kwargs: Additional keyword arguments passed to the parent `MarketData` class.
        """

        if 'buffer' in kwargs:
            super().__init__(buffer=kwargs['buffer'])
            return

        buffer_constructor = _BUFFER_CONSTRUCTOR.new_tick_buffer()

        buffer = buffer_constructor(
            ticker=ticker.encode('utf-8'),
            timestamp=timestamp,
            last_price=last_price,
            bid_price=np.nan if bid_price is None else bid_price,
            bid_volume=np.nan if bid_volume is None else bid_volume,
            ask_price=np.nan if ask_price is None else ask_price,
            ask_volume=np.nan if ask_volume is None else ask_volume,
            total_traded_volume=total_traded_volume,
            total_traded_notional=total_traded_notional,
            total_trade_count=total_trade_count
        )

        super().__init__(buffer=buffer, **kwargs)

        if order_book is not None:
            self._order_book = order_book
            buffer.order_book = self._order_book._buffer
        elif bid and ask:
            self._order_book = OrderBook(ticker=ticker, timestamp=timestamp, bid=bid, ask=ask)
            buffer.order_book = self._order_book._buffer

        if hasattr(self, '_order_book'):
            buffer.bid_price = self._order_book.best_bid_price
            buffer.bid_volume = self._order_book.best_bid_volume
            buffer.ask_price = self._order_book.best_ask_price
            buffer.ask_volume = self._order_book.best_ask_volume

    def __repr__(self) -> str:
        """
        Returns a string representation of the `TickData` instance.

        The string representation includes the class name, market time, ticker symbol, and bid/ask prices.

        Returns:
            str: A string representation of the `TickData` instance.
        """
        return f'<TickData>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker}, bid={self.bid_price}, ask={self.ask_price})'

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict', **kwargs)

        if hasattr(self, '_order_book'):
            data_dict['order_book'] = self._order_book.to_json(fmt='dict')

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, expected "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        """
        Creates a `TickData` instance from a JSON message.

        Args:
            json_message (str | bytes | bytearray | dict): The JSON message containing tick data.

        Returns:
            TickData: A `TickData` instance.

        Raises:
            TypeError: If the JSON message does not match the expected data type.
        """
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        dtype = json_dict.pop('dtype', None)
        if dtype is not None and dtype != cls.__name__:
            raise TypeError(f'dtype mismatch, expect {cls.__name__}, got {dtype}.')

        if 'order_book' in json_dict:
            json_dict['order_book'] = OrderBook.from_json(json_dict.pop('order_book'))

        self = cls(**json_dict)

        return self

    @classmethod
    def from_buffer(cls, buffer: ctypes.Structure | memoryview) -> Self:
        self = cls(
            ticker='',
            timestamp=0,
            last_price=np.nan,
            buffer=buffer
        )
        return self

    @property
    def fields(self) -> Iterable[str]:
        return tuple(name for name, *_ in getattr(self._buffer, '_fields_') if name != 'order_book')

    @property
    def level_2(self) -> OrderBook | None:
        """
        The level 2 order book created from the bid and ask data.

        Returns:
            OrderBook | None: The `OrderBook` instance if available, otherwise `None`.
        """

        if hasattr(self, '_order_book'):
            return self._order_book
        elif order_book_buffer := self._buffer.order_book:
            self._order_book = OrderBook.from_buffer(buffer=order_book_buffer)
            return self._order_book
        else:
            return None

    @property
    def order_book(self) -> OrderBook | None:
        """
        Alias for `level_2`.

        Returns:
            OrderBook | None: The `OrderBook` instance if available, otherwise `None`.
        """
        return self.level_2

    @property
    def last_price(self) -> float:
        """
        The last traded price.

        Returns:
            float: The last traded price.
        """
        return self._buffer.last_price

    @property
    def bid_price(self) -> float | None:
        """
        The bid price.

        Returns:
            float | None: The bid price if available, otherwise `None`.
        """
        return self._buffer.bid_price

    @property
    def ask_price(self) -> float | None:
        """
        The ask price.

        Returns:
            float | None: The ask price if available, otherwise `None`.
        """
        return self._buffer.ask_price

    @property
    def bid_volume(self) -> float | None:
        """
        The bid volume.

        Returns:
            float | None: The bid volume if available, otherwise `None`.
        """
        return self._buffer.bid_volume

    @property
    def ask_volume(self) -> float | None:
        """
        The ask volume.

        Returns:
            float | None: The ask volume if available, otherwise `None`.
        """
        return self._buffer.ask_volume

    @property
    def total_traded_volume(self) -> float:
        """
        The total traded volume.

        Returns:
            float: The total traded volume.
        """
        return self._buffer.total_traded_volume

    @property
    def total_traded_notional(self) -> float:
        """
        The total traded notional value.

        Returns:
            float: The total traded notional value.
        """
        return self._buffer.total_traded_notional

    @property
    def total_trade_count(self) -> float:
        """
        The total number of trades.

        Returns:
            float: The total number of trades.
        """
        return self._buffer.total_trade_count

    @property
    def mid_price(self) -> float:
        """
        The midpoint price calculated as the average of bid and ask prices.

        Returns:
            float: The midpoint price.
        """
        return (self.bid_price + self.ask_price) / 2

    @property
    def market_price(self) -> float:
        """
        The last traded price.

        Returns:
            float: The last traded price.
        """
        return self.last_price


class TransactionData(MarketData):
    """
    Represents transaction data for a specific market.

    This class extends the `MarketData` class to handle transaction-level data, including price, volume, side, and identifiers.

    Attributes:
        ...

    Methods:
        __repr__() -> str:
            Returns a string representation of the `TransactionData` instance.

        from_json(json_message: str | bytes | bytearray | dict) -> TransactionData:
            Creates a `TransactionData` instance from a JSON message.

        to_list() -> list[float | int | str | bool]:
            Converts the `TransactionData` instance to a list of its attributes.

        from_list(data_list: list[float | int | str | bool]) -> TransactionData:
            Creates a `TransactionData` instance from a list of attributes.

        merge(trade_data_list: list[TransactionData]) -> TransactionData | None:
            Merges multiple `TransactionData` instances into a single aggregated `TransactionData` instance.

    Properties:
        ticker (str): The ticker symbol for the transaction.
        timestamp (float): The timestamp of the transaction.

        price (float): The price at which the transaction occurred.
        volume (float): The volume of the transaction.
        side (TransactionSide): The side of the transaction (buy or sell).
        multiplier (float): The multiplier for the transaction.
        transaction_id (int | str | None): The identifier for the transaction.
        buy_id (int | str | None): The identifier for the buying transaction.
        sell_id (int | str | None): The identifier for the selling transaction.
        notional (float): The notional value of the transaction.
        market_price (float): Alias for `price`.
        flow (float): The flow of the transaction, calculated as side.sign * volume.
    """

    def __init__(
            self, *,
            ticker: str,
            price: float,
            volume: float,
            timestamp: float,
            side: int | float | str | TransactionSide = 0,
            multiplier: float = None,
            notional: float = None,
            transaction_id: str | int = None,
            buy_id: str | int = None,
            sell_id: str | int = None,
            **kwargs
    ):
        """
        Initializes a new instance of `TransactionData`.

        Args:
            ticker (str): The ticker symbol for the transaction.
            price (float): The price at which the transaction occurred.
            volume (float): The volume of the transaction.
            timestamp (float): The timestamp of the transaction.
            side (int | float | str | TransactionSide, optional): The side of the transaction (buy or sell). Defaults to 0.
            multiplier (float, optional): The multiplier for the transaction. Defaults to None.
            notional (float, optional): The notional value of the transaction. Defaults to None.
            transaction_id (str | int, optional): The identifier for the transaction. Defaults to None.
            buy_id (str | int, optional): The identifier for the buying transaction. Defaults to None.
            sell_id (str | int, optional): The identifier for the selling transaction. Defaults to None.
            **kwargs: Additional keyword arguments passed to the parent `MarketData` class.
        """

        if 'buffer' in kwargs:
            super().__init__(buffer=kwargs['buffer'])
            return

        buffer_constructor = _BUFFER_CONSTRUCTOR.new_transaction_buffer()
        id_size = buffer_constructor.id_size

        buffer = buffer_constructor(
            ticker=ticker.encode('utf-8'),
            timestamp=timestamp,
            price=price,
            volume=volume,
            side=int(side) if isinstance(side, (int, float)) else TransactionSide(side).value,
            multiplier=np.nan if multiplier is None else multiplier,
            notional=np.nan if notional is None else notional
        )

        super().__init__(buffer=buffer, **kwargs)

        self._set_id(name='transaction_id', value=transaction_id, size=id_size)
        self._set_id(name='buy_id', value=buy_id, size=id_size)
        self._set_id(name='sell_id', value=sell_id, size=id_size)

    def __repr__(self) -> str:
        """
        Returns a string representation of the `TransactionData` instance.

        The string representation includes the class name, market time, side, ticker symbol, price, and volume.

        Returns:
            str: A string representation of the `TransactionData` instance.
        """
        return f'<TransactionData>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.side.side_name} {self.ticker}, price={self.price}, volume={self.volume})'

    def _set_id(self, name: Literal['transaction_id', 'buy_id', 'sell_id', 'order_id'], value: int | str, size: int):
        buffer = getattr(self._buffer, name)

        if isinstance(value, str):
            buffer.id_str.id_type = 0
            buffer.id_str.data = value.encode(self._encoding)
        elif isinstance(value, int):
            buffer.id_int.id_type = 1
            buffer.id_int.data[:] = value.to_bytes(length=size, byteorder='little')
        elif value is None:
            buffer.id_type = -1
        else:
            raise ValueError(f'Invalid id {value}. Expected str or int.')

    def _get_id(self, name: Literal['transaction_id', 'buy_id', 'sell_id', 'order_id']) -> int | str | None:
        buffer = getattr(self._buffer, name)

        match buffer.id_type:
            case 0:
                return buffer.id_str.data.decode(self._encoding)
            case 1:
                return int.from_bytes(buffer.id_int.data, byteorder='little')
            case -1:
                return None
            case _:
                raise ValueError(f'Invalid id type for {name}!')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        """
        Creates a `TransactionData` instance from a JSON message.

        Args:
            json_message (str | bytes | bytearray | dict): The JSON message containing transaction data.

        Returns:
            TransactionData: A `TransactionData` instance.

        Raises:
            TypeError: If the JSON message does not match the expected data type.
        """
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        dtype = json_dict.pop('dtype', None)
        if dtype is not None and dtype != cls.__name__:
            raise TypeError(f'dtype mismatch, expect {cls.__name__}, got {dtype}.')

        self = cls(**json_dict)
        return self

    @classmethod
    def from_buffer(cls, buffer: ctypes.Structure | memoryview) -> Self:
        self = cls(
            ticker='',
            timestamp=0,
            price=0,
            volume=0,
            buffer=buffer
        )
        return self

    @classmethod
    def merge(cls, trade_data_list: list[Self]) -> Self | None:
        """
        Merges multiple `TransactionData` instances into a single aggregated `TransactionData` instance.

        Args:
            trade_data_list (list[TransactionData]): A list of `TransactionData` instances to merge.

        Returns:
            TransactionData | None: A merged `TransactionData` instance if the list is not empty, otherwise `None`.

        Raises:
            AssertionError: If the list contains transaction data for multiple tickers.
        """
        if not trade_data_list:
            return None

        ticker = trade_data_list[0].ticker
        assert all([trade.ticker == ticker for trade in trade_data_list]), 'input contains trade data of multiple ticker'
        timestamp = max([trade.timestamp for trade in trade_data_list])
        sum_volume = sum([trade.volume * trade.side.sign for trade in trade_data_list])
        sum_notional = sum([trade.notional * trade.side.sign for trade in trade_data_list])
        trade_side_sign = np.sign(sum_volume) if sum_volume != 0 else 1

        if sum_notional == 0:
            trade_price = 0
        elif sum_volume == 0:
            trade_price = math.nan
        else:
            trade_price = sum_notional / sum_volume if sum_volume else math.inf * np.sign(sum_notional)

        trade_side = TransactionSide(trade_side_sign)
        trade_volume = abs(sum_volume)
        trade_notional = abs(sum_notional)

        merged_trade_data = cls(
            ticker=ticker,
            timestamp=timestamp,
            side=trade_side,
            price=trade_price,
            volume=trade_volume,
            notional=trade_notional
        )

        return merged_trade_data

    @property
    def price(self) -> float:
        """
        The price at which the transaction occurred.

        Returns:
            float: The transaction price.
        """
        return self._buffer.price

    @property
    def volume(self) -> float:
        """
        The volume of the transaction.

        Returns:
            float: The transaction volume.
        """
        return self._buffer.volume

    @property
    def side(self) -> TransactionSide:
        """
        The side of the transaction (buy or sell).

        Returns:
            TransactionSide: The side of the transaction.
        """
        return TransactionSide(self._buffer.side)

    @property
    def multiplier(self) -> float:
        """
        The multiplier for the transaction. Defaults to 1 if not specified.

        Returns:
            float: The transaction multiplier.
        """
        multiplier = self._buffer.multiplier

        if np.isnan(multiplier):
            multiplier = 1.

        return multiplier

    @property
    def transaction_id(self) -> int | str | None:
        """
        The identifier for the transaction.

        Returns:
            int | str | None: The transaction identifier.
        """
        return self._get_id(name='transaction_id')

    @property
    def buy_id(self) -> int | str | None:
        """
        The identifier for the buying transaction.

        Returns:
            int | str | None: The buying transaction identifier.
        """
        return self._get_id(name='buy_id')

    @property
    def sell_id(self) -> int | str | None:
        """
        The identifier for the selling transaction.

        Returns:
            int | str | None: The selling transaction identifier.
        """
        return self._get_id(name='sell_id')

    @property
    def notional(self) -> float:
        """
        The notional value of the transaction. Calculated as price * volume * multiplier.

        Returns:
            float: The transaction notional.
        """

        notional = self._buffer.notional

        if np.isnan(notional):
            return self.price * self.volume * self.multiplier

        return notional

    @property
    def market_price(self) -> float:
        """
        Alias for the transaction price.

        Returns:
            float: The transaction price.
        """
        return self.price

    @property
    def flow(self) -> float:
        """
        The flow of the transaction, calculated as side.sign * volume.

        Returns:
            float: The transaction flow.
        """
        return self.side.sign * self.volume


class TradeData(TransactionData):
    """
    Alias for `TransactionData` with alternate property names for trade price and volume.

    This class allows initialization with 'trade_price' instead of 'price' and 'trade_volume' instead of 'volume'.
    It provides additional properties for these alternate names.

    Properties:
        trade_price (float): Alias for `price`.
        trade_volume (float): Alias for `volume`.

    Methods:
        from_json(json_message: str | bytes | bytearray | dict) -> TradeData:
            Creates a `TradeData` instance from a JSON message.

        from_list(data_list: list[float | int | str | bool]) -> TradeData:
            Creates a `TradeData` instance from a list of attributes.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new instance of `TradeData`.

        Args:
            **kwargs: Keyword arguments passed to the parent `TransactionData` class.
                If 'trade_price' or 'trade_volume' are provided, they are converted to 'price' and 'volume'.
        """
        if 'trade_price' in kwargs:
            kwargs['price'] = kwargs.pop('trade_price')

        if 'trade_volume' in kwargs:
            kwargs['volume'] = kwargs.pop('trade_volume')

        super().__init__(**kwargs)

    @property
    def trade_price(self) -> float:
        """
        Alias for the transaction price.

        Returns:
            float: The transaction price.
        """
        return self._buffer.price

    @property
    def trade_volume(self) -> float:
        """
        Alias for the transaction volume.

        Returns:
            float: The transaction volume.
        """
        return self._buffer.volume

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        """
        Creates a `TradeData` instance from a JSON message.

        Args:
            json_message (str | bytes | bytearray | dict): The JSON message containing trade data.

        Returns:
            TradeData: A `TradeData` instance.

        Raises:
            TypeError: If the JSON message does not match the expected data type.
        """
        return super(TradeData, cls).from_json(json_message=json_message)

    @classmethod
    def from_buffer(cls, buffer: ctypes.Structure | memoryview) -> Self:
        return super(TradeData, cls).from_buffer(buffer=buffer)

    @classmethod
    def from_bytes(cls, data) -> Self:
        return super(TradeData, cls).from_bytes(data=data)


class OrderData(MarketData):
    def __init__(
            self, *,
            ticker: str,
            price: float,
            volume: float,
            timestamp: float,
            side: int | float | str | TransactionSide = 0,
            order_type: int = 0,
            order_id: str | int = None,
            **kwargs
    ):

        if 'buffer' in kwargs:
            super().__init__(buffer=kwargs['buffer'])
            return

        buffer_constructor = _BUFFER_CONSTRUCTOR.new_order_buffer()
        id_size = buffer_constructor.id_size

        buffer = buffer_constructor(
            ticker=ticker.encode('utf-8'),
            timestamp=timestamp,
            price=price,
            volume=volume,
            side=int(side) if isinstance(side, (int, float)) else TransactionSide(side).value,
            order_type=order_type
        )

        super().__init__(buffer=buffer, **kwargs)

        TransactionData._set_id(self=self, name='order_id', value=order_id, size=id_size)

    def __repr__(self) -> str:
        return f'<OrderData>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.side.side_name} {self.ticker}, price={self.price}, volume={self.volume})'

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        dtype = json_dict.pop('dtype', None)
        if dtype is not None and dtype != cls.__name__:
            raise TypeError(f'dtype mismatch, expect {cls.__name__}, got {dtype}.')

        self = cls(**json_dict)
        return self

    @classmethod
    def from_buffer(cls, buffer: ctypes.Structure | memoryview) -> Self:
        self = cls(
            ticker='',
            timestamp=0,
            price=0,
            volume=0,
            buffer=buffer
        )
        return self

    @property
    def price(self) -> float:
        return self._buffer.price

    @property
    def volume(self) -> float:
        return self._buffer.volume

    @property
    def side(self) -> TransactionSide:
        return TransactionSide(self._buffer.side)

    @property
    def OrderType(self) -> OrderType:
        return OrderType(self._buffer.order_type)

    @property
    def order_id(self) -> int | str | None:
        return TransactionData._get_id(self=self, name='order_id')

    @property
    def market_price(self) -> float:
        """
        Alias for the transaction price.

        Returns:
            float: The transaction price.
        """
        return self.price


class MarketDataBuffer(object):
    ctype_buffer = _BUFFER_CONSTRUCTOR.new_market_data_buffer()

    def __init__(self, buffer: type[ctypes.Structure | ctypes.Union | ctypes.Array] = None):
        self.buffer = RawValue(self.ctype_buffer) if buffer is None else buffer

    @classmethod
    def to_buffer(cls, buffer, market_data: MarketData):
        try:
            if isinstance(market_data, OrderBook):
                buffer.OrderBook = market_data._buffer
            elif isinstance(market_data, (BarData, DailyBar)):
                buffer.BarData = market_data._buffer
            elif isinstance(market_data, TickData):
                buffer.TickData = market_data._buffer
            elif isinstance(market_data, (TransactionData, TradeData)):
                buffer.TransactionData = market_data._buffer
            else:
                raise ValueError(f'Invalid market_data type {type(market_data)}!')
        except TypeError as _:
            raise TypeError('Incompatible types, this might comes from amending Contexts after initialization. Try to clear the cache and run again!')

    @classmethod
    def from_buffer(cls, buffer) -> OrderBook | BarData | DailyBar | TickData | TransactionData | TradeData | MarketData:
        buffer = MarketData.parse_buffer(buffer=buffer)
        md = MarketData.from_buffer(buffer=buffer)
        return md

    @classmethod
    def cast_buffer(cls, buffer) -> OrderBook | BarData | DailyBar | TickData | TransactionData | TradeData | MarketData:
        md = MarketData.cast_buffer(buffer=buffer)
        return md

    @classmethod
    def from_bytes(cls, buffer) -> OrderBook | BarData | DailyBar | TickData | TransactionData | TradeData | MarketData:
        md = MarketData.from_bytes(buffer)
        return md

    def update(self, market_data: MarketData):
        return self.to_buffer(buffer=self.buffer, market_data=market_data)

    def to_market_data(self) -> OrderBook | BarData | DailyBar | TickData | TransactionData | TradeData | MarketData:
        return self.cast_buffer(buffer=self.buffer)

    @property
    def contents(self) -> MarketData:
        return self.from_buffer(buffer=self.buffer)


class MarketDataRingBuffer(MarketDataBuffer):
    def __init__(self, size: int, **kwargs):
        self.size = size
        self.block = kwargs.get('block', False)
        self.condition_put = kwargs.get('condition_put', Condition())
        self.condition_get = kwargs.get('condition_get', Condition())
        self._index = RawArray(ctypes.c_int, 2)

        super().__init__(buffer=RawArray(self.ctype_buffer, self.size))

    @overload
    def __getitem__(self, index: slice) -> list[MarketDataBuffer]:
        ...

    @overload
    def __getitem__(self, index: int) -> MarketDataBuffer:
        ...

    def __getitem__(self, index):
        """
        based on the virtual index, not the internal (actual) index.
        """
        if isinstance(index, slice):
            return self._get_slice(index)
        elif isinstance(index, int):
            return self._get(index)
        else:
            raise TypeError(f'Invalid index {index}. Expected int or slice!')

    def __len__(self) -> int:
        return self.tail - self.head

    def _get_slice(self, index: slice) -> list[MarketDataBuffer]:
        start, stop, step = index.start, index.stop, index.step
        return [self._get(i) for i in range(start, stop, step if step is not None else 1)]

    def _get(self, index: int) -> MarketDataBuffer:
        """
        the internal method of get will not increase the index
        """
        valid_length = self.__len__()
        if -valid_length <= index < valid_length:
            index = index % valid_length
        else:
            raise IndexError(f'Index {index} is out of bounds!')

        internal_index = (index + self.head) % self.size
        return self.at(internal_index)

    def get(self, raise_on_empty: bool = False) -> MarketData | None:
        while self.is_empty():
            if raise_on_empty:
                raise ValueError(f'Buffer {self.__class__.__name__} is empty!')

            if not self.block:
                return None

            with self.condition_get:
                self.condition_get.wait()

        buffer = self.at(index=self.head)
        md = buffer.to_market_data()

        if self.is_full() and self.block:
            self.head += 1
            self.condition_put.notify_all()
        else:
            self.head += 1

        return md

    def _put(self, market_data: MarketData):
        """
        the internal method of put will not increase the index
        """
        buffer = self.at(index=self.tail)
        buffer.update(market_data=market_data)

    def put(self, market_data: MarketData, raise_on_full: bool = False):
        """
        the put method is not thread safe, and should only be called in one single thread.
        """
        while self.is_full():
            if raise_on_full:
                raise ValueError(f'Buffer {self.__class__.__name__} is full!')

            if not self.block:
                continue

            with self.condition_put:
                self.condition_put.wait()

        self._put(market_data=market_data)

        if self.is_empty() and self.block:
            with self.condition_get:
                self.tail += 1
                self.condition_get.notify_all()
        else:
            self.tail += 1

    def at(self, index: int) -> MarketDataBuffer:
        return self._at(index % self.size)

    def _at(self, index: int) -> MarketDataBuffer:
        return MarketDataBuffer(buffer=self.buffer[index])

    def is_full(self) -> bool:
        _tail_next = self.tail + 1
        _head_next_circle = self.head + self.size

        # to be more generic
        # _tail_next = _tail_next % self.size
        # _head_next_circle = _head_next_circle % self.size

        return _tail_next == _head_next_circle

    def is_empty(self) -> bool:
        idx_tail = self.tail
        idx_head = self.head

        # to be more generic
        # idx_tail = idx_tail % self.size
        # idx_head = idx_head % self.size

        return idx_head == idx_tail

    @property
    def head(self) -> int:
        return self._index[0]

    @property
    def tail(self) -> int:
        return self._index[1]

    @head.setter
    def head(self, value: int):
        self._index[0] = value

    @tail.setter
    def tail(self, value: int):
        self._index[1] = value


# alias of the BarData
CandleStick = BarData
