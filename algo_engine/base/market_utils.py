import abc
import datetime
import enum
import json
import math
import re
import warnings
from ctypes import c_ulong, c_double, c_wchar, c_int, c_longlong
from multiprocessing import RawValue, RawArray
from typing import overload, Literal, Self

import numpy as np

from . import LOGGER, PROFILE

LOGGER = LOGGER.getChild('MarketUtils')
__all__ = ['TransactionSide',
           'MarketData', 'OrderBook', 'BarData', 'DailyBar', 'CandleStick', 'TickData', 'TransactionData', 'TradeData',
           'MarketDataMemoryBuffer', 'OrderBookRawValue', 'BarDataRawValue', 'TickDataRawValue', 'TransactionDataRawValue']


class TransactionSide(enum.Enum):
    """
    Enumeration representing different sides of a financial transaction.

    Attributes:
        ShortOrder: Represents an order to short.
        ShortOpen: Represents the opening of a short position.
        ShortFilled: Represents a filled short order.
        UNKNOWN: Represents an unknown transaction side. Normally a cancel order.
        LongFilled: Represents a filled long order.
        ShortClose: Represents the closing of a short position.
        LongOrder: Represents an order to go long.
    """

    ShortOrder = AskOrder = Offer_to_Short = -3
    ShortOpen = Sell_to_Short = -2
    ShortFilled = LongClose = Sell_to_Unwind = ask = -1
    UNKNOWN = CANCEL = 0
    LongFilled = LongOpen = Buy_to_Long = bid = 1
    ShortClose = Buy_to_Cover = 2
    LongOrder = BidOrder = Bid_to_Long = 3

    def __lt__(self, other):
        """
        Compare if this transaction side is less than another.

        This comparison is deprecated.

        Args:
            other: The other TransactionSide to compare with.

        Returns:
            bool: True if this transaction side is less than the other, otherwise False.
        """
        warnings.warn(DeprecationWarning('Comparison of the <TransactionSide> deprecated!'))
        if self.__class__ is other.__class__:
            return self.value < other.value
        else:
            return self.value < other

    def __gt__(self, other):
        """
        Compare if this transaction side is greater than another.

        This comparison is deprecated.

        Args:
            other: The other TransactionSide to compare with.

        Returns:
            bool: True if this transaction side is greater than the other, otherwise False.
        """
        warnings.warn(DeprecationWarning('Comparison of the <TransactionSide> deprecated!'))
        if self.__class__ is other.__class__:
            return self.value > other.value
        else:
            return self.value > other

    def __eq__(self, other):
        """
        Compare if this transaction side is equal to another.

        Args:
            other: The other TransactionSide to compare with.

        Returns:
            bool: True if this transaction side is equal to the other, otherwise False.
        """
        if self.__class__ is other.__class__:
            return self.value == other.value
        else:
            return self.value == other

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

    def __hash__(self):
        """
        Get the hash value of this transaction side.

        Returns:
            int: The hash value of the transaction side.
        """
        return self.value

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
            LOGGER.warning(f'Requesting .sign of {self.name} is not recommended, use .order_sign instead')
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


class MarketData(dict, metaclass=abc.ABCMeta):
    """
    Abstract base class for representing market data in the form of a dictionary.

    Properties:
        ticker (str): The ticker symbol associated with the market data.
        timestamp (float): The timestamp of the market data in seconds since the epoch.
        additional (dict): A dictionary to store any additional attributes provided during initialization.

    Methods:
        __copy__(): Create a shallow copy of the MarketData instance.
        copy(): Alias for __copy__().
        to_json(fmt='str', **kwargs): Convert the MarketData instance to a JSON-compatible format.
        from_json(json_message: str | bytes | bytearray | dict) -> MarketData: Create a MarketData instance from a JSON string or dictionary.
        to_list() -> list[float | int | str | bool]: Abstract method to convert the MarketData instance to a list.
        from_list(data_list: list[float | int | str | bool]) -> MarketData: Create a MarketData instance from a list.
    """

    def __init__(self, ticker: str, timestamp: float, **kwargs):
        """
        Initialize a MarketData instance.

        Args:
            ticker (str): The ticker symbol associated with the market data.
            timestamp (float): The timestamp of the market data in seconds since the epoch.
            **kwargs: Additional keyword arguments to store as extra attributes.
        """
        super().__init__(ticker=ticker, timestamp=timestamp)

        if kwargs:
            self['additional'] = dict(kwargs)

    def __copy__(self):
        """
        Create a shallow copy of the MarketData instance.

        Returns:
            MarketData: A new instance of MarketData with the same data.
        """
        return self.__class__(**self)

    def copy(self):
        """
        Alias for __copy__().

        Returns:
            MarketData: A new instance of MarketData with the same data.
        """
        return self.__copy__()

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        """
        Convert the MarketData instance to a JSON-compatible format.

        Args:
            fmt (str, optional): The output format. Can be 'str' to return a JSON string,
                                 or 'dict' to return a dictionary. Defaults to 'str'.
            **kwargs: Additional keyword arguments passed to the json.dumps function.

        Returns:
            str | dict: The MarketData instance as a JSON string or dictionary.
        """
        data_dict = dict(
            dtype=self.__class__.__name__,
            **self
        )

        if 'additional' in data_dict:
            additional = data_dict.pop('additional')
            data_dict.update(additional)

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, expected "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        """
        Create a MarketData instance from a JSON string or dictionary.

        Args:
            json_message (str | bytes | bytearray | dict): The JSON string or dictionary containing market data information.

        Returns:
            MarketData: An instance of a subclass of MarketData.

        Raises:
            TypeError: If the 'dtype' key in the JSON message is invalid or unrecognized.
        """
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        dtype = json_dict.pop('dtype', None)
        if dtype == 'BarData':
            return BarData.from_json(json_dict)
        elif dtype == 'TickData':
            return TickData.from_json(json_dict)
        elif dtype == 'TransactionData':
            return TransactionData.from_json(json_dict)
        elif dtype == 'TradeData':
            return TradeData.from_json(json_dict)
        elif dtype == 'OrderBook':
            return OrderBook.from_json(json_dict)
        else:
            raise TypeError(f'Invalid dtype {dtype}')

    @abc.abstractmethod
    def to_list(self) -> list[float | int | str | bool]:
        """
        Convert the MarketData instance to a list.

        Returns:
            list[float | int | str | bool]: A list representing the MarketData instance.

        Note:
            This method must be implemented by subclasses.
        """
        ...

    @classmethod
    def from_list(cls, data_list: list[float | int | str | bool]) -> Self:
        """
        Create a MarketData instance from a list.

        Args:
            data_list (list[float | int | str | bool]): A list containing market data information.

        Returns:
            MarketData: An instance of a subclass of MarketData.

        Raises:
            TypeError: If the first element of the list (dtype) is invalid or unrecognized.
        """
        dtype = data_list[0]

        if dtype == 'BarData':
            return BarData.from_list(data_list)
        elif dtype == 'TickData':
            return TickData.from_list(data_list)
        elif dtype == 'TransactionData':
            return TransactionData.from_list(data_list)
        elif dtype == 'TradeData':
            return TradeData.from_list(data_list)
        elif dtype == 'OrderBook':
            return OrderBook.from_list(data_list)
        else:
            raise TypeError(f'Invalid dtype {dtype}')

    @property
    def ticker(self) -> str:
        """
        Get the ticker symbol of the market data.

        Returns:
            str: The ticker symbol.
        """
        return self['ticker']

    @property
    def timestamp(self) -> float:
        """
        Get the timestamp of the market data.

        Returns:
            float: The timestamp in seconds since the epoch.
        """
        return self['timestamp']

    @property
    def additional(self) -> dict:
        """
        Get the additional attributes stored in the market data.

        Returns:
            dict: A dictionary of additional attributes.
        """
        if 'additional' not in self:
            self['additional'] = {}
        return self['additional']

    @property
    def topic(self) -> str:
        """
        Get the topic string for the market data, used for messaging or logging.

        Returns:
            str: The topic string in the format 'ticker.MarketDataClassName'.
        """
        return f'{self.ticker}.{self.__class__.__name__}'

    @property
    def market_time(self) -> datetime.datetime | datetime.date:
        """
        Get the market time as a datetime object.

        Returns:
            datetime.datetime | datetime.date: The market time based on the timestamp.
        """
        return datetime.datetime.fromtimestamp(self.timestamp, tz=PROFILE.time_zone)

    @property
    @abc.abstractmethod
    def market_price(self) -> float:
        """
        Abstract property to get the market price.

        Returns:
            float: The market price.

        Note:
            This property must be implemented by subclasses.
        """
        ...


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
        super().__init__(ticker=ticker, timestamp=timestamp)
        self.update(
            bid=[] if bid is None else bid,
            ask=[] if ask is None else ask
        )
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
            book: list = self[side]

            if level <= 0:
                raise ValueError(f'Level of name [{name}] must be greater than zero!')

            entry_idx = {'price': 0, 'volume': 1, 'order': 2}[key]

            while level > len(book):
                book.append([math.nan, 0, 0])

            book[level - 1][entry_idx] = value

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

        self = cls(**json_dict)
        return self

    def to_list(self) -> list[float | int | str | bool]:
        """
        Convert the OrderBook to a list format.

        Returns:
            list[float | int | str | bool]: A list representation of the OrderBook.
        """
        min_length = min(len(self.bid), len(self.ask))
        r = ([self.__class__.__name__, self.ticker, self.timestamp]
             + [value for entry in self.bid[:min_length] for value in entry]
             + [value for entry in self.ask[:min_length] for value in entry])

        return r

    @classmethod
    def from_list(cls, data_list: list[float | int | str | bool]) -> Self:
        """
        Create an OrderBook instance from a list format.

        Args:
            data_list (list[float | int | str | bool]): A list representation of an OrderBook.

        Returns:
            OrderBook: An instance of the OrderBook class.

        Raises:
            TypeError: If the dtype in the list does not match the class name.
        """
        dtype, ticker, timestamp = data_list[:3]
        bid_data, ask_data = np.array(data_list[3:]).reshape(2, -1).tolist()
        bid = np.array(bid_data).reshape(-1, 3).tolist()
        ask = np.array(ask_data).reshape(-1, 3).tolist()

        if dtype != cls.__name__:
            raise TypeError(f'dtype mismatch, expect {cls.__name__}, got {dtype}.')

        return cls(
            ticker=ticker,
            timestamp=timestamp,
            bid=bid,
            ask=ask
        )

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
        for price, volume, *_ in self['bid']:
            book.add(price=price, volume=volume)
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
        for price, volume, *_ in self['ask']:
            book.add(price=price, volume=volume)
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
            return book[0][1]
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
            return book[0][1]
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
        super().__init__(ticker=ticker, timestamp=timestamp, **kwargs)
        self.update(
            high_price=high_price,
            low_price=low_price,
            open_price=open_price,
            close_price=close_price,
            volume=volume,
            notional=notional,
            trade_count=trade_count
        )

        if bar_span is None and start_timestamp is None:
            raise ValueError('Must assign either start_timestamp or bar_span or both.')
        elif start_timestamp is None:
            # self['start_timestamp'] = timestamp - bar_span.total_seconds()
            if isinstance(bar_span, datetime.timedelta):
                self['bar_span'] = bar_span.total_seconds()
            elif isinstance(bar_span, (int, float)):
                self['bar_span'] = bar_span
            else:
                raise ValueError(f'Invalid bar_span, expect int, float or timedelta, got {bar_span}')
        elif bar_span is None:
            self['start_timestamp'] = start_timestamp
        else:
            self['start_timestamp'] = start_timestamp

            if isinstance(bar_span, datetime.timedelta):
                self['bar_span'] = bar_span.total_seconds()
            elif isinstance(bar_span, (int, float)):
                self['bar_span'] = bar_span
            else:
                raise ValueError(f'Invalid bar_span, expect int, float or timedelta, got {bar_span}')

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

    def to_list(self) -> list[float | int | str | bool]:
        """
        Converts the `BarData` instance to a list of its attributes.

        Returns:
            list[float | int | str | bool]: A list containing the `BarData` instance's attributes.
        """
        return [self.__class__.__name__,
                self.ticker,
                self.timestamp,
                self.high_price,
                self.low_price,
                self.open_price,
                self.close_price,
                self.get('start_timestamp'),
                self.get('bar_span'),
                self.volume,
                self.notional,
                self.trade_count]

    @classmethod
    def from_list(cls, data_list: list[float | int | str | bool]) -> Self:
        """
        Creates a `BarData` instance from a list of attributes.

        Args:
            data_list (list[float | int | str | bool]): A list of attributes representing a `BarData` instance.

        Returns:
            BarData: A `BarData` instance initialized with the data from the list.

        Raises:
            TypeError: If the dtype in the list does not match the class name.
        """
        (dtype, ticker, timestamp, high_price, low_price, open_price, close_price,
         start_timestamp, bar_span, volume, notional, trade_count) = data_list

        if dtype != cls.__name__:
            raise TypeError(f'dtype mismatch, expect {cls.__name__}, got {dtype}.')

        return cls(
            ticker=ticker,
            timestamp=timestamp,
            high_price=high_price,
            low_price=low_price,
            open_price=open_price,
            close_price=close_price,
            start_timestamp=start_timestamp if start_timestamp else None,
            bar_span=datetime.timedelta(bar_span) if bar_span else None,
            volume=volume,
            notional=notional,
            trade_count=trade_count
        )

    @property
    def high_price(self) -> float:
        """
        The highest price during the bar.

        Returns:
            float: The highest price during the bar.
        """
        return self['high_price']

    @property
    def low_price(self) -> float:
        """
        The lowest price during the bar.

        Returns:
            float: The lowest price during the bar.
        """
        return self['low_price']

    @property
    def open_price(self) -> float:
        """
        The opening price of the bar.

        Returns:
            float: The opening price of the bar.
        """
        return self['open_price']

    @property
    def close_price(self) -> float:
        """
        The closing price of the bar.

        Returns:
            float: The closing price of the bar.
        """
        return self['close_price']

    @property
    def bar_span(self) -> datetime.timedelta:
        """
        The duration of the bar.

        Returns:
            datetime.timedelta: The duration of the bar.
        """
        if 'bar_span' in self:
            return datetime.timedelta(seconds=self['bar_span'])
        else:
            return datetime.timedelta(seconds=self['timestamp'] - self['start_timestamp'])

    @property
    def volume(self) -> float:
        """
        The total volume of trades during the bar.

        Returns:
            float: The total volume of trades during the bar.
        """
        return self['volume']

    @property
    def notional(self) -> float:
        """
        The total notional value of trades during the bar.

        Returns:
            float: The total notional value of trades during the bar.
        """
        return self['notional']

    @property
    def trade_count(self) -> int:
        """
        The number of trades that occurred during the bar.

        Returns:
            int: The number of trades that occurred during the bar.
        """
        return self['trade_count']

    @property
    def bar_start_time(self) -> datetime.datetime:
        """
        The start time of the bar.

        Returns:
            datetime.datetime: The start time of the bar.
        """
        if 'start_timestamp' in self:
            return datetime.datetime.fromtimestamp(self['start_timestamp'], tz=PROFILE.time_zone)
        else:
            return datetime.datetime.fromtimestamp(self['timestamp'] - self['bar_span'], tz=PROFILE.time_zone)

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
            assert type(self.ticker) is str, '{} Invalid ticker'.format(str(self))
            assert math.isfinite(self.high_price), '{} Invalid high_price'.format(str(self))
            assert math.isfinite(self.low_price), '{} Invalid low_price'.format(str(self))
            assert math.isfinite(self.open_price), '{} Invalid open_price'.format(str(self))
            assert math.isfinite(self.close_price), '{} Invalid close_price'.format(str(self))
            assert math.isfinite(self.volume), '{} Invalid volume'.format(str(self))
            assert math.isfinite(self.notional), '{} Invalid notional'.format(str(self))
            assert math.isfinite(self.trade_count), '{} Invalid trade_count'.format(str(self))
            assert isinstance(self.bar_start_time, (datetime.datetime, datetime.date)), '{} Invalid bar_start_time'.format(str(self))
            assert isinstance(self.bar_span, datetime.timedelta), '{} Invalid bar_span'.format(str(self))

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
        if self['bar_span'] > 3600:
            return 'Hourly-Plus'
        elif self['bar_span'] == 3600:
            return 'Hourly'
        elif self['bar_span'] > 60:
            return 'Minute-Plus'
        elif self['bar_span'] == 60:
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


# alias of the BarData
CandleStick = BarData


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
            timestamp (float, optional): The timestamp marking the end of the bar. Defaults to None.
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

        if timestamp is None:
            if PROFILE.session_end is None:
                timestamp = datetime.datetime.combine(market_date, datetime.time.min, tzinfo=PROFILE.time_zone).timestamp()
            else:
                timestamp = datetime.datetime.combine(market_date, PROFILE.session_end, tzinfo=PROFILE.time_zone).timestamp()

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

        self['market_date'] = market_date

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

    def to_list(self) -> list[float | int | str | bool]:
        """
        Converts the `DailyBar` instance to a list of its attributes.

        Returns:
            list[float | int | str | bool]: A list containing the `DailyBar` instance's attributes.
        """
        return [self.__class__.__name__,
                self.ticker,
                self.market_date.isoformat(),
                self.timestamp,
                self.high_price,
                self.low_price,
                self.open_price,
                self.close_price,
                self['bar_span'],
                self.volume,
                self.notional,
                self.trade_count]

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
        return datetime.timedelta(days=self['bar_span'])

    @property
    def market_date(self) -> datetime.date:
        """
        The market date of the bar.

        Returns:
            datetime.date: The market date of the bar.
        """
        return self['market_date']

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
        if self['bar_span'] == 1:
            return 'Daily'
        elif self['bar_span'] > 1:
            return 'Daily-Plus'
        else:
            raise ValueError(f'Invalid bar_span for {self.__class__.__name__}! Expect an int greater or equal to 1, got {self["bar_span"]}')


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
            bid: list[list[float | int]] = None,
            ask: list[list[float | int]] = None,
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
            bid (list[list[float | int]], optional): A list of bid prices and volumes. Defaults to None.
            ask (list[list[float | int]], optional): A list of ask prices and volumes. Defaults to None.
            total_traded_volume (float, optional): The total traded volume. Defaults to 0.0.
            total_traded_notional (float, optional): The total traded notional value. Defaults to 0.0.
            total_trade_count (int, optional): The total number of trades. Defaults to 0.
            **kwargs: Additional keyword arguments passed to the parent `MarketData` class.
        """
        super().__init__(ticker=ticker, timestamp=timestamp, **kwargs)

        self.update(
            last_price=last_price,
            total_traded_volume=total_traded_volume,
            total_traded_notional=total_traded_notional,
            total_trade_count=total_trade_count,
        )

        if order_book is not None:
            self['order_book'] = {'bid': order_book['bid'], 'ask': order_book['ask']}
        elif bid and ask:
            self['order_book'] = {
                'bid': sorted(bid, key=lambda _: _[0], reverse=True),
                'ask': sorted(ask, key=lambda _: _[0], reverse=False)
            }

            if bid_price is None:
                bid_price, _, *_ = bid[0]
            if bid_volume is None:
                _, bid_volume, *_ = bid[0]
            if ask_price is None:
                ask_price, _, *_ = ask[0]
            if ask_volume is None:
                _, ask_volume, *_ = ask[0]

        if bid_price is not None and math.isfinite(bid_price):
            self['bid_price'] = bid_price

        if bid_volume is not None and math.isfinite(bid_volume):
            self['bid_volume'] = bid_volume

        if ask_price is not None and math.isfinite(ask_price):
            self['ask_price'] = ask_price

        if ask_volume is not None and math.isfinite(ask_volume):
            self['ask_volume'] = ask_volume

    @property
    def level_2(self) -> OrderBook | None:
        """
        The level 2 order book created from the bid and ask data.

        Returns:
            OrderBook | None: The `OrderBook` instance if available, otherwise `None`.
        """
        if 'order_book' in self:
            return OrderBook(ticker=self.ticker, timestamp=self.timestamp, **self['order_book'])
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
        return self['last_price']

    @property
    def bid_price(self) -> float | None:
        """
        The bid price.

        Returns:
            float | None: The bid price if available, otherwise `None`.
        """
        return self.get('bid_price')

    @property
    def ask_price(self) -> float | None:
        """
        The ask price.

        Returns:
            float | None: The ask price if available, otherwise `None`.
        """
        return self.get('ask_price')

    @property
    def bid_volume(self) -> float | None:
        """
        The bid volume.

        Returns:
            float | None: The bid volume if available, otherwise `None`.
        """
        return self.get('bid_volume')

    @property
    def ask_volume(self) -> float | None:
        """
        The ask volume.

        Returns:
            float | None: The ask volume if available, otherwise `None`.
        """
        return self.get('ask_volume')

    @property
    def total_traded_volume(self) -> float:
        """
        The total traded volume.

        Returns:
            float: The total traded volume.
        """
        return self['total_traded_volume']

    @property
    def total_traded_notional(self) -> float:
        """
        The total traded notional value.

        Returns:
            float: The total traded notional value.
        """
        return self['total_traded_notional']

    @property
    def total_trade_count(self) -> float:
        """
        The total number of trades.

        Returns:
            float: The total number of trades.
        """
        return self['total_trade_count']

    def __repr__(self) -> str:
        """
        Returns a string representation of the `TickData` instance.

        The string representation includes the class name, market time, ticker symbol, and bid/ask prices.

        Returns:
            str: A string representation of the `TickData` instance.
        """
        return f'<TickData>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker}, bid={self.bid_price}, ask={self.ask_price})'

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

        self = cls(**json_dict)
        return self

    def to_list(self) -> list[float | int | str | bool]:
        """
        Converts the `TickData` instance to a list of its attributes.

        Note:
            This method does not retain order book information. To save all information, use `to_json` instead.

        Returns:
            list[float | int | str | bool]: A list of attribute values.
        """
        return [self.__class__.__name__,
                self.ticker,
                self.timestamp,
                self.last_price,
                self.bid_price,
                self.bid_volume,
                self.ask_price,
                self.ask_volume,
                self.total_traded_volume,
                self.total_traded_notional,
                self.total_trade_count]

    @classmethod
    def from_list(cls, data_list: list[float | int | str | bool]) -> Self:
        """
        Creates a `TickData` instance from a list of attributes.

        Args:
            data_list (list[float | int | str | bool]): A list of attributes in the order:
                - dtype (str)
                - ticker (str)
                - timestamp (float)
                - last_price (float)
                - bid_price (float | None)
                - bid_volume (float | None)
                - ask_price (float | None)
                - ask_volume (float | None)
                - total_traded_volume (float)
                - total_traded_notional (float)
                - total_trade_count (int)

        Returns:
            TickData: A `TickData` instance.

        Raises:
            TypeError: If the dtype in the list does not match the class name.
        """
        (dtype, ticker, timestamp, last_price,
         bid_price, bid_volume, ask_price, ask_volume,
         total_traded_volume, total_traded_notional, total_trade_count) = data_list

        if dtype != cls.__name__:
            raise TypeError(f'dtype mismatch, expect {cls.__name__}, got {dtype}.')

        kwargs = {}

        if bid_price is not None:
            kwargs['bid_price'] = bid_price

        if ask_price is not None:
            kwargs['ask_price'] = ask_price

        if bid_volume is not None:
            kwargs['bid_volume'] = bid_volume

        if ask_volume is not None:
            kwargs['ask_volume'] = ask_volume

        return cls(
            ticker=ticker,
            timestamp=timestamp,
            last_price=last_price,
            total_traded_volume=total_traded_volume,
            total_traded_notional=total_traded_notional,
            total_trade_count=total_trade_count,
            **kwargs
        )

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
        super().__init__(ticker=ticker, timestamp=timestamp, **kwargs)

        self['price'] = price
        self['volume'] = volume
        self['side'] = int(side) if isinstance(side, (int, float)) else TransactionSide(side).value

        if multiplier is not None and math.isfinite(multiplier):
            self['multiplier'] = multiplier

        if notional is not None and math.isfinite(notional):
            self['notional'] = notional

        if transaction_id is not None:
            self['transaction_id'] = transaction_id

        if buy_id is not None:
            self['buy_id'] = buy_id

        if sell_id is not None:
            self['sell_id'] = sell_id

    def __repr__(self) -> str:
        """
        Returns a string representation of the `TransactionData` instance.

        The string representation includes the class name, market time, side, ticker symbol, price, and volume.

        Returns:
            str: A string representation of the `TransactionData` instance.
        """
        return f'<TransactionData>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.side.side_name} {self.ticker}, price={self.price}, volume={self.volume})'

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

    def to_list(self) -> list[float | int | str | bool]:
        """
        Converts the `TransactionData` instance to a list of its attributes.

        Returns:
            list[float | int | str | bool]: A list of attribute values.
        """
        return [self.__class__.__name__,
                self.ticker,
                self.timestamp,
                self.price,
                self.volume,
                self['side'],
                self.get('multiplier'),
                self.get('notional'),
                self.get('transaction_id'),
                self.get('buy_id'),
                self.get('sell_id')]

    @classmethod
    def from_list(cls, data_list: list[float | int | str | bool]) -> Self:
        """
        Creates a `TransactionData` instance from a list of attributes.

        Args:
            data_list (list[float | int | str | bool]): A list of attributes in the order:
                - dtype (str)
                - ticker (str)
                - timestamp (float)
                - price (float)
                - volume (float)
                - side (int | float | str)
                - multiplier (float | None)
                - notional (float | None)
                - transaction_id (str | int | None)
                - buy_id (str | int | None)
                - sell_id (str | int | None)

        Returns:
            TransactionData: A `TransactionData` instance.

        Raises:
            TypeError: If the dtype in the list does not match the class name.
        """
        (dtype, ticker, timestamp, price, volume, side,
         multiplier, notional, transaction_id, buy_id, sell_id) = data_list

        if dtype != cls.__name__:
            raise TypeError(f'dtype mismatch, expect {cls.__name__}, got {dtype}.')

        kwargs = {}

        if multiplier is not None:
            kwargs['multiplier'] = multiplier

        if notional is not None:
            kwargs['notional'] = notional

        if transaction_id is not None:
            kwargs['transaction_id'] = transaction_id

        if buy_id is not None:
            kwargs['buy_id'] = buy_id

        if sell_id is not None:
            kwargs['sell_id'] = sell_id

        return cls(
            ticker=ticker,
            timestamp=timestamp,
            price=price,
            volume=volume,
            side=side,
            **kwargs
        )

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
        return self['price']

    @property
    def volume(self) -> float:
        """
        The volume of the transaction.

        Returns:
            float: The transaction volume.
        """
        return self['volume']

    @property
    def side(self) -> TransactionSide:
        """
        The side of the transaction (buy or sell).

        Returns:
            TransactionSide: The side of the transaction.
        """
        return TransactionSide(self['side'])

    @property
    def multiplier(self) -> float:
        """
        The multiplier for the transaction. Defaults to 1 if not specified.

        Returns:
            float: The transaction multiplier.
        """
        return self.get('multiplier', 1.)

    @property
    def transaction_id(self) -> int | str | None:
        """
        The identifier for the transaction.

        Returns:
            int | str | None: The transaction identifier.
        """
        return self.get('transaction_id', None)

    @property
    def buy_id(self) -> int | str | None:
        """
        The identifier for the buying transaction.

        Returns:
            int | str | None: The buying transaction identifier.
        """
        return self.get('buy_id', None)

    @property
    def sell_id(self) -> int | str | None:
        """
        The identifier for the selling transaction.

        Returns:
            int | str | None: The selling transaction identifier.
        """
        return self.get('sell_id', None)

    @property
    def notional(self) -> float:
        """
        The notional value of the transaction. Calculated as price * volume * multiplier.

        Returns:
            float: The transaction notional.
        """
        return self.get('notional', self.price * self.volume * self.multiplier)

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
        return self['price']

    @property
    def trade_volume(self) -> float:
        """
        Alias for the transaction volume.

        Returns:
            float: The transaction volume.
        """
        return self['volume']

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
    def from_list(cls, data_list: list[float | int | str | bool]) -> Self:
        """
        Creates a `TradeData` instance from a list of attributes.

        Args:
            data_list (list[float | int | str | bool]): A list of attributes.

        Returns:
            TradeData: A `TradeData` instance.
        """
        return super(TradeData, cls).from_list(data_list=data_list)


class MarketDataMemoryBuffer(object, metaclass=abc.ABCMeta):
    """
    Abstract base class for a market data memory buffer.

    Attributes:
        dtype (RawArray): Data type of the market data.
        ticker (RawArray): Ticker symbol.
        timestamp (RawValue): Timestamp of the market data.

    Methods:
        update(market_data: MarketData):
            Updates the buffer with new market data.

        to_market_data() -> MarketData:
            Abstract method to convert the buffer data to a `MarketData` instance.
    """

    def __init__(self):
        """
        Initializes the memory buffer with default values.
        """
        LOGGER.warning(DeprecationWarning(f'{self.__class__.__name__} deprecated. Please use the module from algo_engine.base.market_buffer instead.'))
        self.dtype = RawArray(c_wchar, 16)
        self.ticker = RawArray(c_wchar, 32)  # max length of ticker is 32
        self.timestamp = RawValue(c_double)

    def update(self, market_data: MarketData):
        """
        Updates the buffer with new market data.

        Args:
            market_data (MarketData): An instance of `MarketData` containing new data.
        """
        self.dtype.value = market_data.__class__.__name__
        self.ticker.value = market_data['ticker']
        self.timestamp.value = market_data['timestamp']

    @abc.abstractmethod
    def to_market_data(self) -> MarketData:
        """
        Abstract method to convert the buffer data to a `MarketData` instance.

        Returns:
            MarketData: A `MarketData` instance populated with buffer data.
        """
        ...


class OrderBookRawValue(MarketDataMemoryBuffer):
    """
    Memory buffer for order book data.

    Attributes:
        max_level (int): Maximum levels for bid and ask data.
        bid (list[tuple[RawValue, RawValue, RawValue]]): List of bid data buffers.
        ask (list[tuple[RawValue, RawValue, RawValue]]): List of ask data buffers.

    Methods:
        update(market_data: OrderBook):
            Updates the buffer with new order book data.

        to_market_data() -> OrderBook:
            Converts the buffer data to an `OrderBook` instance.
    """

    def __init__(self, max_level: int = 20):
        """
        Initializes the order book buffer with the specified maximum levels.

        Args:
            max_level (int): Maximum number of levels for bid and ask data. Defaults to 20.
        """
        LOGGER.warning(DeprecationWarning(f'{self.__class__.__name__} deprecated. Please use the module from algo_engine.base.market_buffer instead.'))
        super().__init__()

        self.max_level = max_level

        self.bid = [(RawValue(c_double), RawValue(c_double), RawValue(c_ulong)) for _ in range(self.max_level)]
        self.ask = [(RawValue(c_double), RawValue(c_double), RawValue(c_ulong)) for _ in range(self.max_level)]

    def update(self, market_data: OrderBook):
        """
        Updates the buffer with new order book data.

        Args:
            market_data (OrderBook): An instance of `OrderBook` containing new data.
        """
        super().update(market_data=market_data)

        bid = market_data['bid']
        ask = market_data['ask']

        for i in range(self.max_level):
            bid_memory_array = self.bid[i]
            ask_memory_array = self.ask[i]

            if i < len(bid):
                for bid_entry_value, bid_memory in zip(bid[i], bid_memory_array):
                    bid_memory.value = bid_entry_value
            else:
                for bid_memory in bid_memory_array:
                    bid_memory.value = 0

            if i < len(ask):
                for ask_entry_value, ask_memory in zip(ask[i], ask_memory_array):
                    ask_memory.value = ask_entry_value
            else:
                for ask_memory in ask_memory_array:
                    ask_memory.value = 0

    def to_market_data(self) -> OrderBook:
        """
        Converts the buffer data to an `OrderBook` instance.

        Returns:
            OrderBook: An `OrderBook` instance populated with buffer data.
        """
        bid, ask = [], []

        for i in range(self.max_level):
            bid_memory_array = self.bid[i]
            ask_memory_array = self.ask[i]

            bid_price, bid_volume, bid_n_orders = [_.value for _ in bid_memory_array]
            ask_price, ask_volume, ask_n_orders = [_.value for _ in ask_memory_array]

            if bid_volume:
                bid.append([bid_price, bid_volume, bid_n_orders])

            if ask_volume:
                ask.append([ask_price, ask_volume, ask_n_orders])

            if bid_volume == ask_volume == 0:
                break

        order_book = OrderBook(
            ticker=self.ticker.value,
            timestamp=self.timestamp.value,
            bid=bid,
            ask=ask,
        )

        return order_book


class BarDataRawValue(MarketDataMemoryBuffer):
    """
    Memory buffer for bar data.

    Attributes:
        start_timestamp (RawValue): Start timestamp of the bar.
        bar_span (RawValue): Span of the bar.
        high_price (RawValue): Highest price during the bar.
        low_price (RawValue): Lowest price during the bar.
        open_price (RawValue): Opening price of the bar.
        close_price (RawValue): Closing price of the bar.
        volume (RawValue): Total volume during the bar.
        notional (RawValue): Total notional value during the bar.
        trade_count (RawValue): Number of trades during the bar.

    Methods:
        update(market_data: BarData):
            Updates the buffer with new bar data.

        to_market_data() -> BarData:
            Converts the buffer data to a `BarData` instance.
    """

    def __init__(self):
        """
        Initializes the bar data buffer with default values.
        """
        LOGGER.warning(DeprecationWarning(f'{self.__class__.__name__} deprecated. Please use the module from algo_engine.base.market_buffer instead.'))
        super().__init__()

        self.start_timestamp = RawValue(c_double)
        self.bar_span = RawValue(c_double)
        self.high_price = RawValue(c_double)
        self.low_price = RawValue(c_double)
        self.open_price = RawValue(c_double)
        self.close_price = RawValue(c_double)
        self.volume = RawValue(c_double)
        self.notional = RawValue(c_double)
        self.trade_count = RawValue(c_longlong)

    def update(self, market_data: BarData):
        """
        Updates the buffer with new bar data.

        Args:
            market_data (BarData): An instance of `BarData` containing new data.
        """
        super().update(market_data=market_data)

        self.start_timestamp.value = market_data['start_timestamp'] if 'start_timestamp' in market_data else math.nan
        self.bar_span.value = market_data['bar_span'] if 'bar_span' in market_data else math.nan

        self.high_price.value = market_data['high_price']
        self.low_price.value = market_data['low_price']
        self.open_price.value = market_data['open_price']
        self.close_price.value = market_data['close_price']
        self.volume.value = market_data['volume']
        self.notional.value = market_data['notional']
        self.trade_count.value = market_data['trade_count']

    def to_market_data(self) -> BarData:
        """
        Converts the buffer data to a `BarData` instance.

        Returns:
            BarData: A `BarData` instance populated with buffer data.
        """
        bar_data = BarData(
            ticker=self.ticker.value,
            timestamp=self.timestamp.value,
            start_timestamp=self.start_timestamp.value if math.isfinite(self.start_timestamp.value) else None,
            bar_span=self.bar_span.value if math.isfinite(self.bar_span.value) else None,
            high_price=self.high_price.value,
            low_price=self.low_price.value,
            open_price=self.open_price.value,
            close_price=self.close_price.value,
            volume=self.volume.value,
            notional=self.notional.value,
            trade_count=self.trade_count.value
        )

        return bar_data


class TickDataRawValue(MarketDataMemoryBuffer):
    """
    Memory buffer for tick data.

    Attributes:
        last_price (RawValue): Last traded price.
        bid_price (RawValue): Current bid price.
        bid_volume (RawValue): Volume at the bid price.
        ask_price (RawValue): Current ask price.
        ask_volume (RawValue): Volume at the ask price.
        total_traded_volume (RawValue): Total traded volume.
        total_traded_notional (RawValue): Total traded notional value.
        total_trade_count (RawValue): Total number of trades.

    Methods:
        update(market_data: TickData):
            Updates the buffer with new tick data.

        to_market_data() -> TickData:
            Converts the buffer data to a `TickData` instance.
    """

    def __init__(self):
        """
        Initializes the tick data buffer with default values.
        """
        LOGGER.warning(DeprecationWarning(f'{self.__class__.__name__} deprecated. Please use the module from algo_engine.base.market_buffer instead.'))
        super().__init__()

        self.last_price = RawValue(c_double)
        self.bid_price = RawValue(c_double)
        self.bid_volume = RawValue(c_double)
        self.ask_price = RawValue(c_double)
        self.ask_volume = RawValue(c_double)
        self.total_traded_volume = RawValue(c_double)
        self.total_traded_notional = RawValue(c_double)
        self.total_trade_count = RawValue(c_longlong)

    def update(self, market_data: TickData):
        """
        Updates the buffer with new tick data.

        Args:
            market_data (TickData): An instance of `TickData` containing new data.

        Notes:
            The order book is not stored in shared memory. Use `OrderBookShared` to store the level 2 data.
        """
        super().update(market_data=market_data)

        self.last_price.value = market_data.last_price

        self.bid_price.value = market_data['bid_price'] if 'bid_price' in market_data else math.nan
        self.bid_volume.value = market_data['bid_volume'] if 'bid_volume' in market_data else math.nan
        self.ask_price.value = market_data['ask_price'] if 'ask_price' in market_data else math.nan
        self.ask_volume.value = market_data['ask_volume'] if 'ask_volume' in market_data else math.nan

        self.total_traded_volume.value = market_data['total_traded_volume']
        self.total_traded_notional.value = market_data['total_traded_notional']
        self.total_trade_count.value = market_data['total_trade_count']

    def to_market_data(self) -> TickData:
        """
        Converts the buffer data to a `TickData` instance.

        Returns:
            TickData: A `TickData` instance populated with buffer data.
        """
        tick_data = TickData(
            ticker=self.ticker.value,
            timestamp=self.timestamp.value,
            last_price=self.last_price.value,
            bid_price=self.bid_price.value,
            bid_volume=self.bid_volume.value,
            ask_price=self.ask_price.value,
            ask_volume=self.ask_volume.value,
            order_book=None,
            total_traded_volume=self.total_traded_volume.value,
            total_traded_notional=self.total_traded_notional.value,
            total_trade_count=self.total_trade_count.value,
        )

        return tick_data


class TransactionDataRawValue(MarketDataMemoryBuffer):
    """
    Memory buffer for transaction data.

    Attributes:
        price (RawValue): Transaction price.
        volume (RawValue): Transaction volume.
        side (RawValue): Side of the transaction (buy/sell).
        multiplier (RawValue): Multiplier for the transaction.
        notional (RawValue): Notional value of the transaction.
        id_map (dict[str, tuple[RawValue, RawArray]]): Map for various IDs related to the transaction.

    Methods:
        update(market_data: TradeData | TransactionData):
            Updates the buffer with new transaction data.

        to_market_data() -> TradeData | TransactionData:
            Converts the buffer data to a `TradeData` or `TransactionData` instance.
    """

    def __init__(self):
        """
        Initializes the transaction data buffer with default values.
        """
        LOGGER.warning(DeprecationWarning(f'{self.__class__.__name__} deprecated. Please use the module from algo_engine.base.market_buffer instead.'))
        super().__init__()

        self.dtype = RawArray(c_wchar, 16)
        self.price = RawValue(c_double)
        self.volume = RawValue(c_double)
        self.side = RawValue(c_int)

        self.multiplier = RawValue(c_double)
        self.notional = RawValue(c_double)
        self.id_map = dict(
            transaction_id=(RawValue(c_longlong), RawArray(c_wchar, 64)),  # id can be an int, str or None
            buy_id=(RawValue(c_longlong), RawArray(c_wchar, 64)),
            sell_id=(RawValue(c_longlong), RawArray(c_wchar, 64)),
        )

    def update(self, market_data: TradeData | TransactionData):
        """
        Updates the buffer with new transaction data.

        Args:
            market_data (TradeData | TransactionData): An instance of `TradeData` or `TransactionData` containing new data.
        """
        super().update(market_data=market_data)

        self.price.value = market_data['price']
        self.volume.value = market_data['volume']
        self.side.value = market_data['side']

        if 'multiplier' in market_data:
            self.multiplier.value = market_data['multiplier']
        else:
            self.multiplier.value = math.nan

        if 'notional' in market_data:
            self.notional.value = market_data['notional']
        else:
            self.notional.value = math.nan

        for id_name in ['transaction_id', 'buy_id', 'sell_id']:
            if id_name in market_data:
                id_value = market_data[id_name]
                if isinstance(id_value, int):
                    self.id_map[id_name][0].value = id_value
                    self.id_map[id_name][1].value = ''
                elif isinstance(id_value, str):
                    self.id_map[id_name][0].value = -1
                    self.id_map[id_name][1].value = id_value
                else:
                    raise TypeError(f'Invalid {id_name} type: {type(id_name)}, expect int or str.')
            else:
                self.id_map[id_name][0].value = -1
                self.id_map[id_name][1].value = ''

    def to_market_data(self) -> TradeData | TransactionData:
        """
        Converts the buffer data to a `TradeData` or `TransactionData` instance.

        Returns:
            TradeData | TransactionData: An instance of `TradeData` or `TransactionData` populated with buffer data.
        """
        if math.isnan(multiplier := self.multiplier.value):
            multiplier = None

        if math.isnan(notional := self.notional.value):
            notional = None

        id_map = {}
        for id_name in ['transaction_id', 'buy_id', 'sell_id']:
            id_int = self.id_map[id_name][0].value
            id_str = self.id_map[id_name][1].value

            if id_int == -1 and not id_str:
                id_value = None
            elif not id_str:
                id_value = id_int
            elif id_int == -1:
                id_value = id_str
            else:
                raise ValueError(f'id_map can not contain both info of id_int={id_int}, id_str={id_str}.')

            id_map[id_name] = id_value

        if (dtype := self.dtype.value) == 'TransactionData':
            constructor = TransactionData
        elif dtype == 'TradeData':
            constructor = TradeData
        else:
            raise NotImplementedError(f'Constructor for market data {dtype} not implemented.')

        td = constructor(
            ticker=self.ticker.value,
            price=self.price.value,
            volume=self.volume.value,
            timestamp=self.timestamp.value,
            side=self.side.value,
            multiplier=multiplier,
            notional=notional,
            **id_map
        )

        return td
