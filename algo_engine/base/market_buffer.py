import abc
import ctypes
import math
from multiprocessing import RawArray, RawValue, Condition
from typing import overload

import numpy as np

from .market_utils import MarketData, OrderBook, BarData, TickData, TransactionData, TradeData


class MarketDataPointer(object, metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self._ptr_dtype = kwargs.get('dtype')
        self._ptr_ticker = kwargs.get('ticker')
        self._ptr_timestamp = kwargs.get('timestamp')

    def update(self, market_data: MarketData, encoding='utf-8'):
        self._ptr_dtype.contents.value = market_data.__class__.__name__.encode(encoding).ljust(ctypes.sizeof(self._ptr_dtype.contents), b'\x00')
        self._ptr_ticker.contents.value = market_data['ticker'].encode(encoding).ljust(ctypes.sizeof(self._ptr_ticker.contents), b'\x00')
        self._ptr_timestamp.contents.value = float(market_data['timestamp'])

    @abc.abstractmethod
    def to_market_data(self, encoding='utf-8') -> MarketData:
        """
        Abstract method to convert the buffer data to a `MarketData` instance.

        Returns:
            MarketData: A `MarketData` instance populated with buffer data.
        """
        ...

    @property
    def contents(self) -> MarketData:
        return self.to_market_data()


class MarketDataMemoryBuffer(object, metaclass=abc.ABCMeta):
    def __init__(self, size: int, **kwargs):
        self.size = size
        self.block = kwargs.get('block', False)
        self.condition_put = kwargs.get('condition_put', Condition())
        self.condition_get = kwargs.get('condition_get', Condition())
        self._dtype_size = kwargs.get('dtype_size', 16)
        self._ticker_size = kwargs.get('ticker_size', 32)

        self._ticker_c_arr = ctypes.c_char * self._ticker_size
        self._dtype_c_arr = ctypes.c_char * self._dtype_size

        self.dtype = RawValue(self._dtype_c_arr)
        self.ticker = RawArray(self._ticker_c_arr, self.size)
        self.timestamp = RawArray(ctypes.c_double, self.size)
        self._index = RawValue(ctypes.c_int * 2)

    @overload
    def __getitem__(self, index: slice) -> list[MarketDataPointer]:
        ...

    @overload
    def __getitem__(self, index: int) -> MarketDataPointer:
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

    def _get_slice(self, index: slice) -> list[MarketDataPointer]:
        start, stop, step = index.start, index.stop, index.step
        return [self._get(i) for i in range(start, stop, step if step is not None else 1)]

    def _get(self, index: int) -> MarketDataPointer:
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

    def get(self, raise_on_empty: bool = False, encoding='utf-8') -> MarketData | None:
        while self.is_empty():
            if raise_on_empty:
                raise ValueError(f'Buffer {self.__class__.__name__} is empty!')

            if not self.block:
                return None

            with self.condition_get:
                self.condition_get.wait()

        md_ptr = self.at(index=self.head)
        md = md_ptr.to_market_data(encoding=encoding)

        if self.is_full() and self.block:
            self.head += 1
            self.condition_put.notify_all()
        else:
            self.head += 1

        return md

    def _put(self, market_data: MarketData, encoding='utf-8'):
        """
        the internal method of put will not increase the index
        """
        md_ptr = self.at(index=self.tail)
        md_ptr.update(market_data=market_data, encoding=encoding)

    def put(self, market_data: MarketData, raise_on_full: bool = False, encoding='utf-8'):
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

        self._put(market_data=market_data, encoding=encoding)

        if self.is_empty() and self.block:
            with self.condition_get:
                self.tail += 1
                self.condition_get.notify_all()
        else:
            self.tail += 1

    def at(self, index: int) -> MarketDataPointer:
        return self._at(index % self.size)

    @abc.abstractmethod
    def _at(self, index: int) -> MarketDataPointer:
        """
        get pointer by actual index
        """
        ...

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


class OrderBookPointer(MarketDataPointer):
    def __init__(self, max_level: int, **kwargs):
        super().__init__(
            dtype=kwargs.get('dtype'),
            ticker=kwargs.get('ticker'),
            timestamp=kwargs.get('timestamp')
        )

        self.max_level = max_level

        self._ptr_bid_price = kwargs.get('bid_price')
        self._ptr_ask_price = kwargs.get('ask_price')

        self._ptr_bid_volume = kwargs.get('bid_volume')
        self._ptr_ask_volume = kwargs.get('ask_volume')

        self._ptr_bid_n_orders = kwargs.get('bid_n_orders')
        self._ptr_ask_n_orders = kwargs.get('ask_n_orders')

    def update(self, market_data: OrderBook, encoding='utf-8'):
        super().update(market_data, encoding=encoding)

        bid = market_data['bid']
        ask = market_data['ask']

        bid_px, bid_vol, *bid_n = zip(*bid)
        ask_px, ask_vol, *ask_n = zip(*ask)

        self._ptr_bid_price.contents[:] = bid_px[:self.max_level] + [0] * (self.max_level - len(bid_px))
        self._ptr_ask_price.contents[:] = ask_px[:self.max_level] + [0] * (self.max_level - len(ask_px))

        self._ptr_bid_volume.contents[:] = bid_vol[:self.max_level] + [0] * (self.max_level - len(bid_vol))
        self._ptr_ask_volume.contents[:] = ask_vol[:self.max_level] + [0] * (self.max_level - len(ask_vol))

        if bid_n:
            bid_n = bid_n[0]
            self._ptr_bid_n_orders.contents[:] = bid_n[:self.max_level] + [0] * (self.max_level - len(bid_n))

        if ask_n:
            ask_n = ask_n[0]
            self._ptr_ask_n_orders.contents[:] = ask_n[:self.max_level] + [0] * (self.max_level - len(ask_n))

    def to_market_data(self, encoding='utf-8') -> MarketData:
        bid_px = self._ptr_bid_price.contents[:]
        bid_vol = self._ptr_bid_volume.contents[:]
        bid_n = self._ptr_bid_n_orders.contents[:]

        ask_px = self._ptr_ask_price.contents[:]
        ask_vol = self._ptr_ask_volume.contents[:]
        ask_n = self._ptr_ask_n_orders.contents[:]

        if any(bid_n):
            bid = zip(bid_px, bid_vol, bid_n)
        else:
            bid = zip(bid_px, bid_vol)

        if any(ask_n):
            ask = zip(ask_px, ask_vol, ask_n)
        else:
            ask = zip(ask_px, ask_vol)

        order_book = OrderBook(
            ticker=self._ptr_ticker.contents.value.decode(encoding),
            timestamp=self._ptr_timestamp.contents.value,
            bid=[list(_) for _ in bid],
            ask=[list(_) for _ in ask]
        )

        return order_book


class OrderBookBuffer(MarketDataMemoryBuffer):
    def __init__(self, size: int, max_level: int = 20, **kwargs):
        """
        Initializes the order book buffer with the specified maximum levels.

        Args:
            max_level (int): Maximum number of levels for bid and ask data. Defaults to 20.
        """
        super().__init__(size=size, **kwargs)
        self.max_level = max_level

        self._book_c_arr = ctypes.c_double * self.max_level
        self._order_c_arr = ctypes.c_int * self.max_level

        self.bid_price = RawArray(self._book_c_arr, self.size)
        self.ask_price = RawArray(self._book_c_arr, self.size)

        self.bid_volume = RawArray(self._book_c_arr, self.size)
        self.ask_volume = RawArray(self._book_c_arr, self.size)

        self.bid_n_orders = RawArray(self._order_c_arr, self.size)
        self.ask_n_orders = RawArray(self._order_c_arr, self.size)

    def _at(self, index: int) -> OrderBookPointer:
        ptr = OrderBookPointer(
            dtype=ctypes.pointer(self.dtype),
            ticker=ctypes.cast(ctypes.addressof(self.ticker) + index * ctypes.sizeof(self._ticker_c_arr), ctypes.POINTER(self._ticker_c_arr)),
            timestamp=ctypes.cast(ctypes.addressof(self.timestamp) + index * ctypes.sizeof(ctypes.c_double), ctypes.POINTER(ctypes.c_double)),
            max_level=self.max_level
        )

        ptr._ptr_bid_price = ctypes.cast(ctypes.addressof(self.bid_price) + index * ctypes.sizeof(self._book_c_arr), ctypes.POINTER(self._book_c_arr))
        ptr._ptr_ask_price = ctypes.cast(ctypes.addressof(self.ask_price) + index * ctypes.sizeof(self._book_c_arr), ctypes.POINTER(self._book_c_arr))

        ptr._ptr_bid_volume = ctypes.cast(ctypes.addressof(self.bid_volume) + index * ctypes.sizeof(self._book_c_arr), ctypes.POINTER(self._book_c_arr))
        ptr._ptr_ask_volume = ctypes.cast(ctypes.addressof(self.ask_volume) + index * ctypes.sizeof(self._book_c_arr), ctypes.POINTER(self._book_c_arr))

        ptr._ptr_bid_n_orders = ctypes.cast(ctypes.addressof(self.bid_n_orders) + index * ctypes.sizeof(self._order_c_arr), ctypes.POINTER(self._order_c_arr))
        ptr._ptr_ask_n_orders = ctypes.cast(ctypes.addressof(self.ask_n_orders) + index * ctypes.sizeof(self._order_c_arr), ctypes.POINTER(self._order_c_arr))

        return ptr


class BarDataPointer(MarketDataPointer):
    def __init__(self, **kwargs):
        super().__init__(
            dtype=kwargs.get('dtype'),
            ticker=kwargs.get('ticker'),
            timestamp=kwargs.get('timestamp')
        )

        self.data = kwargs.get('data')

    def update(self, market_data: BarData, encoding='utf-8'):
        super().update(market_data, encoding=encoding)

        data = [
            market_data['start_timestamp'] if 'start_timestamp' in market_data else math.nan,
            market_data['bar_span'] if 'bar_span' in market_data else math.nan,
            market_data['high_price'],
            market_data['low_price'],
            market_data['open_price'],
            market_data['close_price'],
            market_data['volume'],
            market_data['notional'],
            float(market_data['trade_count'])
        ]

        self.data.contents[:] = data

    def to_market_data(self, encoding='utf-8') -> BarData:
        data = self.data.contents[:]

        (start_timestamp, bar_span,
         high_price, low_price, open_price, close_price,
         volume, notional,
         trade_count) = data

        bar_data = BarData(
            ticker=self._ptr_ticker.contents.value.decode(encoding),
            timestamp=self._ptr_timestamp.contents.value,
            start_timestamp=start_timestamp if math.isfinite(start_timestamp) else None,
            bar_span=bar_span if math.isfinite(bar_span) else None,
            high_price=high_price,
            low_price=low_price,
            open_price=open_price,
            close_price=close_price,
            volume=volume,
            notional=notional,
            trade_count=int(trade_count)
        )

        return bar_data


class BarDataBuffer(MarketDataMemoryBuffer):
    _c_data_arr = ctypes.c_double * 9

    def __init__(self, size: int, **kwargs):
        super().__init__(size=size, **kwargs)
        # the data should store the info in following orders
        # start_timestamp, bar_span
        # high_price, low_price, open_price, close_price
        # volume, notional
        # trade_count
        self.data = RawArray(self._c_data_arr, self.size)

        # self.start_timestamp = RawArray(c_double)
        # self.bar_span = RawArray(c_double)
        # self.high_price = RawArray(c_double)
        # self.low_price = RawArray(c_double)
        # self.open_price = RawArray(c_double)
        # self.close_price = RawArray(c_double)
        # self.volume = RawArray(c_double)
        # self.notional = RawArray(c_double)
        # self.trade_count = RawArray(ctypes.c_int8)

    def _at(self, index: int) -> BarDataPointer:
        ptr = BarDataPointer(
            dtype=ctypes.pointer(self.dtype),
            ticker=ctypes.cast(ctypes.addressof(self.ticker) + index * ctypes.sizeof(self._ticker_c_arr), ctypes.POINTER(self._ticker_c_arr)),
            timestamp=ctypes.cast(ctypes.addressof(self.timestamp) + index * ctypes.sizeof(ctypes.c_double), ctypes.POINTER(ctypes.c_double)),
            data=ctypes.cast(ctypes.addressof(self.data) + index * ctypes.sizeof(self._c_data_arr), ctypes.POINTER(self._c_data_arr))
        )

        return ptr


class TickDataPointer(MarketDataPointer):
    def __init__(self, **kwargs):
        super().__init__(
            dtype=kwargs.get('dtype'),
            ticker=kwargs.get('ticker'),
            timestamp=kwargs.get('timestamp')
        )

        self.data = kwargs.get('data')

    def update(self, market_data: TickData, encoding='utf-8'):
        super().update(market_data, encoding=encoding)

        data = [
            market_data.last_price,
            market_data['bid_price'] if 'bid_price' in market_data else math.nan,
            market_data['bid_volume'] if 'bid_volume' in market_data else math.nan,
            market_data['ask_price'] if 'ask_price' in market_data else math.nan,
            market_data['ask_volume'] if 'ask_volume' in market_data else math.nan,
            market_data['total_traded_volume'],
            market_data['total_traded_notional'],
            float(market_data['total_trade_count'])
        ]

        self.data.contents[:] = data

    def to_market_data(self, encoding='utf-8') -> TickData:
        data = self.data.contents[:]

        (last_price,
         bid_price, bid_volume,
         ask_price, ask_volume,
         total_traded_volume, total_traded_notional,
         total_trade_count) = data

        tick_data = TickData(
            ticker=self._ptr_ticker.contents.value.decode(encoding),
            timestamp=self._ptr_timestamp.contents.value,
            last_price=last_price,
            bid_price=bid_price,
            bid_volume=bid_volume,
            ask_price=ask_price,
            ask_volume=ask_volume,
            order_book=None,
            total_traded_volume=total_traded_volume,
            total_traded_notional=total_traded_notional,
            total_trade_count=int(total_trade_count),
        )

        return tick_data


class TickDataBuffer(MarketDataMemoryBuffer):
    _c_data_arr = ctypes.c_double * 8

    def __init__(self, size: int, **kwargs):
        super().__init__(size=size, **kwargs)

        self.data = RawArray(self._dtype_c_arr, self.size)

    def _at(self, index: int) -> TickDataPointer:
        ptr = TickDataPointer(
            dtype=ctypes.pointer(self.dtype),
            ticker=ctypes.cast(ctypes.addressof(self.ticker) + index * ctypes.sizeof(self._ticker_c_arr), ctypes.POINTER(self._ticker_c_arr)),
            timestamp=ctypes.cast(ctypes.addressof(self.timestamp) + index * ctypes.sizeof(ctypes.c_double), ctypes.POINTER(ctypes.c_double)),
            data=ctypes.cast(ctypes.addressof(self.data) + index * ctypes.sizeof(self._c_data_arr), ctypes.POINTER(self._c_data_arr))
        )

        return ptr


class TransactionDataPointer(MarketDataPointer):
    def __init__(self, **kwargs):
        super().__init__(
            dtype=kwargs.get('dtype'),
            ticker=kwargs.get('ticker'),
            timestamp=kwargs.get('timestamp')
        )

        self.data = kwargs.get('data')
        self.id_type = kwargs.get('id_type')
        self.transaction_id = kwargs.get('transaction_id')
        self.buy_id = kwargs.get('buy_id')
        self.sell_id = kwargs.get('sell_id')

    def update(self, market_data: TransactionData | TradeData, encoding='utf-8'):
        super().update(market_data, encoding=encoding)

        data = [
            market_data['price'],
            market_data['volume'],
            market_data['side'],
            market_data['multiplier'] if 'multiplier' in market_data else math.nan,
            market_data['notional'] if 'notional' in market_data else math.nan,
        ]

        for i, (id_name, id_ptr) in enumerate(zip(['transaction_id', 'buy_id', 'sell_id'], [self.transaction_id, self.buy_id, self.sell_id])):
            if id_name not in market_data:
                id_ptr.contents[:] = b''.ljust(ctypes.sizeof(id_ptr.contents), b'\x00')
                self.id_type.contents[i] = 0
            elif isinstance(_id := market_data[id_name], str):
                if len(_id) > 16:
                    raise ValueError(f'{id_name} too long, expect 16 bytes.')
                id_ptr.contents[:] = _id.encode(encoding).ljust(ctypes.sizeof(id_ptr.contents), b'\x00')
                self.id_type.contents[i] = 1
            elif isinstance(_id, int):
                id_ptr.contents[:] = _id.to_bytes(length=ctypes.sizeof(id_ptr.contents), byteorder='big')
                self.id_type.contents[i] = 0
            else:
                raise TypeError(f'Invalid {id_name} {_id} type: {type(_id)}, expect int or str.')

        self.data.contents[:] = data

    def to_market_data(self, encoding='utf-8') -> TransactionData | TradeData:
        data = self.data.contents[:]
        data_id_type = self.id_type.contents[:]

        price, volume, side, multiplier, notional = data
        transaction_id_type, buy_id_type, sell_id_type = data_id_type

        if (dtype := self._ptr_dtype.contents.value) == b'TransactionData':
            constructor = TransactionData
        elif dtype == b'TradeData':
            constructor = TradeData
        else:
            raise NotImplementedError(f'Constructor for market data {dtype} not implemented.')

        transaction_data = constructor(
            ticker=self._ptr_ticker.contents.value.decode(encoding),
            timestamp=self._ptr_timestamp.contents.value,
            price=price,
            volume=volume,
            side=int(side),
            multiplier=None if np.isnan(multiplier) else multiplier,
            notional=None if np.isnan(notional) else notional,
            transaction_id=self.transaction_id.contents.value.decode(encoding) if transaction_id_type else int.from_bytes(self.transaction_id.contents),
            buy_id=self.buy_id.contents.value.decode(encoding) if buy_id_type else int.from_bytes(self.buy_id.contents),
            sell_id=self.sell_id.contents.value.decode(encoding) if sell_id_type else int.from_bytes(self.sell_id.contents)
        )

        return transaction_data


class TransactionDataBuffer(MarketDataMemoryBuffer):
    _c_data_arr = ctypes.c_double * 5
    _c_id_type_arr = ctypes.c_int * 3

    def __init__(self, size: int, **kwargs):
        super().__init__(size=size, **kwargs)
        self._id_size = kwargs.get('id_size', 16)
        self._c_id_arr = ctypes.c_char * self._id_size

        self.data = RawArray(self._c_data_arr, self.size)
        self.data_id_type = RawArray(self._c_id_type_arr, self.size)
        self.transaction_id = RawArray(self._c_id_arr, self.size)
        self.buy_id = RawArray(self._c_id_arr, self.size)
        self.sell_id = RawArray(self._c_id_arr, self.size)

    def _at(self, index: int) -> TransactionDataPointer:
        ptr = TransactionDataPointer(
            dtype=ctypes.pointer(self.dtype),
            ticker=ctypes.cast(ctypes.addressof(self.ticker) + index * ctypes.sizeof(self._ticker_c_arr), ctypes.POINTER(self._ticker_c_arr)),
            timestamp=ctypes.cast(ctypes.addressof(self.timestamp) + index * ctypes.sizeof(ctypes.c_double), ctypes.POINTER(ctypes.c_double)),
            data=ctypes.cast(ctypes.addressof(self.data) + index * ctypes.sizeof(self._c_data_arr), ctypes.POINTER(self._c_data_arr)),
            id_type=ctypes.cast(ctypes.addressof(self.data_id_type) + index * ctypes.sizeof(self._c_id_type_arr), ctypes.POINTER(self._c_id_type_arr)),
            transaction_id=ctypes.cast(ctypes.addressof(self.transaction_id) + index * ctypes.sizeof(self._c_id_arr), ctypes.POINTER(self._c_id_arr)),
            buy_id=ctypes.cast(ctypes.addressof(self.buy_id) + index * ctypes.sizeof(self._c_id_arr), ctypes.POINTER(self._c_id_arr)),
            sell_id=ctypes.cast(ctypes.addressof(self.sell_id) + index * ctypes.sizeof(self._c_id_arr), ctypes.POINTER(self._c_id_arr))
        )

        return ptr


__all__ = [
    'MarketDataPointer', 'MarketDataMemoryBuffer',
    'OrderBookPointer', 'OrderBookBuffer',
    'BarDataPointer', 'BarDataBuffer',
    'TickDataPointer', 'TickDataBuffer',
    'TransactionDataPointer', 'TransactionDataBuffer'
]
