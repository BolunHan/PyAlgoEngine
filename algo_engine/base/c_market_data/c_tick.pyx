# cython: language_level=3
from collections.abc import Sequence

cimport cython
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.math cimport NAN
from libc.stdint cimport uint32_t
from libc.stdlib cimport qsort
from libc.string cimport memcpy, memset

from .c_market_data cimport _MarketDataVirtualBase, TICKER_SIZE, _MarketDataBuffer, _TickDataBuffer, _TickDataLiteBuffer, _OrderBookBuffer, _OrderBookEntry, compare_entries_bid, compare_entries_ask, Side, DataType, BOOK_SIZE


@cython.freelist(128)
cdef class TickDataLite:
    def __cinit__(self):
        self._data_ptr = <_MarketDataBuffer*> &self._data
        self._data_addr = <uintptr_t> self._data_ptr

    def __init__(
            self,
            str ticker,
            double timestamp,
            double last_price,
            double bid_price,
            double bid_volume,
            double ask_price,
            double ask_volume,
            double prev_close=NAN,
            double total_traded_volume=0.0,
            double total_traded_notional=0.0,
            uint32_t total_trade_count=0,
            **kwargs
    ):
        # Initialize base class fields
        cdef bytes ticker_bytes = ticker.encode('utf-8')
        cdef size_t ticker_len = min(len(ticker_bytes), TICKER_SIZE - 1)
        memcpy(<void*> &self._data.ticker, <const char*> ticker_bytes, ticker_len)
        self._data.timestamp = timestamp
        self._data.dtype = DataType.DTYPE_TICK_LITE
        if kwargs: self.__dict__.update(kwargs)

        # Set other fields
        self._data.last_price = last_price
        self._data.bid_price = bid_price
        self._data.bid_volume = bid_volume
        self._data.ask_price = ask_price
        self._data.ask_volume = ask_volume
        self._data.prev_close = prev_close
        self._data.total_traded_volume = total_traded_volume
        self._data.total_traded_notional = total_traded_notional
        self._data.total_trade_count = total_trade_count

    def __repr__(self):
        return f'<TickDataLite>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker}, bid={self.bid_price}, ask={self.ask_price})'

    def __reduce__(self):
        return self.__class__.from_bytes, (self.to_bytes(),), self.__dict__

    def __setstate__(self, state):
        if state:
            self.__dict__.update(state)

    def __copy__(self):
        cdef TickDataLite instance = TickDataLite.__new__(TickDataLite)
        memcpy(<void*> &instance._data, <const char*> &self._data, sizeof(_TickDataLiteBuffer))
        return instance

    @classmethod
    def buffer_size(cls):
        return sizeof(_TickDataLiteBuffer)

    cdef bytes c_to_bytes(self):
        return PyBytes_FromStringAndSize(<char*> &self._data, sizeof(self._data))

    @staticmethod
    cdef TickDataLite c_from_bytes(bytes data):
        cdef TickDataLite instance = TickDataLite.__new__(TickDataLite)
        memcpy(<void*> &instance._data, <const char*> data, sizeof(_TickDataLiteBuffer))
        return instance

    def to_bytes(self) -> bytes:
        return self.c_to_bytes()

    @classmethod
    def from_bytes(cls, bytes data):
        return TickDataLite.c_from_bytes(data)

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
    def last_price(self):
        return self._data.last_price

    @property
    def bid_price(self):
        return self._data.bid_price

    @property
    def bid_volume(self):
        return self._data.bid_volume

    @property
    def ask_price(self):
        return self._data.ask_price

    @property
    def ask_volume(self):
        return self._data.ask_volume

    @property
    def prev_close(self):
        return self._data.prev_close

    @property
    def total_traded_volume(self):
        return self._data.total_traded_volume

    @property
    def total_traded_notional(self):
        return self._data.total_traded_notional

    @property
    def total_trade_count(self):
        return self._data.total_trade_count

    @property
    def mid_price(self):
        return (self._data.bid_price + self._data.ask_price) / 2.0

    @property
    def spread(self):
        return self._data.ask_price - self._data.bid_price

    @property
    def market_price(self):
        return self.last_price


cdef class OrderBook:
    def __init__(self, uint8_t side = Side.SIDE_UNKNOWN, price: Sequence[float] = None, volume: Sequence[float] = None, n_orders: Sequence[int] = None, is_sorted: bool = False):
        self.sorted = is_sorted
        self._owner = True

        if side == Side.SIDE_UNKNOWN or side == Side.SIDE_BID or side == Side.SIDE_ASK:
            self.side = side
        else:
            raise ValueError(f'Invalid side {side}, expect [Side.SIDE_BID, Side.SIDE_ASK]')

        # Allocate memory for _data
        self._data = <_OrderBookBuffer*> PyMem_Malloc(sizeof(_OrderBookBuffer))
        memset(self._data, 0, sizeof(_OrderBookBuffer))

        # If prices, volumes, or n_orders are provided, populate them
        if price is None and self.volume is None:
            return

        n_entries = len(price)
        assert n_entries  == len(volume)

        if n_orders is None:
            for i in range(min(n_entries, BOOK_SIZE)):
                self._data.entries[i].price = price[i]
                self._data.entries[i].volume = volume[i]
                self._data.entries[i].n_orders = 1
        else:
            assert n_entries == len(n_orders)
            for i in range(min(n_entries, BOOK_SIZE)):
                self._data.entries[i].price = price[i]
                self._data.entries[i].volume = volume[i]
                self._data.entries[i].n_orders = n_orders[i]

        self.c_sort()

    def __dealloc__(self):
        """
        Free allocated memory if this instance owns it.
        """
        if self._data is not NULL and self._owner:
            PyMem_Free(self._data)
            self._data = NULL

    def __getbuffer__(self, Py_buffer *view, int flags):
        # Fill in the Py_buffer structure
        view.buf = <void*> self._data.entries
        view.obj = self
        view.len = BOOK_SIZE * sizeof(_OrderBookEntry)
        view.readonly = 0  # Make the buffer writable
        view.itemsize = sizeof(double)  # Each field (price, volume, n_orders) is a double
        view.format = NULL  # 'd' for double
        view.ndim = 2  # 2D buffer

        # Allocate memory for shape and strides
        view.shape = <Py_ssize_t*> PyMem_Malloc(2 * sizeof(Py_ssize_t))
        view.strides = <Py_ssize_t*> PyMem_Malloc(2 * sizeof(Py_ssize_t))

        if view.shape == NULL or view.strides == NULL:
            # PyMem_Free(view.shape)
            # PyMem_Free(view.strides)
            raise MemoryError("Failed to allocate memory for shape and strides")

        # Set shape and strides
        view.shape[0] = BOOK_SIZE  # Number of entries
        view.shape[1] = 3  # Each entry has 3 fields (price, volume, n_orders)

        view.strides[0] = sizeof(_OrderBookEntry)  # Stride between entries
        view.strides[1] = sizeof(double)  # Stride between fields within an entry

        view.suboffsets = NULL
        view.internal = NULL

    def __releasebuffer__(self, Py_buffer *view):
        # Free allocated memory for shape and strides
        if view.shape != NULL:
            PyMem_Free(view.shape)
            view.shape = NULL
        if view.strides != NULL:
            PyMem_Free(view.strides)
            view.strides = NULL

    def __iter__(self):
        self.c_sort()
        self._iter_index = 0
        return self

    def __next__(self) -> tuple[float, float, int]:
        while self._iter_index < BOOK_SIZE:
            entry = self._data.entries[self._iter_index]
            self._iter_index += 1
            if entry.n_orders:
                return entry.price, entry.volume, entry.n_orders
        raise StopIteration

    cdef void c_sort(self):
        if self.sorted:
            return  # Skip sorting if already sorted

        if self.side == Side.SIDE_UNKNOWN:
            return  # No way to know the sorting orders

        if self.side == Side.SIDE_BID:
            qsort(self._data.entries, BOOK_SIZE, sizeof(_OrderBookEntry), compare_entries_bid)
        elif self.side == Side.SIDE_ASK:
            qsort(self._data.entries, BOOK_SIZE, sizeof(_OrderBookEntry), compare_entries_ask)
        else:
            raise ValueError(f'Invalid TransactionSide {self.side}.')

        self.sorted = True

    cdef double c_loc_volume(self, double p0, double p1):
        self.c_sort()

        cdef double volume = 0.

        for i in range(BOOK_SIZE):
            if not self._data.entries[i].n_orders:
                break

            if p0 <= self._data.entries[i].price < p1:
                volume += self._data.entries[i].volume

        return volume

    cdef bytes c_to_bytes(self):
        """
        Convert the transaction data to bytes.
        """
        if self._data == NULL:
            raise ValueError("Cannot convert uninitialized data to bytes")

        return PyBytes_FromStringAndSize(<char*>self._data, sizeof(_OrderBookBuffer))

    @staticmethod
    cdef OrderBook c_from_bytes(bytes data, uint8_t side = Side.SIDE_UNKNOWN):
        """
        Create a new instance from bytes.
        Creates a copy of the data, so the instance owns the memory.
        """
        cdef OrderBook instance = OrderBook.__new__(OrderBook)
        cdef const char* data_ptr = <const char*>data

        instance._owner = True
        instance._data = <_OrderBookBuffer*> PyMem_Malloc(sizeof(_OrderBookBuffer))

        if instance._data == NULL:
            raise MemoryError("Failed to allocate memory for TransactionData")

        memcpy(instance._data, data_ptr, sizeof(_OrderBookBuffer))

        if side is not None:
            instance.side = side

        return instance

    def at_price(self, price: float) -> tuple[float, float, float]:
        for i in range(BOOK_SIZE):
            if self._data.entries[i].price == price and self._data.entries[i].n_orders:
                return self._data.entries[i].price, self._data.entries[i].volume, self._data.entries[i].n_orders

        raise IndexError(f'price {price} not found!')

    def at_level(self, index: int) -> tuple[float, float, float]:
        self.c_sort()

        if 0 <= index < BOOK_SIZE and self._data.entries[index].n_orders:
            return self._data.entries[index].price, self._data.entries[index].volume, self._data.entries[index].n_orders

        raise IndexError(f'level {index} not found!')

    # Method to create OrderBook instance from an existing buffer (without owning the data)
    @classmethod
    def from_buffer(cls, const unsigned char[:] buffer, side: int | object = None):
        cdef OrderBook instance = cls.__new__(cls)

        # Point to the buffer data
        instance._data = <_OrderBookBuffer*> &buffer[0]
        instance._owner = False

        if side is not None:
            instance.side = side

        return instance

    def loc_volume(self, double p0, double p1) -> float:
        return self.c_loc_volume(p0=p0, p1=p1)

    def sort(self):
        return self.c_sort()

    def to_bytes(self) -> bytes:
        return self.c_to_bytes()

    @classmethod
    def from_bytes(bytes data, uint8_t side = Side.SIDE_UNKNOWN):
        return OrderBook.c_from_bytes(data=data, side=side)

    # Numpy conversion
    def to_numpy(self):
        import numpy as np
        dtype = [('price', np.float64), ('volume', np.float64), ('n_orders', np.uint64)]
        arr = np.ndarray(shape=(BOOK_SIZE,), dtype=dtype, buffer=self)
        return arr

    @property
    def price(self):
        return [self._data.entries[i].price for i in range(BOOK_SIZE) if self._data.entries[i].n_orders]

    @property
    def volume(self):
        return [self._data.entries[i].volume for i in range(BOOK_SIZE) if self._data.entries[i].n_orders]

    @property
    def n_orders(self):
        return [self._data.entries[i].n_orders for i in range(BOOK_SIZE) if self._data.entries[i].n_orders]


cdef class TickData:
    def __cinit__(self):
        self._data_ptr = <_MarketDataBuffer*> &self._data
        self._data_addr = <uintptr_t> self._data_ptr

    def __init__(
            self,
            str ticker,
            double timestamp,
            double last_price,
            double total_traded_volume=0.0,
            double total_traded_notional=0.0,
            uint32_t total_trade_count=0,
            **kwargs
    ):
        # Initialize MarketData base
        cdef bytes ticker_bytes = ticker.encode('utf-8')
        cdef size_t ticker_len = min(len(ticker_bytes), TICKER_SIZE - 1)
        memcpy(<void*> &self._data.lite.ticker, <const char*> ticker_bytes, ticker_len)
        self._data.lite.timestamp = timestamp
        self._data.lite.dtype = DataType.DTYPE_TICK
        # if kwargs: self.__dict__.update(kwargs)

        # Set TickDataLite fields
        self._data.lite.last_price = last_price
        self._data.lite.bid_price = kwargs.get('bid_price_1', NAN)
        self._data.lite.bid_volume = kwargs.get('bid_volume_1', NAN)
        self._data.lite.ask_price = kwargs.get('ask_price_1', NAN)
        self._data.lite.ask_volume = kwargs.get('ask_volume_1', NAN)
        self._data.lite.prev_close = kwargs.get('prev_close', NAN)
        self._data.lite.total_traded_volume = total_traded_volume
        self._data.lite.total_traded_notional = total_traded_notional
        self._data.lite.total_trade_count = total_trade_count

        # Initialize bid and ask books
        self._init_order_book()

        # Parse kwargs to initialize the order books
        if kwargs:
            self.parse(kwargs)

    def __repr__(self):
        return f'<TickData>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker}, bid={self.bid_price}, ask={self.ask_price})'

    def __reduce__(self):
        return self.__class__.from_bytes, (self.to_bytes(),), self.__dict__

    def __setstate__(self, state):
        if state:
            self.__dict__.update(state)

    def __copy__(self):
        cdef TickData instance = TickData.__new__(TickData)
        memcpy(<void*> &instance._data, <const char*> &self._data, sizeof(_TickDataBuffer))
        instance._init_order_book()
        return instance

    cdef _init_order_book(self):
        # Initialize bid and ask books
        self._bid_book = OrderBook.__new__(OrderBook)
        self._bid_book._data = &self._data.bid
        self._bid_book.side = Side.SIDE_BID

        self._ask_book = OrderBook.__new__(OrderBook)
        self._ask_book._data = &self._data.ask
        self._ask_book.side = Side.SIDE_ASK

    cpdef void parse(self, dict kwargs):
        """
        Parse keyword arguments to initialize bid and ask books.

        Parameters:
        -----------
        kwargs : dict
            Keyword arguments in the format bid_price_1=10, ask_volume_3=10.23, bid_orders_3=16
        """
        cdef int level
        cdef str key_type, book_type
        cdef object value
        cdef bint[BOOK_SIZE] bid_init
        cdef bint[BOOK_SIZE] ask_init

        memset(&bid_init[0], 0, BOOK_SIZE * sizeof(bint))
        memset(&ask_init[0], 0, BOOK_SIZE * sizeof(bint))

        # Process each keyword argument
        for key, value in kwargs.items():
            parts = key.split('_')
            if len(parts) != 3:
                self.__dict__[key] = value
                continue

            book_type = parts[0]  # 'bid' or 'ask'
            key_type = parts[1]   # 'price', 'volume', or 'orders'

            # Check if there's a level number
            if not parts[2].isdigit():
                self.__dict__[key] = value
                continue

            level = int(parts[2]) - 1  # Convert to 0-based index

            # Skip if level is out of range
            if level < 0 or level >= BOOK_SIZE:
                self.__dict__[key] = value
                continue

            # Store in appropriate book data dictionary
            if book_type == 'bid':
                if key_type == 'price':
                    self._data.bid.entries[level].price = value
                elif key_type == 'volume':
                    self._data.bid.entries[level].volume = value
                elif key_type == 'orders':
                    self._data.bid.entries[level].n_orders = int(value)
                else:
                    self.__dict__[key] = value
                    continue
                bid_init[level] = True
            elif book_type == 'ask':
                if key_type == 'price':
                    self._data.ask.entries[level].price = value
                elif key_type == 'volume':
                    self._data.ask.entries[level].volume = value
                elif key_type == 'orders':
                    self._data.ask.entries[level].n_orders = int(value)
                else:
                    self.__dict__[key] = value
                    continue
                ask_init[level] = True
            else:
                self.__dict__[key] = value

        # Validate and fix the books
        for i in range(BOOK_SIZE):
            if bid_init[i] and not self._data.bid.entries[i].n_orders:
                self._data.bid.entries[i].n_orders = 1

            if ask_init[i] and not self._data.ask.entries[i].n_orders:
                self._data.ask.entries[i].n_orders = 1

        # Sort the books
        self._bid_book.sort()
        self._ask_book.sort()

    @classmethod
    def buffer_size(cls):
        return sizeof(_TickDataBuffer)

    cdef bytes c_to_bytes(self):
        return PyBytes_FromStringAndSize(<char*> &self._data, sizeof(self._data))

    @staticmethod
    cdef TickData c_from_bytes(bytes data):
        cdef TickData instance = TickData.__new__(TickData)
        memcpy(<void*> &instance._data, <const char*> data, sizeof(_TickDataBuffer))
        instance._init_order_book()
        return instance

    def to_bytes(self) -> bytes:
        return self.c_to_bytes()

    @classmethod
    def from_bytes(cls, bytes data):
        return TickData.c_from_bytes(data)

    cpdef TickDataLite lite(self):
        cdef TickDataLite instance = TickDataLite.__new__(TickDataLite)
        memcpy(<void*> &instance._data, <const char*> &self._data.lite, sizeof(_TickDataLiteBuffer))
        return instance

    @property
    def ticker(self) -> str:
        return self._data.lite.ticker.decode('utf-8')

    @property
    def timestamp(self) -> float:
        return self._data.lite.timestamp

    @property
    def dtype(self) -> int:
        return self._data.lite.dtype

    @property
    def topic(self) -> str:
        ticker_str = self._data.lite.ticker.decode('utf-8')
        return f'{ticker_str}.{self.__class__.__name__}'

    @property
    def market_time(self) :
        return _MarketDataVirtualBase.c_to_dt(self._data.lite.timestamp)

    @property
    def bid(self) -> OrderBook:
        """Get the bid book."""
        return self._bid_book

    @property
    def ask(self) -> OrderBook:
        """Get the ask book."""
        return self._ask_book

    @property
    def best_ask_price(self) -> float:
        return self.ask.price[0]

    @property
    def best_bid_price(self) -> float:
        return self.bid.price[0]

    @property
    def best_ask_volume(self) -> float:
        return self.ask.volume[0]

    @property
    def best_bid_volume(self) -> float:
        return self.bid.volume[0]

    @property
    def last_price(self):
        return self._data.lite.last_price

    @property
    def bid_price(self):
        return self._data.lite.bid_price

    @property
    def bid_volume(self):
        return self._data.lite.bid_volume

    @property
    def ask_price(self):
        return self._data.lite.ask_price

    @property
    def ask_volume(self):
        return self._data.lite.ask_volume

    @property
    def prev_close(self):
        return self._data.lite.prev_close

    @property
    def total_traded_volume(self):
        return self._data.lite.total_traded_volume

    @property
    def total_traded_notional(self):
        return self._data.lite.total_traded_notional

    @property
    def total_trade_count(self):
        return self._data.lite.total_trade_count

    @property
    def mid_price(self):
        return (self._data.lite.bid_price + self._data.lite.ask_price) / 2.0

    @property
    def spread(self):
        return self._data.lite.ask_price - self._data.lite.bid_price

    @property
    def market_price(self):
        return self.last_price
