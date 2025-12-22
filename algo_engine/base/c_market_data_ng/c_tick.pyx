from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.unicode cimport PyUnicode_AsUTF8
from libc.math cimport NAN
from libc.stdint cimport uint64_t, uintptr_t
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy

from .c_market_data cimport (
    SHM_ALLOCATOR, HEAP_ALLOCATOR,
    md_data_type, md_direction, MD_CFG_LOCKED, MD_CFG_SHARED, MD_CFG_FREELIST, MD_CFG_BOOK_SIZE,
    md_orderbook_entry, c_md_orderbook_sort,
    c_init_buffer, c_md_orderbook_new, c_md_orderbook_free
)
from .c_transaction import TransactionDirection, TransactionOffset


cdef class TickDataLite(MarketData):
    def __init__(
            self,
            *,
            str ticker,
            double timestamp,
            double last_price,
            double bid_price,
            double bid_volume,
            double ask_price,
            double ask_volume,
            double open_price=NAN,
            double prev_close=NAN,
            double total_traded_volume=0.0,
            double total_traded_notional=0.0,
            uint64_t total_trade_count=0,
            **kwargs
    ):
        self.header = c_init_buffer(
            md_data_type.DTYPE_TICK_LITE,
            PyUnicode_AsUTF8(ticker),
            timestamp
        )

        # Set other fields
        self.header.tick_data_lite.last_price = last_price
        self.header.tick_data_lite.bid_price = bid_price
        self.header.tick_data_lite.bid_volume = bid_volume
        self.header.tick_data_lite.ask_price = ask_price
        self.header.tick_data_lite.ask_volume = ask_volume
        self.header.tick_data_lite.open_price = open_price
        self.header.tick_data_lite.prev_close = prev_close
        self.header.tick_data_lite.total_traded_volume = total_traded_volume
        self.header.tick_data_lite.total_traded_notional = total_traded_notional
        self.header.tick_data_lite.total_trade_count = total_trade_count

        self.data_addr = <uintptr_t> self.header
        self.owner = True

        if kwargs:
            self.__dict__.update(kwargs)

    def __repr__(self):
        if not self.header:
            return f"<{self.__class__.__name__}>(Uninitialized)"
        return f'<{self.__class__.__name__}>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker}, bid={self.bid_price}, ask={self.ask_price})'

    property last_price:
        def __get__(self):
            return self.header.tick_data_lite.last_price

    property bid_price:
        def __get__(self):
            return self.header.tick_data_lite.bid_price

    property bid_volume:
        def __get__(self):
            return self.header.tick_data_lite.bid_volume

    property ask_price:
        def __get__(self):
            return self.header.tick_data_lite.ask_price

    property ask_volume:
        def __get__(self):
            return self.header.tick_data_lite.ask_volume

    property open_price:
        def __get__(self):
            return self.header.tick_data_lite.open_price

    property prev_close:
        def __get__(self):
            return self.header.tick_data_lite.prev_close

    property total_traded_volume:
        def __get__(self):
            return self.header.tick_data_lite.total_traded_volume

    property total_traded_notional:
        def __get__(self):
            return self.header.tick_data_lite.total_traded_notional

    property total_trade_count:
        def __get__(self):
            return self.header.tick_data_lite.total_trade_count

    property mid_price:
        def __get__(self):
            cdef double bid_price = self.header.tick_data_lite.bid_price
            cdef double ask_price = self.header.tick_data_lite.ask_price
            return (bid_price + ask_price) / 2.0

    property spread:
        def __get__(self):
            cdef double bid_price = self.header.tick_data_lite.bid_price
            cdef double ask_price = self.header.tick_data_lite.ask_price
            return ask_price - bid_price


cdef class OrderBook:
    def __cinit__(
            self,
            *,
            md_direction direction = md_direction.DIRECTION_UNKNOWN,
            object price=None,
            object volume=None,
            object n_orders=None,
            bint is_sorted=False,
            **kwargs
    ):
        if direction == md_direction.DIRECTION_UNKNOWN:
            return
        elif direction == md_direction.DIRECTION_LONG or direction == md_direction.DIRECTION_SHORT:
            pass
        else:
            raise ValueError(f'Invalid Direction {direction}. expecting {TransactionDirection.DIRECTION_SHORT} or {TransactionDirection.DIRECTION_LONG}')

        cdef size_t book_size = MD_CFG_BOOK_SIZE

        if MD_CFG_SHARED:
            self.header = c_md_orderbook_new(book_size, SHM_ALLOCATOR, NULL, <int> MD_CFG_LOCKED)
        elif MD_CFG_FREELIST:
            self.header = c_md_orderbook_new(book_size, NULL, HEAP_ALLOCATOR, <int> MD_CFG_LOCKED)
        else:
            self.header = c_md_orderbook_new(book_size, NULL, NULL, 0)

        if not self.header:
            raise MemoryError('Failed to allocate memory for OrderBook')

        self.header.direction = direction
        self.header.size = 0
        self.header.sorted = is_sorted

        self.owner = True

        # If prices, volumes, or n_orders are provided, populate them
        if price is None and self.volume is None:
            self.header.sorted = True
            return

        cdef size_t i
        cdef size_t len_price = 0 if price is None else len(price)
        cdef size_t len_volume = 0 if volume is None else len(volume)
        cdef size_t len_n_orders = 0 if n_orders is None else len(n_orders)
        cdef md_orderbook_entry* entry

        for i in range(self.header.capacity):
            entry = self.header.entries + i

            # Update price if available
            if i < len_price:
                entry.price = price[i]
                entry.n_orders = 1
            else:
                entry.price = NAN

            # Update volume if available
            if i < len_volume:
                entry.volume = volume[i]
                entry.n_orders = 1
            else:
                entry.volume = NAN

            # Update n_orders if available
            if i < len_n_orders:
                entry.n_orders = n_orders[i]
            else:
                entry.n_orders = 0

            if entry.n_orders:
                self.header.size += 1

        self.c_sort()

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            c_md_orderbook_free(self.header, <int> MD_CFG_LOCKED)

    def __getbuffer__(self, Py_buffer *view, int flags):
        # Fill in the Py_buffer structure
        view.buf = <void*> self.header.entries
        view.obj = self
        view.len = self.header.capacity * sizeof(md_orderbook_entry)
        view.readonly = 0
        view.itemsize = sizeof(double)
        view.format = NULL
        view.ndim = 2

        # Allocate memory for shape and strides
        view.shape = <Py_ssize_t*> calloc(2, sizeof(Py_ssize_t))
        view.strides = <Py_ssize_t*> calloc(2, sizeof(Py_ssize_t))

        if view.shape == NULL or view.strides == NULL:
            if view.shape:
                free(view.shape)
            if view.strides:
                free(view.strides)
            raise MemoryError("Failed to allocate memory for shape and strides")

        # Set shape and strides
        view.shape[0] = self.header.capacity  # Number of entries
        view.shape[1] = 3  # Each entry has 3 fields (price, volume, n_orders)

        view.strides[0] = sizeof(md_orderbook_entry)  # Stride between entries
        view.strides[1] = sizeof(double)  # Stride between fields within an entry

        view.suboffsets = NULL
        view.internal = NULL

    def __releasebuffer__(self, Py_buffer *view):
        if view.shape:
            free(view.shape)
            view.shape = NULL
        if view.strides:
            free(view.strides)
            view.strides = NULL

    cdef void c_sort(self):
        if not self.header:
            raise RuntimeError('Uninitialized OrderBook cannot be sorted.')

        cdef int ret_code = c_md_orderbook_sort(self.header)

        if ret_code != 0:
            raise RuntimeError(f'OrderBook sorting failed with error code {ret_code}.')

    cdef double c_loc_volume(self, double p0, double p1):
        self.c_sort()

        cdef double volume = 0.0
        cdef md_orderbook_entry* entry
        cdef size_t i

        for i in range(self.header.size):
            entry = self.header.entries + i
            if not entry.n_orders:
                break

            if p0 <= entry.price < p1:
                volume += entry.volume

        return volume

    cdef bytes c_to_bytes(self):
        if not self.header:
            raise ValueError("Cannot convert uninitialized data to bytes")
        cdef size_t buffer_size = sizeof(md_orderbook) + self.header.capacity * sizeof(md_orderbook_entry);
        return PyBytes_FromStringAndSize(<char*> self.header, buffer_size)

    @staticmethod
    cdef OrderBook c_from_bytes(const char* data):
        cdef OrderBook instance = OrderBook.__new__(OrderBook)
        cdef const md_orderbook* borrowed = <const md_orderbook*> data
        cdef size_t book_size = borrowed.capacity
        cdef size_t buffer_size = sizeof(md_orderbook) + borrowed.capacity * sizeof(md_orderbook_entry);

        cdef md_orderbook* header
        if MD_CFG_SHARED:
            header = c_md_orderbook_new(book_size, SHM_ALLOCATOR, NULL, <int> MD_CFG_LOCKED)
        elif MD_CFG_FREELIST:
            header = c_md_orderbook_new(book_size, NULL, HEAP_ALLOCATOR, <int> MD_CFG_LOCKED)
        else:
            header = c_md_orderbook_new(book_size, NULL, NULL, 0)

        if not header:
            raise MemoryError('Failed to allocate memory for OrderBook')

        memcpy(<void*> header, <void*> borrowed, buffer_size)

        instance.header = header
        instance.owner = True
        return instance

    # === Python Interfaces ===

    def __repr__(self):
        if not self.header:
            return f'<{self.__class__.__name__}>(Uninitialized)'
        cdef str orderbook_type = 'bid' if self.header.direction == md_direction.DIRECTION_LONG \
            else 'ask' if self.header.direction == md_direction.DIRECTION_SHORT \
            else 'unknown'
        return f'<{self.__class__.__name__} {orderbook_type}>(size={self.header.size}, capacity={self.header.capacity}, sorted={self.header.sorted})'

    def __len__(self):
        return self.header.size

    def __iter__(self):
        self.c_sort()
        self.iter_index = 0
        return self

    def __next__(self):
        cdef md_orderbook_entry* entry
        while self.iter_index < self.header.size:
            entry = self.header.entries + self.iter_index
            self.iter_index += 1
            return entry.price, entry.volume, entry.n_orders
        raise StopIteration

    def __getitem__(self, ssize_t idx):
        self.c_sort()
        cdef ssize_t ttl = self.header.size
        if idx >= ttl or idx < -ttl:
            raise IndexError('OrderBook index out of range')
        if idx < 0:
            idx += ttl
        cdef md_orderbook_entry* entry = self.header.entries + idx
        return entry.price, entry.volume, entry.n_orders

    cpdef tuple at_price(self, double price):
        cdef size_t n = self.header.size
        cdef size_t i
        cdef md_orderbook_entry* entry

        for i in range(n):
            entry = self.header.entries + i
            if entry.price == price:
                return entry.price, entry.volume, entry.n_orders

        raise IndexError(f'price {price} not found!')

    cpdef tuple at_level(self, ssize_t idx):
        self.c_sort()
        cdef ssize_t ttl = self.header.size

        if idx >= ttl or idx < -ttl:
            raise IndexError('OrderBook index out of range')
        if idx < 0:
            idx += ttl

        cdef md_orderbook_entry* entry = self.header.entries + idx
        return entry.price, entry.volume, entry.n_orders

    @classmethod
    def from_buffer(cls, const unsigned char[:] buffer):
        cdef OrderBook instance = cls.__new__(cls)
        instance.header = <md_orderbook*> &buffer[0]
        instance.owner = False
        return instance

    def loc_volume(self, double p0, double p1):
        return self.c_loc_volume(p0, p1)

    def sort(self):
        return self.c_sort()

    def to_bytes(self):
        return self.c_to_bytes()

    @classmethod
    def from_bytes(cls, bytes data):
        return OrderBook.c_from_bytes(<const char*> data)

    def to_numpy(self):
        import numpy as np
        dtype = [('price', np.float64), ('volume', np.float64), ('n_orders', np.uint64)]
        arr = np.ndarray(
            shape=(self.header.capacity,),
            dtype=dtype,
            buffer=self
        )
        return arr

    property price:
        def __get__(self):
            if self.header.size == 0:
                return None
            elif self.header.size == self.header.capacity:
                return self.to_numpy()['price']
            else:
                return self.to_numpy()['price'][:self.header.size]

    property volume:
        def __get__(self):
            if self.header.size == 0:
                return None
            elif self.header.size == self.header.capacity:
                return self.to_numpy()['volume']
            else:
                return self.to_numpy()['volume'][:self.header.size]

    property n_orders:
        def __get__(self):
            if self.header.size == 0:
                return None
            elif self.header.size == self.header.capacity:
                return self.to_numpy()['n_orders']
            else:
                return self.to_numpy()['n_orders'][:self.header.size]

    property sorted:
        def __get__(self):
            return self.header.sorted

    property side:
        def __get__(self):
            return TransactionDirection(self.header.direction) | TransactionOffset.OFFSET_ORDER

    property direction:
        def __get__(self):
            return TransactionDirection(self.header.direction)

    property size:
        def __get__(self):
            return self.header.size

    property capacity:
        def __get__(self):
            return self.header.capacity


cdef class TickData(MarketData):
    def __init__(
            self,
            *,
            str ticker,
            double timestamp,
            double last_price,
            double open_price=NAN,
            double prev_close=NAN,
            double total_traded_volume=0.0,
            double total_traded_notional=0.0,
            uint64_t total_trade_count=0,
            double total_bid_volume=0.0,
            double total_ask_volume=0.0,
            double weighted_bid_price=NAN,
            double weighted_ask_price=NAN,
            **kwargs
    ):
        self.header = c_init_buffer(
            md_data_type.DTYPE_TICK,
            PyUnicode_AsUTF8(ticker),
            timestamp
        )

        # Set TickDataLite fields
        self.header.tick_data_full.lite.last_price = last_price
        self.header.tick_data_full.lite.bid_price = kwargs.get('bid_price_1', NAN)
        self.header.tick_data_full.lite.bid_volume = kwargs.get('bid_volume_1', NAN)
        self.header.tick_data_full.lite.ask_price = kwargs.get('ask_price_1', NAN)
        self.header.tick_data_full.lite.ask_volume = kwargs.get('ask_volume_1', NAN)
        self.header.tick_data_full.lite.open_price = open_price
        self.header.tick_data_full.lite.prev_close = prev_close
        self.header.tick_data_full.lite.total_traded_volume = total_traded_volume
        self.header.tick_data_full.lite.total_traded_notional = total_traded_notional
        self.header.tick_data_full.lite.total_trade_count = total_trade_count
        self.header.tick_data_full.total_bid_volume = total_bid_volume
        self.header.tick_data_full.total_ask_volume = total_ask_volume
        self.header.tick_data_full.weighted_bid_price = weighted_bid_price
        self.header.tick_data_full.weighted_ask_price = weighted_ask_price

        self.bid = OrderBook.__new__(OrderBook, direction=md_direction.DIRECTION_LONG, is_sorted=False)
        self.ask = OrderBook.__new__(OrderBook, direction=md_direction.DIRECTION_SHORT, is_sorted=False)
        self.header.tick_data_full.bid = self.bid.header
        self.header.tick_data_full.ask = self.ask.header

        self.data_addr = <uintptr_t> self.header
        self.owner = True

        if kwargs:
            self.parse(kwargs)

    def __repr__(self):
        if not self.header:
            return f"<{self.__class__.__name__}>(Uninitialized)"
        return f'<{self.__class__.__name__}>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker}, bid={self.bid_price}, ask={self.ask_price})'

    def __copy__(self):
        cdef TickData instance = super().__copy__()
        cdef md_orderbook* bid_header = c_md_orderbook_new(self.header.tick_data_full.bid.capacity, NULL, NULL, <int> MD_CFG_LOCKED)
        cdef md_orderbook* ask_header = c_md_orderbook_new(self.header.tick_data_full.ask.capacity, NULL, NULL, <int> MD_CFG_LOCKED)

        memcpy(<void*> bid_header, <void*> self.header.tick_data_full.bid, sizeof(md_orderbook) + self.header.tick_data_full.bid.capacity * sizeof(md_orderbook_entry))
        memcpy(<void*> ask_header, <void*> self.header.tick_data_full.ask, sizeof(md_orderbook) + self.header.tick_data_full.ask.capacity * sizeof(md_orderbook_entry))

        instance.header.tick_data_full.bid = bid_header
        instance.header.tick_data_full.ask = ask_header

        instance.bid.header = bid_header
        instance.ask.header = ask_header
        return instance

    cpdef void parse(self, dict kwargs):
        cdef list parts
        cdef str key
        cdef object value
        cdef str key_type, book_type
        cdef size_t capacity
        cdef size_t level
        cdef md_orderbook* orderbook
        cdef md_orderbook_entry* entry

        for key, value in kwargs.items():
            parts = key.split('_')
            if len(parts) != 3:
                self.__dict__[key] = value
                continue

            book_type = parts[0]  # 'bid' or 'ask'
            key_type = parts[1]  # 'price', 'volume', or 'orders'

            # Check if there's a level number
            if not parts[2].isdigit():
                self.__dict__[key] = value
                continue

            # Convert to 0-based index
            # if level part is provided 0, it will be converted to SIZE_MAX, which will out of range and be saved by __dict__
            level = int(parts[2]) - 1

            if book_type == 'bid':
                orderbook = self.header.tick_data_full.bid
            elif book_type == 'ask':
                orderbook = self.header.tick_data_full.ask
            else:
                self.__dict__[key] = value
                continue

            # Skip if level is out of range
            capacity = orderbook.capacity
            if level < 0 or level >= capacity:
                self.__dict__[key] = value
                continue

            entry = orderbook.entries + level
            if key_type == 'price':
                entry.price = value
            elif key_type == 'volume':
                entry.volume = value
            elif key_type == 'orders' or key_type == 'n_orders':
                entry.n_orders = int(value)
            else:
                self.__dict__[key] = value
                continue

            if level >= orderbook.size:
                orderbook.size = level + 1

            if not entry.n_orders:
                entry.n_orders = 1

        # Sort the books
        c_md_orderbook_sort(self.header.tick_data_full.bid)
        c_md_orderbook_sort(self.header.tick_data_full.ask)

    cpdef TickDataLite lite(self):
        cdef TickDataLite instance = TickDataLite.__new__(TickDataLite)
        instance.header = <md_variant*> &self.header.tick_data_full.lite
        instance.owner = False
        return instance

    property best_bid_price:
        def __get__(self):
            return self.header.tick_data_full.bid.entries.price

    property best_ask_price:
        def __get__(self):
            return self.header.tick_data_full.ask.entries.price

    property best_bid_volume:
        def __get__(self):
            return self.header.tick_data_full.bid.entries.volume

    property best_ask_volume:
        def __get__(self):
            return self.header.tick_data_full.ask.entries.volume

    property last_price:
        def __get__(self):
            return self.header.tick_data_full.lite.last_price

    property bid_price:
        def __get__(self):
            return self.header.tick_data_full.lite.bid_price

    property bid_volume:
        def __get__(self):
            return self.header.tick_data_full.lite.bid_volume

    property ask_price:
        def __get__(self):
            return self.header.tick_data_full.lite.ask_price

    property ask_volume:
        def __get__(self):
            return self.header.tick_data_full.lite.ask_volume

    property open_price:
        def __get__(self):
            return self.header.tick_data_full.lite.open_price

    property prev_close:
        def __get__(self):
            return self.header.tick_data_full.lite.prev_close

    property total_traded_volume:
        def __get__(self):
            return self.header.tick_data_full.lite.total_traded_volume

    property total_traded_notional:
        def __get__(self):
            return self.header.tick_data_full.lite.total_traded_notional

    property total_trade_count:
        def __get__(self):
            return self.header.tick_data_full.lite.total_trade_count

    property total_bid_volume:
        def __get__(self):
            return self.header.tick_data_full.total_bid_volume

    property total_ask_volume:
        def __get__(self):
            return self.header.tick_data_full.total_ask_volume

    property weighted_bid_price:
        def __get__(self):
            return self.header.tick_data_full.weighted_bid_price

    property weighted_ask_price:
        def __get__(self):
            return self.header.tick_data_full.weighted_ask_price

    property mid_price:
        def __get__(self):
            cdef double bid = self.header.tick_data_full.lite.bid_price
            cdef double ask = self.header.tick_data_full.lite.ask_price
            return (bid + ask) / 2.0

    property spread:
        def __get__(self):
            cdef double bid = self.header.tick_data_full.lite.bid_price
            cdef double ask = self.header.tick_data_full.lite.ask_price
            return ask - bid


from . cimport c_market_data

c_market_data.tick_lite_from_header = tick_lite_from_header
c_market_data.tick_from_header = tick_from_header
