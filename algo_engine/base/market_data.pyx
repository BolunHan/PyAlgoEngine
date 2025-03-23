# cython: language_level=3
from libc.stdio cimport printf
from libc.string cimport strcmp, memcpy, memset
from libc.stdint cimport uint8_t, int32_t, uint32_t, int64_t, uint64_t
from libc.stddef cimport size_t
from libc.stdlib cimport qsort
from cpython.buffer cimport PyBuffer_FillInfo
from cpython.object cimport PyObject
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.math cimport fabs, isnan, INFINITY, NAN


# Define compile-time constants
cdef extern from "market_data_external.c":
    int compare_entries_bid(const void* a, const void* b) nogil
    int compare_entries_ask(const void* a, const void* b) nogil
    const int TICKER_SIZE
    const int BOOK_SIZE
    const int ID_SIZE

# Enum definitions
cdef public enum TransactionSide:
    ASK_ORDER = -4
    ShortOrder = -3
    ShortOpen = -2
    ShortFilled = -1
    LongClose = -1
    UNKNOWN = 0
    LongFilled = 1
    LongOpen = 1
    ShortClose = 2
    LongOrder = 3
    BID_ORDER = 4

cdef public enum OrderType:
    UnknownOrder = -20
    CancelOrder = -10
    Generic = 0
    LimitOrder = 10
    LimitMarketMaking = 11
    MarketOrder = 2
    FOK = 21
    FAK = 22
    IOC = 23

# Data type mapping
cdef public enum DataType:
    DTYPE_UNKNOWN = 0
    DTYPE_MARKET_DATA = 10
    DTYPE_TRANSACTION = 20
    DTYPE_BOOK = 21
    DTYPE_ORDER = 30
    DTYPE_TICK_LITE = 31
    DTYPE_TICK = 32
    DTYPE_BAR = 40

    ORDER_BOOK = 2
    BAR_DATA = 3
    TICK_DATA = 4
    ORDER_DATA = 6

# Helper class for TransactionSide
cdef class TransactionHelper:
    """
    Helper class to manage TransactionSide behaviors.
    """

    @staticmethod
    cdef int get_opposite(TransactionSide side):
        """
        Get the opposite transaction side.
        """
        if side == LongOpen:
            return ShortClose
        elif side == ShortClose:
            return LongOpen
        elif side == ShortOpen:
            return LongClose
        elif side == LongClose:
            return ShortOpen
        elif side == BID_ORDER:
            return TransactionSide.ASK_ORDER
        elif side == TransactionSide.ASK_ORDER:
            return BID_ORDER
        printf("No valid registered opposite trade side for %d\n", side)
        return UNKNOWN

    @staticmethod
    cdef int from_offset(const char* direction, const char* offset):
        """
        Determine the transaction side from direction and offset.
        """
        if strcmp(direction, "buy") == 0 or strcmp(direction, "long") == 0:
            if strcmp(offset, "open") == 0:
                return LongOpen
            elif strcmp(offset, "close") == 0:
                return ShortOpen
        elif strcmp(direction, "sell") == 0 or strcmp(direction, "short") == 0:
            if strcmp(offset, "open") == 0:
                return ShortOpen
            elif strcmp(offset, "close") == 0:
                return LongClose
        printf("Not recognized: %s %s\n", direction, offset)
        return UNKNOWN

    @staticmethod
    cdef int get_sign(TransactionSide side):
        """
        Get the sign of the transaction side.
        """
        if side == LongOpen or side == ShortClose:
            return 1
        elif side == ShortOpen or side == LongClose:
            return -1
        return 0

    @staticmethod
    cdef int get_order_sign(TransactionSide side):
        """
        Get the order sign of the transaction side.
        """
        if side == LongOrder:
            return 1
        elif side == ShortOrder:
            return -1
        elif side == UNKNOWN:
            return 0
        return 0

    @staticmethod
    cdef const char* get_side_name(TransactionSide side):
        """
        Get the name of the transaction side.
        """
        if side == LongOpen or side == ShortClose:
            return "long"
        elif side == ShortOpen or side == LongClose:
            return "short"
        elif side == TransactionSide.ASK_ORDER:
            return "ask"
        elif side == BID_ORDER:
            return "bid"
        return "Unknown"

    @staticmethod
    cdef const char* get_offset_name(TransactionSide side):
        """
        Get the offset name of the transaction side.
        """
        if side == LongOpen or side == ShortOpen:
            return "open"
        elif side == LongClose or side == ShortClose:
            return "close"
        elif side == ASK_ORDER or side == BID_ORDER:
            return "ask" if side == ASK_ORDER else "bid"
        return "unknown"

# ID Structures
cdef struct IntID:
    int32_t id_type
    char data[ID_SIZE]  # Using char array for byte storage

cdef struct StrID:
    int32_t id_type
    char data[ID_SIZE]

cdef union UnionID:
    int32_t id_type
    IntID id_int
    StrID id_str

# Meta info structure
cdef struct _MetaInfo:
    uint8_t dtype
    char ticker[TICKER_SIZE]
    double timestamp

cdef struct _Entry:
    double price
    double volume
    uint32_t n_orders

# OrderBookBuffer structure
cdef struct _OrderBookBuffer:
    _Entry entries[BOOK_SIZE]

# BarData structure
cdef struct _CandlestickBuffer:
    uint8_t dtype
    char ticker[TICKER_SIZE]
    double timestamp
    double bar_span
    double high_price
    double low_price
    double open_price
    double close_price
    double volume
    double notional
    uint32_t trade_count

# TickDataLite structure
cdef struct _TickDataLiteBuffer:
    uint8_t dtype
    char ticker[TICKER_SIZE]
    double timestamp
    double bid_price
    double bid_volume
    double ask_price
    double ask_volume
    double last_price
    double total_traded_volume
    double total_traded_notional
    uint32_t total_trade_count

cdef struct _TickDataBuffer:
    _TickDataLiteBuffer lite
    _OrderBookBuffer bid_book
    _OrderBookBuffer ask_book

# TransactionData structure
cdef struct _TransactionDataBuffer:
    uint8_t dtype
    char ticker[TICKER_SIZE]
    double timestamp
    double price
    double volume
    int32_t side
    double multiplier
    double notional
    UnionID transaction_id
    UnionID buy_id
    UnionID sell_id

# OrderData structure
cdef struct _OrderDataBuffer:
    uint8_t dtype
    char ticker[TICKER_SIZE]
    double timestamp
    double price
    double volume
    int32_t side
    UnionID order_id
    int32_t order_type

# Base MarketData structure as a union
cdef union _MarketDataBuffer:
    _MetaInfo MetaInfo
    _TransactionDataBuffer TransactionData
    _OrderDataBuffer OrderData
    _CandlestickBuffer BarData
    _TickDataLiteBuffer TickDataLite
    _TickDataBuffer TickData

# Base MarketData class
cdef class MarketData:
    """
    Base class for all market data types.
    """
    cdef _MarketDataBuffer* _data
    cdef bint _owner
    cdef int _dtype

    def __cinit__(self):
        """
        Initialize the class but don't allocate memory.
        Child classes will allocate the appropriate memory.
        """
        self._data = NULL
        self._owner = False

    def __dealloc__(self):
        """
        Free allocated memory if this instance owns it.
        """
        if self._data is not NULL and self._owner:
            PyMem_Free(self._data)
            self._data = NULL

    def __init__(self, str ticker, double timestamp):
        """
        Initialize the market data with values.
        This method should only be called after memory has been allocated by a child class.
        """
        if self._data == NULL:
            raise ValueError("Memory not allocated. This class should not be instantiated directly.")

        cdef bytes ticker_bytes = ticker.encode('utf-8')
        cdef int ticker_len = min(len(ticker_bytes), TICKER_SIZE - 1)

        memset(&self._data.MetaInfo.ticker, 0, TICKER_SIZE)
        memcpy(&self._data.MetaInfo.ticker, <char*>ticker_bytes, ticker_len)

        self._data.MetaInfo.timestamp = timestamp
        self._data.MetaInfo.dtype = DataType.DTYPE_MARKET_DATA

    @classmethod
    def from_buffer(cls, const unsigned char[:] buffer):
        """
        Create a new instance from a buffer.
        Uses meta info to determine the correct data structure.
        """
        # Get the data type from the buffer
        cdef uint8_t dtype = buffer[0]

        # Create the appropriate class based on the data type
        cdef MarketData instance

        if dtype == DataType.DTYPE_TRANSACTION:
            instance = TransactionData.__new__(TransactionData)
        elif dtype == DataType.BAR_DATA:
            instance = BarData.__new__(BarData)
        elif dtype == DataType.ORDER_BOOK:
            # OrderBook will be implemented later
            raise NotImplementedError("OrderBook not yet implemented")
        elif dtype == DataType.TICK_DATA:
            # TickData will be implemented later
            raise NotImplementedError("TickData not yet implemented")
        elif dtype == DataType.ORDER_DATA:
            # OrderData will be implemented later
            raise NotImplementedError("OrderData not yet implemented")
        else:
            # Default to MarketData
            instance = cls.__new__(cls)

        # Free the automatically allocated memory if any
        if instance._data is not NULL and instance._owner:
            PyMem_Free(instance._data)
            instance._data = NULL

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
        # Get the data type from the bytes
        cdef uint8_t dtype = <uint8_t>data[0]
        cdef const char* data_ptr = <const char*>data

        # Create the appropriate class based on the data type
        cdef MarketData instance

        if dtype == DataType.DTYPE_TRANSACTION:
            instance = TransactionData.__new__(TransactionData)
            # Free any existing memory
            if instance._data is not NULL and instance._owner:
                PyMem_Free(instance._data)
                instance._data = NULL

            # Allocate new memory
            instance._data = <_MarketDataBuffer*>PyMem_Malloc(sizeof(_TransactionDataBuffer))
            if instance._data == NULL:
                raise MemoryError("Failed to allocate memory for TransactionData")

            # Copy the data
            memcpy(instance._data, data_ptr, sizeof(_TransactionDataBuffer))

        elif dtype == DataType.BAR_DATA:
            instance = BarData.__new__(BarData)
            # Free any existing memory
            if instance._data is not NULL and instance._owner:
                PyMem_Free(instance._data)
                instance._data = NULL

            # Allocate new memory
            instance._data = <_MarketDataBuffer*>PyMem_Malloc(sizeof(_CandlestickBuffer))
            if instance._data == NULL:
                raise MemoryError("Failed to allocate memory for BarData")

            # Copy the data
            memcpy(instance._data, data_ptr, sizeof(_CandlestickBuffer))

        elif dtype == DataType.ORDER_BOOK:
            # OrderBook will be implemented later
            raise NotImplementedError("OrderBook not yet implemented")

        elif dtype == DataType.TICK_DATA:
            # TickData will be implemented later
            raise NotImplementedError("TickData not yet implemented")

        elif dtype == DataType.ORDER_DATA:
            # OrderData will be implemented later
            raise NotImplementedError("OrderData not yet implemented")

        else:
            # Default to MarketData
            instance = cls.__new__(cls)
            # Free any existing memory
            if instance._data is not NULL and instance._owner:
                PyMem_Free(instance._data)
                instance._data = NULL

            # Allocate new memory
            instance._data = <_MarketDataBuffer*>PyMem_Malloc(sizeof(_MarketDataBuffer))
            if instance._data == NULL:
                raise MemoryError("Failed to allocate memory for MarketData")

            # Copy the data
            memcpy(instance._data, data_ptr, sizeof(_MarketDataBuffer))

        instance._owner = True
        return instance

    def to_bytes(self):
        """
        Convert the market data to bytes.
        Uses the meta info to determine the data type and size.
        """
        if self._data == NULL:
            raise ValueError("Cannot convert uninitialized data to bytes")

        cdef uint8_t dtype = self._data.MetaInfo.dtype
        cdef size_t size

        if dtype == DataType.DTYPE_TRANSACTION:
            size = sizeof(_TransactionDataBuffer)
        elif dtype == DataType.BAR_DATA:
            size = sizeof(_CandlestickBuffer)
        elif dtype == DataType.ORDER_BOOK:
            size = sizeof(_OrderBookBuffer)
        elif dtype == DataType.TICK_DATA:
            size = sizeof(_TickDataBuffer)
        elif dtype == DataType.ORDER_DATA:
            size = sizeof(_OrderDataBuffer)
        else:
            # Default to MarketData
            size = sizeof(_MarketDataBuffer)

        return PyBytes_FromStringAndSize(<char*>self._data, size)

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        """
        Implement the buffer protocol for read-only access.
        """
        if self._data == NULL:
            raise ValueError("Cannot get buffer from uninitialized data")

        cdef uint8_t dtype = self._data.MetaInfo.dtype
        cdef size_t size

        if dtype == DataType.DTYPE_TRANSACTION:
            size = sizeof(_TransactionDataBuffer)
        elif dtype == DataType.BAR_DATA:
            size = sizeof(_CandlestickBuffer)
        elif dtype == DataType.ORDER_BOOK:
            size = sizeof(_OrderBookBuffer)
        elif dtype == DataType.TICK_DATA:
            size = sizeof(_TickDataBuffer)
        elif dtype == DataType.ORDER_DATA:
            size = sizeof(_OrderDataBuffer)
        else:
            # Default to MarketData
            size = sizeof(_MarketDataBuffer)

        PyBuffer_FillInfo(buffer, self, <void*>self._data, size, 1, flags)

    def __releasebuffer__(self, Py_buffer* buffer):
        """
        Release the buffer.
        """
        pass

    @property
    def ticker(self) -> str:
        """
        Get the ticker symbol.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.MetaInfo.ticker.decode('utf-8').rstrip('\0')

    @property
    def timestamp(self) -> float:
        """
        Get the timestamp.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.MetaInfo.timestamp

    @property
    def dtype(self) -> int:
        """
        Get the data type.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.MetaInfo.dtype

    def __copy__(self):
        """
        Create a copy of this instance.
        """
        if self._data == NULL:
            raise ValueError("Cannot copy uninitialized data")

        # Create a new instance of the same class
        cdef MarketData new_md = type(self).__new__(type(self))

        # Allocate memory based on the data type
        cdef uint8_t dtype = self._data.MetaInfo.dtype
        cdef size_t size

        if dtype == DataType.DTYPE_TRANSACTION:
            size = sizeof(_TransactionDataBuffer)
        elif dtype == DataType.BAR_DATA:
            size = sizeof(_CandlestickBuffer)
        elif dtype == DataType.ORDER_BOOK:
            size = sizeof(_OrderBookBuffer)
        elif dtype == DataType.TICK_DATA:
            size = sizeof(_TickDataBuffer)
        elif dtype == DataType.ORDER_DATA:
            size = sizeof(_OrderDataBuffer)
        else:
            # Default to MarketData
            size = sizeof(_MarketDataBuffer)

        new_md._data = <_MarketDataBuffer*>PyMem_Malloc(size)
        if new_md._data == NULL:
            raise MemoryError("Failed to allocate memory for copy")

        new_md._owner = True
        memcpy(new_md._data, self._data, size)

        return new_md

# TransactionData class
cdef class TransactionData(MarketData):
    """
    Represents transaction data for a specific market.
    """
    def __cinit__(self):
        """
        Allocate memory for the transaction data structure but don't initialize it.
        """
        self._dtype = DataType.DTYPE_TRANSACTION
        self._owner = True

    def __init__(self, str ticker, double timestamp, double price, double volume, int side=0, double multiplier=1.0, double notional=0.0, object transaction_id=None, object buy_id=None, object sell_id=None):
        """
        Initialize the transaction data with values.
        """
        if not self._owner:
            raise MemoryError(f"Can not initialize a view of {self.__class__.__name__}.")

        self._data = <_MarketDataBuffer*> PyMem_Malloc(sizeof(_TransactionDataBuffer))
        memset(self._data, 0, sizeof(_TransactionDataBuffer))

        # Initialize base class fields
        MarketData.__init__(self, ticker, timestamp)

        # Set data type for TransactionData
        self._data.MetaInfo.dtype = DataType.DTYPE_TRANSACTION

        # Initialize transaction-specific fields
        self._data.TransactionData.price = price
        self._data.TransactionData.volume = volume
        self._data.TransactionData.side = side
        self._data.TransactionData.multiplier = multiplier

        # Calculate notional if not provided
        if notional == 0.0:
            self._data.TransactionData.notional = price * volume * multiplier
        else:
            self._data.TransactionData.notional = notional

        # Initialize IDs
        self._set_id(name='transaction_id', id_value=transaction_id)
        self._set_id(name='buy_id', id_value=buy_id)
        self._set_id(name='sell_id', id_value=sell_id)

    cdef void _set_id(self, str name, object id_value):
        """
        Set an ID field in the transaction data.
        """
        cdef UnionID* id_ptr
        cdef bytes id_bytes
        cdef int id_len

        if name == 'transaction_id':
            id_ptr = &self._data.TransactionData.transaction_id
        elif name == 'buy_id':
            id_ptr = &self._data.TransactionData.buy_id
        elif name == 'sell_id':
            id_ptr = &self._data.TransactionData.sell_id
        else:
            return

        if id_value is None:
            id_ptr.id_type = 0  # None type
        elif isinstance(id_value, int):
            id_ptr.id_int.id_type = 1  # Int type
            # Convert int to string and store in data
            id_bytes = id_value.to_bytes((id_value.bit_length() + 7) // 8, byteorder='little', signed=True)
            id_len = min(len(id_bytes), ID_SIZE - 1)
            memset(id_ptr.id_int.data, 0, ID_SIZE)
            memcpy(id_ptr.id_int.data, <char*>id_bytes, id_len)
        elif isinstance(id_value, str):
            id_ptr.id_str.id_type = 2  # Str type
            id_bytes = id_value.encode('utf-8')
            id_len = min(len(id_bytes), ID_SIZE - 1)
            memset(id_ptr.id_str.data, 0, ID_SIZE)
            memcpy(id_ptr.id_str.data, <char*>id_bytes, id_len)

    cdef object _get_id(self, str name):
        """
        Get an ID field from the transaction data.
        """
        cdef UnionID* id_ptr

        if name == 'transaction_id':
            id_ptr = &self._data.TransactionData.transaction_id
        elif name == 'buy_id':
            id_ptr = &self._data.TransactionData.buy_id
        elif name == 'sell_id':
            id_ptr = &self._data.TransactionData.sell_id
        else:
            return None

        if id_ptr.id_type == 0:
            return None
        elif id_ptr.id_type == 1:  # Int type
            return int.from_bytes(id_ptr.id_int.data[:ID_SIZE], byteorder='little', signed=True)
        elif id_ptr.id_type == 2:  # Str type
            return id_ptr.id_str.data.decode('utf-8').rstrip('\0')

        return None

    @classmethod
    def from_buffer(cls, const unsigned char[:] buffer):
        """
        Create a new instance from a buffer.
        """
        cdef TransactionData instance = cls.__new__(cls)

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
        cdef TransactionData instance = cls.__new__(cls)
        cdef const char* data_ptr = <const char*>data

        instance._owner = True
        instance._data = <_MarketDataBuffer*>PyMem_Malloc(sizeof(_TransactionDataBuffer))

        if instance._data == NULL:
            raise MemoryError("Failed to allocate memory for TransactionData")

        memcpy(instance._data, data_ptr, sizeof(_TransactionDataBuffer))

        return instance

    def to_bytes(self):
        """
        Convert the transaction data to bytes.
        """
        if self._data == NULL:
            raise ValueError("Cannot convert uninitialized data to bytes")

        return PyBytes_FromStringAndSize(<char*>self._data, sizeof(_TransactionDataBuffer))

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        """
        Implement the buffer protocol for read-only access.
        """
        if self._data == NULL:
            raise ValueError("Cannot get buffer from uninitialized data")
        PyBuffer_FillInfo(buffer, self, <_TransactionDataBuffer*>self._data, sizeof(_TransactionDataBuffer), 1, flags)

    def __copy__(self):
        """
        Create a copy of this instance.
        """
        if self._data == NULL:
            raise ValueError("Cannot copy uninitialized data")

        cdef TransactionData new_td = TransactionData.__new__(TransactionData)
        # Allocate memory specifically for a TransactionData buffer
        new_td._data = <_MarketDataBuffer*>PyMem_Malloc(sizeof(_TransactionDataBuffer))
        if new_td._data == NULL:
            raise MemoryError("Failed to allocate memory for copy")

        new_td._owner = True
        memcpy(new_td._data, self._data, sizeof(_TransactionDataBuffer))

        return new_td

    @property
    def price(self) -> float:
        """
        Get the transaction price.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TransactionData.price

    @property
    def volume(self) -> float:
        """
        Get the transaction volume.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TransactionData.volume

    @property
    def side(self) -> int:
        """
        Get the transaction side.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TransactionData.side

    @property
    def multiplier(self) -> float:
        """
        Get the transaction multiplier.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TransactionData.multiplier

    @property
    def notional(self) -> float:
        """
        Get the transaction notional value.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TransactionData.notional

    @property
    def transaction_id(self) -> object:
        """
        Get the transaction ID.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._get_id('transaction_id')

    @property
    def buy_id(self) -> object:
        """
        Get the buy ID.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._get_id('buy_id')

    @property
    def sell_id(self) -> object:
        """
        Get the sell ID.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._get_id('sell_id')

    @property
    def market_price(self) -> float:
        """
        Alias for the transaction price.
        """
        return self.price

    @property
    def flow(self) -> float:
        """
        Calculate the flow of the transaction.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        cdef int sign = TransactionHelper.get_sign(<TransactionSide>self._data.TransactionData.side)
        return sign * self._data.TransactionData.volume

    def __repr__(self) -> str:
        """
        String representation of the transaction data.
        """
        if self._data == NULL:
            return "TransactionData(uninitialized)"
        side_name = TransactionHelper.get_side_name(<TransactionSide>self._data.TransactionData.side).decode('utf-8')
        return (f"TransactionData(ticker='{self.ticker}', timestamp={self.timestamp}, "
                f"price={self.price}, volume={self.volume}, side={side_name})")

# BarData class
cdef class BarData(MarketData):
    """
    Represents a single bar of market data for a specific ticker within a given time frame.
    """
    def __cinit__(self):
        """
        Allocate memory for the bar data structure but don't initialize it.
        """
        self._dtype = DataType.BAR_DATA
        self._owner = True

    def __init__(self, str ticker, double timestamp, double high_price, double low_price,
                 double open_price, double close_price, double volume=0.0, double notional=0.0,
                 uint32_t trade_count=0, double bar_span=0.0):
        """
        Initialize the bar data with values.
        """
        if not self._owner:
            raise MemoryError(f"Can not initialize a view of {self.__class__.__name__}.")

        self._data = <_MarketDataBuffer*>PyMem_Malloc(sizeof(_CandlestickBuffer))
        memset(self._data, 0, sizeof(_CandlestickBuffer))

        # Initialize base class fields
        MarketData.__init__(self, ticker, timestamp)

        # Set data type for BarData
        self._data.MetaInfo.dtype = DataType.BAR_DATA

        # Initialize bar-specific fields
        self._data.BarData.high_price = high_price
        self._data.BarData.low_price = low_price
        self._data.BarData.open_price = open_price
        self._data.BarData.close_price = close_price
        self._data.BarData.volume = volume
        self._data.BarData.notional = notional
        self._data.BarData.trade_count = trade_count
        self._data.BarData.bar_span = bar_span

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

    def to_bytes(self):
        """
        Convert the bar data to bytes.
        """
        if self._data == NULL:
            raise ValueError("Cannot convert uninitialized data to bytes")

        return PyBytes_FromStringAndSize(<char*>self._data, sizeof(_CandlestickBuffer))

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
    def bar_span(self) -> float:
        """
        Get the duration of the bar.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.BarData.bar_span

    @property
    def vwap(self) -> float:
        """
        Get the volume-weighted average price for the bar.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        if self._data.BarData.volume <= 0:
            return 0.0
        return self._data.BarData.notional / self._data.BarData.volume

    def __repr__(self) -> str:
        """
        String representation of the bar data.
        """
        if self._data == NULL:
            return "BarData(uninitialized)"
        return (f"BarData(ticker='{self.ticker}', timestamp={self.timestamp}, "
                f"open={self.open_price}, high={self.high_price}, "
                f"low={self.low_price}, close={self.close_price}, "
                f"volume={self.volume})")

# TickDataLite class
cdef class TickDataLite(MarketData):
    """
    Represents tick data for a specific ticker without the order_book field.
    """

    def __cinit__(self):
        """
        Initialize the class but don't allocate memory.
        """
        self._dtype = DataType.DTYPE_TICK_LITE
        self._owner = True

    def __init__(self, str ticker, double timestamp, double last_price,
                 double bid_price=NAN, double bid_volume=NAN,
                 double ask_price=NAN, double ask_volume=NAN,
                 double total_traded_volume=0.0, double total_traded_notional=0.0,
                 uint32_t total_trade_count=0):
        """
        Initialize the tick data with values.
        """
        if not self._owner:
            raise MemoryError(f"Can not initialize a view of {self.__class__.__name__}.")

        self._data = <_MarketDataBuffer*>PyMem_Malloc(sizeof(_TickDataLiteBuffer))
        memset(self._data, 0, sizeof(_TickDataLiteBuffer))

        # Initialize base class fields
        MarketData.__init__(self, ticker, timestamp)

        # Set data type for TickDataLite
        self._data.MetaInfo.dtype = DataType.DTYPE_TICK_LITE

        # Set other fields
        self._data.TickDataLite.last_price = last_price
        self._data.TickDataLite.bid_price = bid_price
        self._data.TickDataLite.bid_volume = bid_volume
        self._data.TickDataLite.ask_price = ask_price
        self._data.TickDataLite.ask_volume = ask_volume
        self._data.TickDataLite.total_traded_volume = total_traded_volume
        self._data.TickDataLite.total_traded_notional = total_traded_notional
        self._data.TickDataLite.total_trade_count = total_trade_count

    @classmethod
    def from_buffer(cls, const unsigned char[:] buffer):
        cdef TickDataLite instance = cls.__new__(cls)

        # Point to the buffer data
        instance._data = <_MarketDataBuffer*>&buffer[0]
        instance._owner = False

        return instance

    def to_bytes(self):
        """
        Convert the instance to bytes.
        """
        if self._data is NULL:
            raise ValueError("No data available")

        return PyBytes_FromStringAndSize(<char*>self._data, sizeof(_TickDataLiteBuffer))

    @classmethod
    def from_bytes(cls, bytes data):
        cdef TickDataLite instance = cls.__new__(cls)
        cdef const char * data_ptr = <const char *> data

        instance._owner = True
        instance._data = <_MarketDataBuffer *> PyMem_Malloc(sizeof(_TickDataLiteBuffer))

        if instance._data == NULL:
            raise MemoryError("Failed to allocate memory for BarData")

        memcpy(instance._data, data_ptr, sizeof(_TickDataLiteBuffer))

        return instance

    @property
    def last_price(self):
        """Get the last price."""
        if self._data is NULL:
            raise ValueError("No data available")
        return self._data.TickDataLite.last_price

    @property
    def bid_price(self):
        """Get the bid price."""
        if self._data is NULL:
            raise ValueError("No data available")
        return self._data.TickDataLite.bid_price

    @property
    def bid_volume(self):
        """Get the bid volume."""
        if self._data is NULL:
            raise ValueError("No data available")
        return self._data.TickDataLite.bid_volume

    @property
    def ask_price(self):
        """Get the ask price."""
        if self._data is NULL:
            raise ValueError("No data available")
        return self._data.TickDataLite.ask_price

    @property
    def ask_volume(self):
        """Get the ask volume."""
        if self._data is NULL:
            raise ValueError("No data available")
        return self._data.TickDataLite.ask_volume

    @property
    def total_traded_volume(self):
        """Get the total traded volume."""
        if self._data is NULL:
            raise ValueError("No data available")
        return self._data.TickDataLite.total_traded_volume

    @property
    def total_traded_notional(self):
        """Get the total traded notional."""
        if self._data is NULL:
            raise ValueError("No data available")
        return self._data.TickDataLite.total_traded_notional

    @property
    def total_trade_count(self):
        """Get the total trade count."""
        if self._data is NULL:
            raise ValueError("No data available")
        return self._data.TickDataLite.total_trade_count

    @property
    def mid_price(self):
        """Get the mid price."""
        if self._data is NULL:
            raise ValueError("No data available")
        return (self._data.TickDataLite.bid_price + self._data.TickDataLite.ask_price) / 2.0

    @property
    def market_price(self):
        """Get the market price (alias for last_price)."""
        return self.last_price

cdef class OrderBook:
    cdef _OrderBookBuffer * _data
    cdef public TransactionSide side
    cdef public bint sorted
    cdef bint _owner  # Flag to indicate ownership of the buffer

    # Constructor to initialize the object (without allocating memory)
    def __cinit__(self):
        self._data = NULL
        self.sorted = False
        self._owner = False
        self.side = TransactionSide.UNKNOWN

    # Initialization function with memory allocation and population
    def __init__(self, side: TransactionSide, price=None, volume=None, n_orders=None, is_sorted = False):
        self.sorted = is_sorted
        self._owner = True
        self.side = side

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

        self.sort()

    def __dealloc__(self):
        """
        Free allocated memory if this instance owns it.
        """
        if self._data is not NULL and self._owner:
            PyMem_Free(self._data)
            self._data = NULL

    # Method to get price, volume, and n_orders at a specific level
    def at_price(self, price: float) -> tuple[float, float, float]:
        for i in range(BOOK_SIZE):
            if self._data.entries[i].price == price and self._data.entries[i].n_orders > 0:
                return self._data.entries[i].price, self._data.entries[i].volume, self._data.entries[i].n_orders

        raise IndexError(f'price {price} not found!')

    def at_level(self, index: int) -> tuple[float, float, float]:
        self.sort()

        if 0 <= index < BOOK_SIZE and self._data.entries[index].n_orders > 0:
            return self._data.entries[index].price, self._data.entries[index].volume, self._data.entries[index].n_orders

        raise IndexError(f'level {index} not found!')

    # Method to create OrderBook instance from an existing buffer (without owning the data)
    @classmethod
    def from_buffer(cls, buffer):
        cdef OrderBook instance = cls.__new__(cls)
        instance._data = <_OrderBookBuffer*> buffer
        instance._owner = False
        return instance

    # Numpy conversion
    def to_numpy(self):
        import numpy as np
        return np.array([[self._data.entries[i].price, self._data.entries[i].volume, self._data.entries[i].n_orders] for i in range(BOOK_SIZE)])

    @property
    def price(self):
        return [self._data.entries[i].price for i in range(BOOK_SIZE) if self._data.entries[i].n_orders > 0]

    @property
    def volume(self):
        return [self._data.entries[i].volume for i in range(BOOK_SIZE) if self._data.entries[i].n_orders > 0]

    @property
    def n_orders(self):
        return [self._data.entries[i].n_orders for i in range(BOOK_SIZE) if self._data.entries[i].n_orders > 0]

    # Sort method based on side using qsort
    def sort(self):
        if self.sorted:
            return  # Skip sorting if already sorted
        if self.side == TransactionSide.BID_ORDER:
            qsort(self._data.entries, BOOK_SIZE, sizeof(_Entry), compare_entries_bid)
        elif self.side == TransactionSide.ASK_ORDER:
            qsort(self._data.entries, BOOK_SIZE, sizeof(_Entry), compare_entries_ask)
        else:
            raise ValueError(f'Invalid TransactionSide {self.side}.')

        self.sorted = True
