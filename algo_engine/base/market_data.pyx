# cython: language_level=3
from libc.stdio cimport printf
from libc.string cimport strcmp, memcpy, memset
from libc.stdint cimport uint8_t, int32_t, uint32_t, int64_t, uint64_t
from libc.stdlib cimport atoi
from cpython.buffer cimport PyBuffer_FillInfo
from cpython.object cimport PyObject
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.bytes cimport PyBytes_FromStringAndSize

# Define compile-time constants
cdef extern from *:
    """
    #define TICKER_SIZE 32
    #define BOOK_SIZE 10
    #define ID_SIZE 16
    """
    const int TICKER_SIZE
    const int BOOK_SIZE
    const int ID_SIZE

# Enum definitions
cdef public enum TransactionSide:
    ShortOrder = -3
    ShortOpen = -2
    ShortFilled = -1
    LongClose = -1
    UNKNOWN = 0
    LongFilled = 1
    LongOpen = 1
    ShortClose = 2
    LongOrder = 3
    BidOrder = 3

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
    UNKNOWN_TYPE = 0
    MARKET_DATA = 1
    ORDER_BOOK = 2
    BAR_DATA = 3
    TICK_DATA = 4
    TRANSACTION_DATA = 5
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
        elif side == BidOrder:
            return ShortOrder
        elif side == ShortOrder:
            return BidOrder
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
        elif side == ShortOrder:
            return "ask"
        elif side == BidOrder:
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
        elif side == ShortOrder or side == BidOrder:
            return "ask" if side == ShortOrder else "bid"
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

# OrderBook structure
cdef struct _OrderBookBuffer:
    uint8_t dtype
    char ticker[TICKER_SIZE]
    double timestamp
    double bid_price[BOOK_SIZE]
    double ask_price[BOOK_SIZE]
    double bid_volume[BOOK_SIZE]
    double ask_volume[BOOK_SIZE]
    uint32_t bid_n_orders[BOOK_SIZE]
    uint32_t ask_n_orders[BOOK_SIZE]

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

# TickData structure
cdef struct _TickDataBuffer:
    uint8_t dtype
    char ticker[TICKER_SIZE]
    double timestamp
    _OrderBookBuffer order_book
    double bid_price
    double bid_volume
    double ask_price
    double ask_volume
    double last_price
    double total_traded_volume
    double total_traded_notional
    uint32_t total_trade_count

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
    _OrderBookBuffer OrderBook
    _CandlestickBuffer BarData
    _TickDataBuffer TickData
    _TransactionDataBuffer TransactionData
    _OrderDataBuffer OrderData

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
        self._data.MetaInfo.dtype = DataType.MARKET_DATA

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

        if dtype == DataType.TRANSACTION_DATA:
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

        if dtype == DataType.TRANSACTION_DATA:
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

        if dtype == DataType.TRANSACTION_DATA:
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

        if dtype == DataType.TRANSACTION_DATA:
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

        if dtype == DataType.TRANSACTION_DATA:
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
        self._dtype = DataType.TRANSACTION_DATA
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
        self._data.MetaInfo.dtype = DataType.TRANSACTION_DATA

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
        self._set_id('transaction_id', transaction_id)
        self._set_id('buy_id', buy_id)
        self._set_id('sell_id', sell_id)

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
            id_bytes = str(id_value).encode('utf-8')
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
            return int(id_ptr.id_int.data.decode('utf-8').rstrip('\0'))
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
        self._dtype = DataType.TRANSACTION_DATA
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


cdef class Book:
    """
    Class representing a side of the order book (either bid or ask).
    """
    cdef int side
    cdef list _book
    cdef dict _dict
    cdef bint sorted_status

    def __cinit__(self):
        """
        Initialize the class but don't allocate memory.
        """
        self.side = 0
        self._book = []
        self._dict = {}
        self.sorted_status = False

    def __init__(self, int side):
        """
        Initialize the order book for a specific side.

        Args:
            side (int): Side of the book; positive for bid, negative for ask.
        """
        self.side = side
        self._book = []
        self._dict = {}
        self.sorted_status = False

    def __iter__(self):
        """
        Iterate over the sorted order book.

        Returns:
            iterator: An iterator over the sorted book entries.
        """
        self.sort()
        return iter(self._book)

    def __contains__(self, float price):
        """
        Check if a price exists in the order book.

        Args:
            price (float): The price to check.

        Returns:
            bool: True if the price exists, False otherwise.
        """
        return price in self._dict

    def __len__(self):
        """
        Get the number of entries in the order book.

        Returns:
            int: The number of entries.
        """
        return len(self._book)

    def __repr__(self):
        """
        Get a string representation of the book.

        Returns:
            str: A string indicating whether the book is for bids or asks.
        """
        return f'<Book.{"Bid" if self.side > 0 else "Ask"}>'

    def __bool__(self):
        """
        Check if the order book has any entries.

        Returns:
            bool: True if the book is not empty, False otherwise.
        """
        return bool(self._book)

    def __sub__(self, Book other):
        """
        Subtract another order book from this one to find the differences in volumes at matching prices.

        Args:
            other (Book): The other book to compare against.

        Returns:
            dict: A dictionary of volume differences.

        Raises:
            TypeError: If the other object is not of type Book.
            ValueError: If the sides of the books do not match.
        """
        if not isinstance(other, Book):
            raise TypeError(f'Expect type Book, got {type(other)}')

        if self.side != other.side:
            raise ValueError(f'Expect side {self.side}, got {other.side}')

        cdef dict diff = {}
        cdef float price, volume, limit, limit_0, limit_1
        cdef bint contain_limit
        cdef tuple entry

        if not other._dict:
            return self._dict

        if not self._dict:
            return {p:-v for p,v in other._dict.items()}

        # bid book
        elif self.side > 0:
            limit_0 = min(self._dict)
            limit_1 = min(other._dict)
            limit = max(limit_0, limit_1)
            contain_limit = limit_0 == limit_1

            for entry in self._book:
                price = entry[0]
                volume = entry[1]

                if price > limit or (price >= limit and contain_limit):
                    diff[price] = volume

            for entry in other._book:
                price = entry[0]
                volume = entry[1]

                if price > limit or (price >= limit and contain_limit):
                    diff[price] = diff.get(price, 0.0) - volume
        # ask book
        else:
            limit_0 = max(self._dict)
            limit_1 = max(other._dict)
            limit = min(limit_0, limit_1)
            contain_limit = limit_0 == limit_1

            for entry in self._book:
                price = entry[0]
                volume = entry[1]

                if price < limit or (price <= limit and contain_limit):
                    diff[price] = volume

            for entry in other._book:
                price = entry[0]
                volume = entry[1]

                if price < limit or (price <= limit and contain_limit):
                    diff[price] = diff.get(price, 0.0) - volume

        return diff

    def pop(self, float price):
        """
        Remove and return an entry at the specified price.

        Args:
            price (float): The price of the entry to remove.

        Returns:
            tuple: The removed entry.

        Raises:
            KeyError: If the price does not exist in the book.
        """
        if price not in self._dict:
            raise KeyError(f'Price {price} not in book')

        entry = self._dict.pop(price)
        self._book.remove(entry)
        self.sorted_status = False
        return entry

    def sort(self):
        """
        Sort the book entries by price.
        For bid book, sort in descending order.
        For ask book, sort in ascending order.
        """
        if self.sorted_status:
            return

        if self.side > 0:
            self._book.sort(key=lambda x: x[0], reverse=True)
        else:
            self._book.sort(key=lambda x: x[0])
        self.sorted_status = True

    def at_level(self, int level):
        """
        Get the entry at a specific level in the sorted book.

        Args:
            level (int): The level to retrieve (0-indexed).

        Returns:
            tuple: The entry at the specified level.

        Raises:
            IndexError: If the level is out of range.
        """
        self.sort()
        return self._book[level]

    def at_price(self, float price):
        """
        Get the entry at a specific price.

        Args:
            price (float): The price to retrieve.

        Returns:
            tuple: The entry at the specified price.

        Raises:
            KeyError: If the price does not exist in the book.
        """
        if price not in self._dict:
            raise KeyError(f'Price {price} not in book')
        return self._dict[price]

    def update(self, float price, float volume, uint32_t n_orders = 0):
        """
        Update or add an entry in the book.

        Args:
            price (float): The price of the entry.
            volume (float): The volume of the entry.
            n_orders (int): The number of orders of the entry, default is 0, which means not available

        Returns:
            tuple: The updated or added entry.
        """
        entry = (price, volume, n_orders)
        if price in self._dict:
            for i, existing_entry in enumerate(self._book):
                if existing_entry[0] == price:
                    self._book.pop(i)
                    break

        self._dict[price] = entry
        self._book.append(entry)
        self.sorted_status = False
        return entry

    def clear(self):
        """
        Clear all entries from the book.
        """
        self._book.clear()
        self._dict.clear()
        self.sorted_status = True

    @property
    def is_sorted(self):
        """
        Check if the book is sorted.

        Returns:
            bool: True if the book is sorted, False otherwise.
        """
        return self.sorted_status

    @property
    def best_price(self):
        """
        Get the best price in the book.

        Returns:
            float: The best price, or NaN if the book is empty.
        """
        if not self._book:
            return float('nan')
        self.sort()
        return self._book[0][0]

    @property
    def best_volume(self):
        """
        Get the volume at the best price.

        Returns:
            float: The volume at the best price, or NaN if the book is empty.
        """
        if not self._book:
            return float('nan')
        self.sort()
        return self._book[0][1]

    @property
    def prices(self):
        """
        Get all prices in the book.

        Returns:
            list: A list of all prices in the book.
        """
        return list(self._dict.keys())

    @property
    def volumes(self):
        """
        Get all volumes in the book.

        Returns:
            list: A list of all volumes in the book.
        """
        return [entry[1] for entry in self._book]

    @property
    def n_orders(self):
        """
        Get all volumes in the book.

        Returns:
            list: A list of all volumes in the book.
        """
        return [entry[2] for entry in self._book]


cdef class OrderBook(MarketData):
    """
    Class representing an order book, which tracks bid and ask orders for a financial instrument.
    """
    cdef Book _bid
    cdef Book _ask

    def __cinit__(self):
        """
        Initialize the class but don't allocate memory.
        """
        self._dtype = DataType.ORDER_BOOK
        self._owner = True
        self._bid = None
        self._ask = None

    def __init__(self, str ticker, double timestamp, list bid=None, list ask=None, **kwargs):
        """
        Initialize the order book with values.

        Args:
            ticker (str): The ticker symbol for the market data.
            timestamp (double): The timestamp of the order book.
            bid (list, optional): List of bid entries [price, volume, n_orders].
            ask (list, optional): List of ask entries [price, volume, n_orders].
        """
        if not self._owner:
            raise MemoryError(f"Can not initialize a view of {self.__class__.__name__}.")

        self._data = <_MarketDataBuffer*>PyMem_Malloc(sizeof(_OrderBookBuffer))
        memset(self._data, 0, sizeof(_OrderBookBuffer))

        # Initialize base class fields
        MarketData.__init__(self, ticker, timestamp)

        # Set data type for OrderBook
        self._data.MetaInfo.dtype = DataType.ORDER_BOOK

        # Initialize book sides
        self._bid = Book(1)  # positive for bid
        self._ask = Book(-1)  # negative for ask

        # Initialize with kwargs if available
        if kwargs:
            for name, value in kwargs.items():
                self.parse(name=name, value=value)
            self._update_books_from_buffer()
        # Initialize with provided data if available
        else:
            if bid is not None:
                self.update_bid(bid)
            if ask is not None:
                self.update_ask(ask)

    def _parse_entry_name(cls, str name, bint validate=False) -> tuple[str, str, int]:
        """
        Parse an entry name like bid_price_X into its components.
        """
        cdef str side, key
        cdef int level

        parts = name.split('_')
        if len(parts) != 3:
            raise ValueError(f'Cannot parse kwargs {name}.')

        side, key, level_str = parts

        if validate:
            if side not in {"bid", "ask"} or key not in {"price", "volume"}:
                raise ValueError(f'Invalid entry name {name}.')
            if not level_str.isdigit():
                raise ValueError(f'Invalid level {level_str} in {name}.')

        level = atoi(level_str)
        return side, key, level

    def parse(self, str name, float value):
        cdef int level

        level = atoi(name.split('_')[-1])

        if name.startswith('bid_price_'):
            self._data.OrderBook.bid_price[level] = value
        elif name.startswith('bid_volume_'):
            self._data.OrderBook.bid_volume[level] = value
        # elif name.startswith('bid_order_'):
        #     self._data.OrderBook.bid_n_orders[level] = value
        elif name.startswith('ask_price_'):
            self._data.OrderBook.ask_price[level] = value
        elif name.startswith('ask_volume_'):
            self._data.OrderBook.ask_volume[level] = value
        # elif name.startswith('ask_order_'):
        #     self._data.OrderBook.ask_n_orders[level] = value

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        """
        Implement the buffer protocol for read-only access.
        """
        if self._data == NULL:
            raise ValueError("Cannot get buffer from uninitialized data")
        PyBuffer_FillInfo(buffer, self, <_OrderBookBuffer*>self._data, sizeof(_OrderBookBuffer), 1, flags)

    def update_bid(self, list bid_data):
        """
        Update the bid side of the order book.

        Args:
            bid_data (list): List of bid entries [price, volume, n_orders].
        """
        cdef double price, volume
        cdef uint32_t n_orders
        cdef int i = 0

        # Clear existing bid data
        self._bid.clear()

        # Update the Cython buffer
        for i in range(min(len(bid_data), BOOK_SIZE)):
            entry = bid_data[i]
            if len(entry) == 2:
                price, volume = entry
                n_orders = 0
                self._data.OrderBook.bid_price[i] = price
                self._data.OrderBook.bid_volume[i] = volume
            elif len(entry) == 3:
                price, volume, n_orders = entry
                self._data.OrderBook.bid_price[i] = price
                self._data.OrderBook.bid_volume[i] = volume
                self._data.OrderBook.bid_n_orders[i] = n_orders
            else:
                raise ValueError(f'Bid entry {entry} having too many dimensions.')

            # Update the Book object
            self._bid.update(price=price, volume=volume, n_orders=n_orders)

    def update_ask(self, list ask_data):
        """
        Update the ask side of the order book.

        Args:
            ask_data (list): List of ask entries [price, volume, n_orders].
        """
        cdef double price, volume
        cdef uint32_t n_orders
        cdef int i = 0

        # Clear existing ask data
        self._ask.clear()

        # Update the Cython buffer
        for i in range(min(len(ask_data), BOOK_SIZE)):
            entry = ask_data[i]
            if len(entry) == 2:
                price, volume = entry
                n_orders = 0
                self._data.OrderBook.ask_price[i] = price
                self._data.OrderBook.ask_volume[i] = volume
            elif len(entry) == 3:
                price, volume, n_orders = entry
                self._data.OrderBook.ask_price[i] = price
                self._data.OrderBook.ask_volume[i] = volume
                self._data.OrderBook.ask_n_orders[i] = n_orders
            else:
                raise ValueError(f'Ask entry {entry} having too many dimensions.')

            # Update the Book object
            self._ask.update(price=price, volume=volume, n_orders=n_orders)

    @staticmethod
    def from_buffer(const unsigned char[:] buffer):
        """
        Create an OrderBook instance from a buffer.

        Args:
            buffer (memoryview): Buffer containing OrderBook data.

        Returns:
            OrderBook: New OrderBook instance.
        """
        cdef OrderBook instance = OrderBook.__new__(OrderBook)

        # Set up the instance
        instance._owner = True
        instance._data = <_MarketDataBuffer*>PyMem_Malloc(sizeof(_OrderBookBuffer))

        # Copy data from buffer
        memcpy(instance._data, &buffer[0], sizeof(_OrderBookBuffer))

        # Initialize book sides
        instance._bid = Book(1)  # positive for bid
        instance._ask = Book(-1)  # negative for ask

        # Populate the book sides from the buffer
        instance._update_books_from_buffer()

        return instance

    def to_bytes(self):
        """
        Convert the order book to bytes.

        Returns:
            bytes: Serialized order book data.
        """
        if self._data == NULL:
            raise ValueError("Cannot convert uninitialized data to bytes")

        # Update the buffer from the Book objects
        self._update_buffer_from_books()

        return PyBytes_FromStringAndSize(<char*>self._data, sizeof(_OrderBookBuffer))

    @classmethod
    def from_bytes(cls, bytes data):
        """
        Update the order book from bytes.

        Args:
            data (bytes): Serialized order book data.

        Returns:
            OrderBook: Self for method chaining.
        """
        cdef OrderBook instance = cls.__new__(cls)
        cdef const char * data_ptr = <const char *> data

        instance._owner = True
        instance._data = <_MarketDataBuffer *> PyMem_Malloc(sizeof(_OrderBookBuffer))

        if instance._data == NULL:
            raise MemoryError("Failed to allocate memory for BarData")

        memcpy(instance._data, data_ptr, sizeof(_OrderBookBuffer))

        # Recreate book sides
        instance._bid = Book(1)
        instance._ask = Book(-1)

        # Populate the book sides from the buffer
        instance._update_books_from_buffer()

        return instance

    cdef void _update_books_from_buffer(self):
        cdef int i

        self._bid.clear()
        self._ask.clear()

        for i in range(BOOK_SIZE):
            if self._data.OrderBook.bid_price[i]:
                self._bid.update(
                    price=self._data.OrderBook.bid_price[i],
                    volume=self._data.OrderBook.bid_volume[i],
                    n_orders=self._data.OrderBook.bid_n_orders[i]
                )

            if self._data.OrderBook.ask_price[i]:
                self._ask.update(
                    price=self._data.OrderBook.ask_price[i],
                    volume=self._data.OrderBook.ask_volume[i],
                    n_orders=self._data.OrderBook.ask_n_orders[i]
                )


    cdef void _update_buffer_from_books(self):
        """
        Update the internal buffer from the Book objects.
        """
        cdef int i

        # Update bid side
        self._bid.sort()
        valid_idx = min(len(self._bid), BOOK_SIZE)
        for i in range(valid_idx):
            entry = self._bid._book[i]
            self._data.OrderBook.bid_price[i] = entry[0]
            self._data.OrderBook.bid_volume[i] = entry[1]
            self._data.OrderBook.bid_n_orders[i] = entry[2]

        for i in range(valid_idx, BOOK_SIZE):
            self._data.OrderBook.bid_price[i] = 0
            self._data.OrderBook.bid_volume[i] = 0
            self._data.OrderBook.bid_n_orders[i] = 0

        # Update ask side
        self._ask.sort()
        valid_idx = min(len(self._ask), BOOK_SIZE)
        for i in range(valid_idx):
            entry = self._ask._book[i]
            self._data.OrderBook.ask_price[i] = entry[0]
            self._data.OrderBook.ask_volume[i] = entry[1]
            self._data.OrderBook.ask_n_orders[i] = entry[2]

        for i in range(valid_idx, BOOK_SIZE):
            self._data.OrderBook.ask_price[i] = 0
            self._data.OrderBook.ask_volume[i] = 0
            self._data.OrderBook.ask_n_orders[i] = 0

    @property
    def bid(self):
        """
        Get the bid side of the order book.

        Returns:
            Book: The bid side.
        """
        return self._bid

    @property
    def ask(self):
        """
        Get the ask side of the order book.

        Returns:
            Book: The ask side.
        """
        return self._ask

    @property
    def best_bid_price(self):
        """
        Get the best bid price in the order book.

        Returns:
            float: The best bid price, or NaN if not available.
        """
        if self._bid:
            return self._bid.best_price
        return float('nan')

    @property
    def best_ask_price(self):
        """
        Get the best ask price in the order book.

        Returns:
            float: The best ask price, or NaN if not available.
        """
        if self._ask:
            return self._ask.best_price
        return float('nan')

    @property
    def best_bid_volume(self):
        """
        Get the best bid volume in the order book.

        Returns:
            float: The best bid volume, or NaN if not available.
        """
        if self._bid:
            return self._bid.best_volume
        return float('nan')

    @property
    def best_ask_volume(self):
        """
        Get the best ask volume in the order book.

        Returns:
            float: The best ask volume, or NaN if not available.
        """
        if self._ask:
            return self._ask.best_volume
        return float('nan')

    @property
    def spread(self) -> float:
        if not (self._bid and self._ask):
            return float('nan')
        return self._ask.best_volume - self._bid.best_volume

    @property
    def mid_price(self) -> float:
        if not (self.bid and self.ask):
            return float('nan')
        return (self._ask.best_volume + self._bid.best_volume) / 2
