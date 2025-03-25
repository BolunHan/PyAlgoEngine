# cython: language_level=3
from cpython.buffer cimport PyBuffer_FillInfo
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.datetime cimport datetime, date
from cpython.mem cimport PyMem_Free
from libc.stdint cimport uint8_t
from libc.string cimport memcpy, memset

from ..profile import PROFILE

# Helper class for TransactionSide
cdef class TransactionHelper:
    """
    Helper class to manage TransactionSide behaviors.
    """

    @staticmethod
    cdef int get_opposite(int side):
        """
        Get the opposite side by flipping the direction (long -> short, short -> long).
        The offset remains unchanged.
        """
        cdef int direction = side & 0x03  # Extract the direction bits (0x03 = 00000011)
        cdef int offset = side & 0xFC     # Extract the offset bits (0xFC = 11111100)

        # Flip the direction: long -> short, short -> long
        if direction == DIRECTION_LONG:
            direction = DIRECTION_SHORT
        elif direction == DIRECTION_SHORT:
            direction = DIRECTION_LONG
        # If direction is unknown, leave it unchanged

        # Combine the new direction with the original offset
        return direction | offset

    @staticmethod
    cdef int get_sign(int side):
        """
        Get the sign of the transaction side.
        """
        cdef int direction = side & 0x03
        return direction - 1

    @staticmethod
    cdef int get_offset(int side):
        """
        Returns the offset of the given side.
        """
        return side & 0xFC  # Mask to get the offset bits (0xFC = 11111100)

    @staticmethod
    cdef int get_direction(int side):
        """
        Returns the direction of the given side.
        """
        return side & 0x03  # Mask to get the direction bits (0x03 = 00000011)

    @staticmethod
    cdef const char* get_side_name(int side):
        """
        Returns the name of the given side.
        """
        if side == Side.SIDE_LONG_OPEN:
            return "buy"
        elif side == Side.SIDE_LONG_CLOSE:
            return "cover"
        elif side == Side.SIDE_LONG_CANCEL:
            return "cancel bid"
        elif side == Side.SIDE_SHORT_OPEN:
            return "short"
        elif side == Side.SIDE_SHORT_CLOSE:
            return "sell"
        elif side == Side.SIDE_SHORT_CANCEL:
            return "cancel ask"
        elif side == Side.SIDE_BID:
            return "bid"
        elif side == Side.SIDE_ASK:
            return "ask"
        elif side == Side.SIDE_CANCEL:
            return "cancel"
        else:
            return f"unknown({side})"

    @staticmethod
    cdef const char* get_order_type_name(int order_type):
        """
        Get the string representation of the order type.
        """
        if order_type == OrderType.ORDER_UNKNOWN:
            return "unknown"
        elif order_type == OrderType.ORDER_CANCEL:
            return "cancel"
        elif order_type == OrderType.ORDER_GENERIC:
            return "generic"
        elif order_type == OrderType.ORDER_LIMIT:
            return "limit"
        elif order_type == OrderType.ORDER_LIMIT_MAKER:
            return "limit_maker"
        elif order_type == OrderType.ORDER_MARKET:
            return "market"
        elif order_type == OrderType.ORDER_FOK:
            return "fok"
        elif order_type == OrderType.ORDER_FAK:
            return "fak"
        elif order_type == OrderType.ORDER_IOC:
            return "ioc"
        else:
            return f"unknown({order_type})"

    @staticmethod
    cdef const char* get_direction_name(int side):
        """
        Returns the name of the direction of the given side.
        """
        cdef int direction = TransactionHelper.get_direction(side)

        if direction == Direction.DIRECTION_SHORT:
            return "short"
        elif direction == Direction.DIRECTION_LONG:
            return "long"
        else:
            return "unknown"

    @staticmethod
    cdef const char* get_offset_name(int side):
        """
        Returns the name of the offset of the given side.
        """
        cdef int offset = TransactionHelper.get_offset(side)

        if offset == Offset.OFFSET_CANCEL:
            return "cancel"
        elif offset == Offset.OFFSET_ORDER:
            return "order"
        elif offset == Offset.OFFSET_OPEN:
            return "open"
        elif offset == Offset.OFFSET_CLOSE:
            return "close"
        else:
            return "unknown"

    @classmethod
    def pyget_opposite(cls, int side) -> int:
        return TransactionHelper.get_opposite(side=side)

    @classmethod
    def pyget_sign(cls, int side) -> int:
        return TransactionHelper.get_sign(side=side)

    @classmethod
    def pyget_direction(cls, int side) -> int:
        return TransactionHelper.get_direction(side=side)

    @classmethod
    def pyget_offset(cls, int side) -> int:
        return TransactionHelper.get_offset(side=side)

    @classmethod
    def pyget_side_name(cls, int side) -> str:
        if side == Side.SIDE_LONG_OPEN:
            return "buy"
        elif side == Side.SIDE_LONG_CLOSE:
            return "cover"
        elif side == Side.SIDE_LONG_CANCEL:
            return "cancel bid"
        elif side == Side.SIDE_SHORT_OPEN:
            return "short"
        elif side == Side.SIDE_SHORT_CLOSE:
            return "sell"
        elif side == Side.SIDE_SHORT_CANCEL:
            return "cancel ask"
        elif side == Side.SIDE_BID:
            return "bid"
        elif side == Side.SIDE_ASK:
            return "ask"
        elif side == Side.SIDE_CANCEL:
            return "cancel"
        else:
            return f"unknown({side})"

    @classmethod
    def pyget_order_type_name(cls, int order_type) -> str:
        if order_type == OrderType.ORDER_UNKNOWN:
            return "unknown"
        elif order_type == OrderType.ORDER_CANCEL:
            return "cancel"
        elif order_type == OrderType.ORDER_GENERIC:
            return "generic"
        elif order_type == OrderType.ORDER_LIMIT:
            return "limit"
        elif order_type == OrderType.ORDER_LIMIT_MAKER:
            return "limit_maker"
        elif order_type == OrderType.ORDER_MARKET:
            return "market"
        elif order_type == OrderType.ORDER_FOK:
            return "fok"
        elif order_type == OrderType.ORDER_FAK:
            return "fak"
        elif order_type == OrderType.ORDER_IOC:
            return "ioc"
        else:
            return f"unknown({order_type})"

    @classmethod
    def pyget_direction_name(cls, int side) -> str:
        cdef int direction = TransactionHelper.get_direction(side)

        if direction == Direction.DIRECTION_SHORT:
            return "short"
        elif direction == Direction.DIRECTION_LONG:
            return "long"
        else:
            return "unknown"

    @classmethod
    def pyget_offset_name(cls, int side) -> str:
        cdef int offset = TransactionHelper.get_offset(side)

        if offset == Offset.OFFSET_CANCEL:
            return "cancel"
        elif offset == Offset.OFFSET_ORDER:
            return "order"
        elif offset == Offset.OFFSET_OPEN:
            return "open"
        elif offset == Offset.OFFSET_CLOSE:
            return "close"
        else:
            return "unknown"

# Base MarketData class
cdef class MarketData:
    """
    Base class for all market data types.
    """
    # cdef _MarketDataBuffer* _data
    # cdef bint _owner
    # cdef int _dtype

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

    def __init__(self, str ticker, double timestamp, **kwargs):
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

        if kwargs:
            self._additional = kwargs.copy()

    def __reduce__(self):
        """Support for pickle serialization"""
        return self.__class__.from_bytes, (self.to_bytes(),), self._additional

    def __setstate__(self, state):
        """Restore state from pickle"""
        if state:
            self._additional = state.copy()

    def __copy__(self):
        return self.__class__.from_bytes(self.to_bytes())

    def __setattr__(self, key, value):
        if hasattr(self, key):
            raise AttributeError(f'{self.__class__.__name__} is readonly.')

        self._set_additional(name=key, value=value)

    def __getattr__(self, key):
        if key in self._additional:
            return self._additional[key]

        raise AttributeError(f'Can not find attribute {key}.')

    cdef void _set_additional(self, str name, object value):
        if self._additional is None:
            self._additional = {name: value}
        else:
            self._additional[name] = value

    @classmethod
    def from_buffer(cls, const unsigned char[:] buffer):
        ...

    @classmethod
    def from_bytes(cls, bytes data):
        ...

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
        elif dtype == DataType.DTYPE_ORDER:
            size = sizeof(_OrderDataBuffer)
        elif dtype == DataType.DTYPE_TICK_LITE:
            size = sizeof(_TickDataLiteBuffer)
        elif dtype == DataType.DTYPE_TICK:
            size = sizeof(_TickDataBuffer)
        elif dtype == DataType.DTYPE_BAR:
            size = sizeof(_CandlestickBuffer)
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
        elif dtype == DataType.DTYPE_ORDER:
            size = sizeof(_OrderDataBuffer)
        elif dtype == DataType.DTYPE_TICK_LITE:
            size = sizeof(_TickDataLiteBuffer)
        elif dtype == DataType.DTYPE_TICK:
            size = sizeof(_TickDataBuffer)
        elif dtype == DataType.DTYPE_BAR:
            size = sizeof(_CandlestickBuffer)
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

    @property
    def topic(self) -> str:
        return f'{self.ticker}.{self.__class__.__name__}'

    @property
    def market_time(self) -> datetime | date:
        return datetime.fromtimestamp(self.timestamp, tz=PROFILE.time_zone)
