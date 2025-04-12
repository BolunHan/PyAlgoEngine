# cython: language_level=3
import enum
import uuid

from cpython cimport PyList_Size, PyList_GET_ITEM
from cpython.buffer cimport PyBuffer_FillInfo
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.mem cimport PyMem_Malloc
from libc.stdint cimport uint8_t, int8_t
from libc.math cimport INFINITY, NAN, copysign, fabs
from libc.string cimport memcpy, memset, memcmp

from .market_data cimport MarketData, _MarketDataBuffer, _TransactionDataBuffer, _OrderDataBuffer, _ID, DataType, ID_SIZE, Direction, Offset, Side, OrderType as OrderTypeCython


class OrderType(enum.IntEnum):
    ORDER_UNKNOWN = OrderTypeCython.ORDER_UNKNOWN
    ORDER_CANCEL = OrderTypeCython.ORDER_CANCEL
    ORDER_GENERIC = OrderTypeCython.ORDER_GENERIC
    ORDER_LIMIT = OrderTypeCython.ORDER_LIMIT
    ORDER_LIMIT_MAKER = OrderTypeCython.ORDER_LIMIT_MAKER
    ORDER_MARKET = OrderTypeCython.ORDER_MARKET
    ORDER_FOK = OrderTypeCython.ORDER_FOK
    ORDER_FAK = OrderTypeCython.ORDER_FAK
    ORDER_IOC = OrderTypeCython.ORDER_IOC


# Python wrapper for Direction enum
class TransactionDirection(enum.IntEnum):
    """
    Direction enum for transaction sides.
    """
    DIRECTION_UNKNOWN = Direction.DIRECTION_UNKNOWN  # 1
    DIRECTION_SHORT = Direction.DIRECTION_SHORT  # 0
    DIRECTION_LONG = Direction.DIRECTION_LONG  # 2

    def __or__(self, offset):
        """
        Combine with TransactionOffset to get TransactionSide.

        Args:
            offset (TransactionOffset): The offset to combine with

        Returns:
            TransactionSide: The combined transaction side
        """
        if not isinstance(offset, TransactionOffset):
            raise TypeError(f'{self.__class__.__name__} Can only merge with a TransactionOffset.')

        combined_value = self.value + offset.value
        side = TransactionSide(combined_value)

        if side is TransactionSide.FAULTY:
            raise ValueError(f"Combination of {self.name} and {offset.name} doesn't correspond to a valid TransactionSide")

        return side

    @property
    def sign(self):
        return self.value - 1


# Python wrapper for Offset enum
class TransactionOffset(enum.IntEnum):
    """
    Offset enum for transaction sides.
    """
    OFFSET_CANCEL = Offset.OFFSET_CANCEL  # 0
    OFFSET_ORDER = Offset.OFFSET_ORDER  # 4
    OFFSET_OPEN = Offset.OFFSET_OPEN  # 8
    OFFSET_CLOSE = Offset.OFFSET_CLOSE  # 16

    def __or__(self, direction):
        """
        Combine with TransactionDirection to get TransactionSide.

        Args:
            direction (TransactionDirection): The direction to combine with

        Returns:
            TransactionSide: The combined transaction side
        """
        if not isinstance(direction, TransactionDirection):
            raise TypeError(f'{self.__class__.__name__} Can only merge with a TransactionDirection.')

        combined_value = self.value + direction.value
        side = TransactionSide(combined_value)

        if side is TransactionSide.FAULTY:
            raise ValueError(f"Combination of {self.name} and {direction.name} doesn't correspond to a valid TransactionSide")

        return side


# Python wrapper for TransactionSide enum
class TransactionSide(enum.IntEnum):
    """
    Transaction side enum combining direction and offset.
    """
    # Long Side
    SIDE_LONG_OPEN = Side.SIDE_LONG_OPEN  # 2 + 8 = 10
    SIDE_LONG_CLOSE = Side.SIDE_LONG_CLOSE  # 2 + 16 = 18
    SIDE_LONG_CANCEL = Side.SIDE_LONG_CANCEL  # 2 + 0 = 2

    # Short Side
    SIDE_SHORT_OPEN = Side.SIDE_SHORT_OPEN  # 0 + 8 = 8
    SIDE_SHORT_CLOSE = Side.SIDE_SHORT_CLOSE  # 0 + 16 = 16
    SIDE_SHORT_CANCEL = Side.SIDE_SHORT_CANCEL  # 0 + 0 = 0

    # Order
    SIDE_BID = Side.SIDE_BID  # 2 + 4 = 6
    SIDE_ASK = Side.SIDE_ASK  # 0 + 4 = 4

    # Generic Cancel
    SIDE_CANCEL = Side.SIDE_CANCEL  # 1 + 0 = 1

    # C Alias
    SIDE_UNKNOWN = Side.SIDE_CANCEL  # 1
    SIDE_LONG = Side.SIDE_LONG_OPEN  # 10
    SIDE_SHORT = Side.SIDE_SHORT_OPEN  # 8

    # Backward compatibility
    ShortOrder = AskOrder = Ask = SIDE_ASK
    LongOrder = BidOrder = Bid = SIDE_BID

    ShortFilled = Unwind = Sell = SIDE_SHORT_CLOSE
    LongFilled = LongOpen = Buy = SIDE_LONG_OPEN

    ShortOpen = Short = SIDE_SHORT_OPEN
    Cover = SIDE_LONG_CLOSE

    UNKNOWN = CANCEL = SIDE_CANCEL
    FAULTY = -1

    def __neg__(self) -> TransactionSide:
        """
        Get the opposite side.
        """
        return self.__class__(TransactionHelper.get_opposite(self.value))

    @classmethod
    def _missing_(cls, value):
        """
        Handle missing values in the enumeration.

        Args:
            value (str | int): The value to resolve.

        Returns:
            TransactionSide: The resolved transaction side, or UNKNOWN if not recognized.
        """
        side_str = str(value).lower()

        if side_str in ('long', 'buy', 'b'):
            trade_side = cls.SIDE_LONG_OPEN
        elif side_str in ('short', 'sell', 's'):
            trade_side = cls.SIDE_SHORT_CLOSE
        elif side_str in ('short', 'ss'):
            trade_side = cls.SIDE_SHORT_OPEN
        elif side_str in ('cover', 'bc'):
            trade_side = cls.SIDE_LONG_CLOSE
        elif side_str == 'ask':
            trade_side = cls.SIDE_ASK
        elif side_str == 'bid':
            trade_side = cls.SIDE_BID
        else:
            try:
                trade_side = cls.__getitem__(value)
            except Exception:
                trade_side = cls.FAULTY
                print(f'WARNING: {value} is not recognized, return TransactionSide.FAULTY')

        return trade_side

    @property
    def sign(self) -> int:
        """
        Get the sign of the transaction side.
        """
        return TransactionHelper.get_sign(self.value)

    @property
    def offset(self) -> TransactionOffset:
        """
        Get the offset of the transaction side.

        Returns:
            TransactionOffset: The offset value.
        """
        return TransactionOffset(TransactionHelper.get_offset(self.value))

    @property
    def direction(self) -> TransactionDirection:
        """
        Get the direction of the transaction side.

        Returns:
            TransactionDirection: The direction value.
        """
        return TransactionDirection(TransactionHelper.get_direction(self.value))

    @property
    def side_name(self) -> str:
        """
        Get the name of the transaction side.
        """
        return TransactionHelper.get_side_name(self.value).decode('utf-8')

    @property
    def offset_name(self) -> str:
        """
        Get the name of the offset.
        """
        return TransactionHelper.get_offset_name(self.value).decode('utf-8')

    @property
    def direction_name(self) -> str:
        """
        Get the name of the direction.
        """
        return TransactionHelper.get_direction_name(self.value).decode('utf-8')


# Helper class for TransactionSide
cdef class TransactionHelper:
    """
    Helper class to manage TransactionSide behaviors.
    """

    @staticmethod
    cdef uint8_t get_opposite(uint8_t side):
        """
        Get the opposite side by flipping the direction (long -> short, short -> long).
        The offset remains unchanged.
        """
        cdef uint8_t direction = side & 0x03  # Extract the direction bits (0x03 = 00000011)
        cdef uint8_t offset = side & 0xFC     # Extract the offset bits (0xFC = 11111100)

        # Flip the direction: long -> short, short -> long
        if direction == Direction.DIRECTION_LONG:
            direction = Direction.DIRECTION_SHORT
        elif direction == Direction.DIRECTION_SHORT:
            direction = Direction.DIRECTION_LONG
        # If direction is unknown, leave it unchanged

        # Combine the new direction with the original offset
        return direction | offset

    @staticmethod
    cdef int8_t get_sign(uint8_t side):
        """
        Get the sign of the transaction side.
        """
        cdef int direction = side & 0x03
        return direction - 1

    @staticmethod
    cdef uint8_t get_offset(uint8_t side):
        """
        Returns the offset of the given side.
        """
        return side & 0xFC  # Mask to get the offset bits (0xFC = 11111100)

    @staticmethod
    cdef uint8_t get_direction(uint8_t side):
        """
        Returns the direction of the given side.
        """
        return side & 0x03  # Mask to get the direction bits (0x03 = 00000011)

    @staticmethod
    cdef const char* get_side_name(uint8_t side):
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
            return f"unknown({side})".encode('utf-8')

    @staticmethod
    cdef const char* get_order_type_name(uint8_t order_type):
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
            return f"unknown({order_type})".encode('utf-8')

    @staticmethod
    cdef const char* get_direction_name(uint8_t side):
        """
        Returns the name of the direction of the given side.
        """
        cdef uint8_t direction = TransactionHelper.get_direction(side)

        if direction == Direction.DIRECTION_SHORT:
            return "short"
        elif direction == Direction.DIRECTION_LONG:
            return "long"
        else:
            return "unknown"

    @staticmethod
    cdef const char* get_offset_name(uint8_t side):
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


cdef class TransactionData(MarketData):
    """
    Represents transaction data for a specific market.
    """
    _dtype = DataType.DTYPE_TRANSACTION

    def __cinit__(self):
        """
        Allocate memory for the transaction data structure but don't initialize it.
        """
        self._owner = True

    def __init__(self, ticker: str, double timestamp, double price, double volume, int side, double multiplier=1.0, double notional=0.0, object transaction_id=None, object buy_id=None, object sell_id=None, **kwargs):
        """
        Initialize the transaction data with values.
        """
        if not self._owner:
            raise MemoryError(f"Can not initialize a view of {self.__class__.__name__}.")

        self._data = <_MarketDataBuffer*> PyMem_Malloc(sizeof(_TransactionDataBuffer))
        memset(self._data, 0, sizeof(_TransactionDataBuffer))

        # Initialize base class fields
        MarketData.__init__(self, ticker=ticker, timestamp=timestamp, **kwargs)

        # Set data type for TransactionData
        # self._data.MetaInfo.dtype = DataType.DTYPE_TRANSACTION

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
        TransactionData._set_id(id_ptr=&self._data.TransactionData.transaction_id, id_value=transaction_id)
        TransactionData._set_id(id_ptr=&self._data.TransactionData.buy_id, id_value=buy_id)
        TransactionData._set_id(id_ptr=&self._data.TransactionData.sell_id, id_value=sell_id)

    def __repr__(self) -> str:
        """
        String representation of the order data.
        """
        if self._data == NULL:
            return "TransactionData(uninitialized)"
        side_name = TransactionHelper.get_side_name(self._data.TransactionData.side).decode('utf-8')

        return f"<TransactionData>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker}, price={self.price}, volume={self.volume}, side={side_name})"

    @staticmethod
    cdef void _set_id(_ID* id_ptr, object id_value):
        """
        Set an ID field in the transaction data.
        """
        cdef bytes id_bytes
        cdef int id_len

        if id_value is None:
            id_ptr.id_type = 0  # None type
        elif isinstance(id_value, int):
            id_ptr.id_type = 1  # Int type
            id_bytes = id_value.to_bytes(ID_SIZE, byteorder='little', signed=True)
            # memset(id_ptr.data, 0, ID_SIZE)
            memcpy(id_ptr.data, <char*> id_bytes, ID_SIZE)
        elif isinstance(id_value, str):
            id_ptr.id_type = 2  # Str type
            id_bytes = id_value.encode('utf-8')
            id_len = min(len(id_bytes), ID_SIZE)
            memset(id_ptr.data, 0, ID_SIZE)
            memcpy(id_ptr.data, <char*> id_bytes, id_len)
        elif isinstance(id_value, bytes):
            id_ptr.id_type = 3  # Bytes type
            id_len = min(len(id_value), ID_SIZE)
            memset(id_ptr.data, 0, ID_SIZE)
            memcpy(id_ptr.data, <char*> id_value, id_len)
        elif isinstance(id_value, uuid.UUID):
            id_ptr.id_type = 4  # UUID type
            id_bytes = id_value.bytes_le
            memset(id_ptr.data, 0, ID_SIZE)
            memcpy(id_ptr.data, <char*> id_bytes, 16)

    @staticmethod
    cdef object _get_id(_ID* id_ptr):
        """
        Get an ID field from the transaction data.
        """
        if id_ptr.id_type == 0:
            return None
        elif id_ptr.id_type == 1:  # Int type
            return int.from_bytes(id_ptr.data[:ID_SIZE], byteorder='little', signed=True)
        elif id_ptr.id_type == 2:  # Str type
            return id_ptr.data.decode('utf-8').rstrip('\0')
        elif id_ptr.id_type == 3:  # Bytes type
            return PyBytes_FromStringAndSize(id_ptr.data, ID_SIZE).rstrip(b'\0')
        elif id_ptr.id_type == 4:  # UUID type
            return uuid.UUID(bytes_le=id_ptr.data[:16])

        raise ValueError(f'Can not decode the id buffer with type {id_ptr.id_type}.')

    @staticmethod
    cdef bint _id_equal(const _ID* id1, const _ID* id2):
        """Ultra-efficient ID comparison using single memcmp of entire struct"""
        return memcmp(id1, id2, ID_SIZE + 1) == 0

    @classmethod
    def merge(cls, list data_list not None):
        """
        Merges multiple TransactionData instances into a single aggregated instance.

        Args:
            data_list: List[TransactionData] - Non-empty list of TransactionData to merge

        Returns:
            TransactionData - Merged transaction data

        Raises:
            ValueError: If list is empty
            AssertionError: If tickers or multipliers don't match
        """
        cdef Py_ssize_t i, n = PyList_Size(data_list)
        cdef TransactionData first, td
        cdef str ticker
        cdef double multiplier, timestamp = 0.0
        cdef double sum_volume = 0.0, sum_notional = 0.0
        cdef double trade_price, trade_volume, trade_notional
        cdef int trade_sign, trade_side

        if n == 0:
            raise ValueError('Data list empty')

        first = <TransactionData> PyList_GET_ITEM(data_list, 0)
        ticker = first.ticker
        multiplier = first.multiplier

        # Single combined loop for validation and calculations
        for i in range(n):
            td = <TransactionData> PyList_GET_ITEM(data_list, i)

            # Validation
            if td.ticker != ticker:
                raise AssertionError(f'Ticker mismatch, expect {ticker}, got {td.ticker}')
            if td.multiplier != multiplier:
                raise AssertionError(f'Multiplier mismatch, expect {multiplier}, got {td.multiplier}')

            # Calculations
            sum_volume += td.volume_flow
            sum_notional += td.notional_flow
            if td.timestamp > timestamp:
                timestamp = td.timestamp

        # Determine trade parameters using copysign and fabs
        trade_side = Side.SIDE_LONG if sum_volume > 0 else Side.SIDE_SHORT if sum_volume < 0 else Side.SIDE_UNKNOWN
        trade_volume = fabs(sum_volume)
        trade_notional = fabs(sum_notional)

        # Price calculation with copysign for infinity cases
        if sum_notional == 0:
            trade_price = 0.0
        elif sum_volume == 0:
            trade_price = NAN
        else:
            trade_price = sum_notional / sum_volume if sum_volume != 0 else copysign(INFINITY, sum_notional)

        return cls(
            ticker=ticker,
            timestamp=timestamp,
            price=trade_price,
            volume=trade_volume,
            side=trade_side,
            multiplier=multiplier,
            notional=trade_notional
        )

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
    def side_int(self) -> int:
        """
        Get the transaction side.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TransactionData.side

    @property
    def side(self) -> TransactionSide:
        return TransactionSide(self.side_int)

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
        return TransactionData._get_id(&self._data.TransactionData.transaction_id)

    @property
    def buy_id(self) -> object:
        """
        Get the buy ID.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return TransactionData._get_id(&self._data.TransactionData.buy_id)

    @property
    def sell_id(self) -> object:
        """
        Get the sell ID.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return TransactionData._get_id(&self._data.TransactionData.sell_id)

    @property
    def market_price(self) -> float:
        """
        Alias for the transaction price.
        """
        return self.price

    @property
    def volume_flow(self) -> float:
        """
        Calculate the flow of the transaction volume.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        cdef int sign = TransactionHelper.get_sign(self._data.TransactionData.side)
        return sign * self._data.TransactionData.volume

    @property
    def notional_flow(self) -> float:
        """
        Calculate the flow of the transaction notional.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        cdef int sign = TransactionHelper.get_sign(self._data.TransactionData.side)
        return sign * self._data.TransactionData.notional


cdef class OrderData(MarketData):
    """
    Represents order data for a specific market.
    """
    _dtype = DataType.DTYPE_ORDER

    def __cinit__(self):
        """
        Allocate memory for the order data structure but don't initialize it.
        """
        self._owner = True

    def __init__(self, str ticker, double timestamp, double price, double volume, uint8_t side, object order_id=None, uint8_t order_type=0, **kwargs):
        """
        Initialize the order data with values.
        """
        if not self._owner:
            raise MemoryError(f"Can not initialize a view of {self.__class__.__name__}.")

        self._data = <_MarketDataBuffer*> PyMem_Malloc(sizeof(_OrderDataBuffer))
        memset(self._data, 0, sizeof(_OrderDataBuffer))

        # Initialize base class fields
        MarketData.__init__(self, ticker=ticker, timestamp=timestamp, **kwargs)

        # Set data type for OrderData
        # self._data.MetaInfo.dtype = DataType.DTYPE_ORDER

        # Initialize order-specific fields
        self._data.OrderData.price = price
        self._data.OrderData.volume = volume
        self._data.OrderData.side = side
        self._data.OrderData.order_type = order_type

        # Initialize order_id
        TransactionData._set_id(id_ptr=&self._data.OrderData.order_id, id_value=order_id)

    def __repr__(self) -> str:
        """
        String representation of the order data.
        """
        if self._data == NULL:
            return "<OrderData>(uninitialized)"

        side_name = TransactionHelper.get_side_name(self._data.OrderData.side).decode('utf-8')
        order_type_name = TransactionHelper.get_order_type_name(self._data.OrderData.order_type).decode('utf-8')

        return f"<OrderData>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker}, price={self.price}, volume={self.volume}, side={side_name}, order_type={order_type_name})"

    @classmethod
    def from_buffer(cls, const unsigned char[:] buffer):
        """
        Create a new instance from a buffer.
        """
        cdef OrderData instance = cls.__new__(cls)

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
        cdef OrderData instance = cls.__new__(cls)
        cdef const char* data_ptr = <const char*>data

        instance._owner = True
        instance._data = <_MarketDataBuffer*>PyMem_Malloc(sizeof(_OrderDataBuffer))

        if instance._data == NULL:
            raise MemoryError("Failed to allocate memory for OrderData")

        memcpy(instance._data, data_ptr, sizeof(_OrderDataBuffer))

        return instance

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        """
        Implement the buffer protocol for read-only access.
        """
        if self._data == NULL:
            raise ValueError("Cannot get buffer from uninitialized data")
        PyBuffer_FillInfo(buffer, self, <_OrderDataBuffer*>self._data, sizeof(_OrderDataBuffer), 1, flags)

    def __copy__(self):
        """
        Create a copy of this instance.
        """
        if self._data == NULL:
            raise ValueError("Cannot copy uninitialized data")

        cdef OrderData new_od = OrderData.__new__(OrderData)
        # Allocate memory specifically for an OrderData buffer
        new_od._data = <_MarketDataBuffer*>PyMem_Malloc(sizeof(_OrderDataBuffer))
        if new_od._data == NULL:
            raise MemoryError("Failed to allocate memory for copy")

        new_od._owner = True
        memcpy(new_od._data, self._data, sizeof(_OrderDataBuffer))

        return new_od

    @property
    def price(self) -> float:
        """
        Get the order price.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.OrderData.price

    @property
    def volume(self) -> float:
        """
        Get the order volume.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.OrderData.volume

    @property
    def side_int(self) -> int:
        """
        Get the order side.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.OrderData.side

    @property
    def side(self) -> TransactionSide:
        return TransactionSide(self.side_int)

    @property
    def order_id(self) -> object:
        """
        Get the order ID.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return TransactionData._get_id(&self._data.OrderData.order_id)

    @property
    def order_type_int(self) -> int:
        """
        Get the order type.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.OrderData.order_type

    @property
    def order_type(self) -> OrderType:
        """
        Get the order type.
        """
        return OrderType(self.order_type_int)

    @property
    def market_price(self) -> float:
        """
        Alias for the order price.
        """
        return self.price

    @property
    def flow(self) -> float:
        """
        Calculate the flow of the order.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        cdef int sign = TransactionHelper.get_sign(self._data.OrderData.side)
        return sign * self._data.OrderData.volume

cdef class TradeData(TransactionData):
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

    def __init__(self, str ticker, double timestamp, double trade_price, double trade_volume, int side, object order_id=None, int order_type=0, **kwargs):
        super().__init__(ticker=ticker, timestamp=timestamp, price=trade_price, volume=trade_volume, side=side, order_id=order_id, order_type=order_type, **kwargs)

    @property
    def trade_price(self) -> float:
        return super().price

    @property
    def trade_volume(self) -> float:
        return super().volume
