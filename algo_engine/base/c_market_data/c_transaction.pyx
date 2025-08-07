# cython: language_level=3
import enum
import uuid
from typing import Literal

cimport cython
from cpython cimport PyList_Size, PyList_GET_ITEM
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.unicode cimport PyUnicode_FromString, PyUnicode_AsUTF8String, PyUnicode_AsUTF8AndSize
from libc.stdint cimport uint8_t, int8_t, uint64_t, UINT64_MAX
from libc.math cimport INFINITY, NAN, copysign, fabs
from libc.string cimport memcpy, memset, memcmp

from .c_market_data cimport direction_to_sign, _MarketDataVirtualBase, _MarketDataBuffer, TICKER_SIZE, _TransactionDataBuffer, _OrderDataBuffer, _ID, DataType, ID_SIZE, Direction, Offset, Side, OrderType as E_OrderType


class OrderType(enum.IntEnum):
    ORDER_UNKNOWN = E_OrderType.ORDER_UNKNOWN
    ORDER_CANCEL = E_OrderType.ORDER_CANCEL
    ORDER_GENERIC = E_OrderType.ORDER_GENERIC
    ORDER_LIMIT = E_OrderType.ORDER_LIMIT
    ORDER_LIMIT_MAKER = E_OrderType.ORDER_LIMIT_MAKER
    ORDER_MARKET = E_OrderType.ORDER_MARKET
    ORDER_FOK = E_OrderType.ORDER_FOK
    ORDER_FAK = E_OrderType.ORDER_FAK
    ORDER_IOC = E_OrderType.ORDER_IOC


# Python wrapper for Direction enum
class TransactionDirection(enum.IntEnum):
    """
    Direction enum for transaction sides.
    """
    DIRECTION_UNKNOWN = Direction.DIRECTION_UNKNOWN  # 1
    DIRECTION_SHORT = Direction.DIRECTION_SHORT  # 0
    DIRECTION_LONG = Direction.DIRECTION_LONG  # 2
    DIRECTION_NEUTRAL = Direction.DIRECTION_NEUTRAL  # 3

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
        return direction_to_sign(self.value)


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

    # Neutral Side
    SIDE_NEUTRAL_OPEN = Side.SIDE_NEUTRAL_OPEN  # 3 + 8 = 11
    SIDE_NEUTRAL_CLOSE = Side.SIDE_NEUTRAL_CLOSE  # 3 + 16 = 19

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
    FAULTY = 255

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
    def sign(self) -> Literal[-1 ,0, 1]:
        """
        Get the sign of the transaction side.
        """
        return direction_to_sign(self.value)

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
        return TransactionHelper.pyget_side_name(self.value)

    @property
    def offset_name(self) -> str:
        """
        Get the name of the offset.
        """
        return TransactionHelper.pyget_offset_name(self.value)

    @property
    def direction_name(self) -> str:
        """
        Get the name of the direction.
        """
        return TransactionHelper.pyget_direction_name(self.value)


# Helper class for TransactionSide
cdef class TransactionHelper:
    """
    Helper class to manage TransactionSide behaviors.
    """

    @staticmethod
    cdef void set_id(_ID* id_ptr, object id_value):
        """
        Set an ID field in the transaction data.
        """
        cdef bytes id_bytes
        cdef int id_len
        cdef uint64_t u64_val

        if id_value is None:
            id_ptr.id_type = 0  # None type

        elif isinstance(id_value, int):
            if 0 <= id_value <= UINT64_MAX:
                # uint64_t type
                id_ptr.id_type = 64
                u64_val = <uint64_t> id_value
                memset(<void*> id_ptr.data, 0, ID_SIZE)
                memcpy(<void*> id_ptr.data, &u64_val, sizeof(uint64_t))
            else:
                # signed int type
                id_ptr.id_type = 1
                id_bytes = id_value.to_bytes(ID_SIZE, byteorder='little', signed=True)
                memcpy(<void*> id_ptr.data, <const char*> id_bytes, ID_SIZE)

        elif isinstance(id_value, str):
            id_ptr.id_type = 2  # Str type
            id_bytes = PyUnicode_AsUTF8String(id_value)
            id_len = min(len(id_bytes), ID_SIZE)
            memset(<void*> id_ptr.data, 0, ID_SIZE)
            memcpy(<void*> id_ptr.data, <const char*> id_bytes, id_len)

        elif isinstance(id_value, bytes):
            id_ptr.id_type = 3  # Bytes type
            id_len = min(len(id_value), ID_SIZE)
            memset(<void*> id_ptr.data, 0, ID_SIZE)
            memcpy(<void*> id_ptr.data, <const char*> id_value, id_len)

        elif isinstance(id_value, uuid.UUID):
            id_ptr.id_type = 4  # UUID type
            id_bytes = id_value.bytes_le
            memset(<void*> id_ptr.data, 0, ID_SIZE)
            memcpy(<void*> id_ptr.data, <const char*> id_bytes, 16)

    @staticmethod
    cdef object get_id(_ID* id_ptr):
        """
        Get an ID field from the transaction data.
        """
        if id_ptr.id_type == 0:
            return None
        elif id_ptr.id_type == 64:  # uint64_t
            return (<uint64_t*> id_ptr.data)[0]
        elif id_ptr.id_type == 1:  # signed int
            return int.from_bytes(id_ptr.data[:ID_SIZE], byteorder='little', signed=True)
        elif id_ptr.id_type == 2:  # Str type
            return PyUnicode_FromString(&id_ptr.data[0]).rstrip('\0')
        elif id_ptr.id_type == 3:  # Bytes type
            return PyBytes_FromStringAndSize(id_ptr.data, ID_SIZE).rstrip(b'\0')
        elif id_ptr.id_type == 4:  # UUID type
            return uuid.UUID(bytes_le=id_ptr.data[:16])

        raise ValueError(f'Cannot decode the id buffer with type {id_ptr.id_type}.')

    @staticmethod
    cdef bint compare_id(const _ID* id1, const _ID* id2):
        return memcmp(id1, id2, ID_SIZE + 1) == 0

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
        return direction_to_sign(side)

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
    cdef bytes get_side_name(uint8_t side):
        """
        Returns the name of the given side.
        """
        if side == Side.SIDE_LONG_OPEN:
            return b"buy"
        elif side == Side.SIDE_LONG_CLOSE:
            return b"cover"
        elif side == Side.SIDE_LONG_CANCEL:
            return b"cancel bid"
        elif side == Side.SIDE_SHORT_OPEN:
            return b"short"
        elif side == Side.SIDE_SHORT_CLOSE:
            return b"sell"
        elif side == Side.SIDE_SHORT_CANCEL:
            return b"cancel ask"
        elif side == Side.SIDE_NEUTRAL_OPEN:
            return b"open"
        elif side == Side.SIDE_NEUTRAL_CLOSE:
            return b"close"
        elif side == Side.SIDE_BID:
            return b"bid"
        elif side == Side.SIDE_ASK:
            return b"ask"
        elif side == Side.SIDE_CANCEL:
            return b"cancel"
        else:
            return PyUnicode_AsUTF8String(f"unknown({side})")

    @staticmethod
    cdef bytes get_order_type_name(uint8_t order_type):
        """
        Get the string representation of the order type.
        """
        if order_type == E_OrderType.ORDER_UNKNOWN:
            return b"unknown"
        elif order_type == E_OrderType.ORDER_CANCEL:
            return b"cancel"
        elif order_type == E_OrderType.ORDER_GENERIC:
            return b"generic"
        elif order_type == E_OrderType.ORDER_LIMIT:
            return b"limit"
        elif order_type == E_OrderType.ORDER_LIMIT_MAKER:
            return b"limit_maker"
        elif order_type == E_OrderType.ORDER_MARKET:
            return b"market"
        elif order_type == E_OrderType.ORDER_FOK:
            return b"fok"
        elif order_type == E_OrderType.ORDER_FAK:
            return b"fak"
        elif order_type == E_OrderType.ORDER_IOC:
            return b"ioc"
        else:
            return PyUnicode_AsUTF8String(f"unknown({order_type})")

    @staticmethod
    cdef bytes get_direction_name(uint8_t side):
        """
        Returns the name of the direction of the given side.
        """
        cdef uint8_t direction = TransactionHelper.get_direction(side)

        if direction == Direction.DIRECTION_SHORT:
            return b"short"
        elif direction == Direction.DIRECTION_LONG:
            return b"long"
        elif direction == Direction.DIRECTION_NEUTRAL:
            return b'neutral'
        else:
            return b"unknown"

    @staticmethod
    cdef bytes get_offset_name(uint8_t side):
        """
        Returns the name of the offset of the given side.
        """
        cdef int offset = TransactionHelper.get_offset(side)

        if offset == Offset.OFFSET_CANCEL:
            return b"cancel"
        elif offset == Offset.OFFSET_ORDER:
            return b"order"
        elif offset == Offset.OFFSET_OPEN:
            return b"open"
        elif offset == Offset.OFFSET_CLOSE:
            return b"close"
        else:
            return b"unknown"

    @classmethod
    def pyget_opposite(cls, int side) -> int:
        return TransactionHelper.get_opposite(side=side)

    @classmethod
    def pyget_sign(cls, int side) -> Literal[-1, 0, 1]:
        return direction_to_sign(side)

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
        elif side == Side.SIDE_NEUTRAL_OPEN:
            return "open"
        elif side == Side.SIDE_NEUTRAL_CLOSE:
            return "close"
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
        if order_type == E_OrderType.ORDER_UNKNOWN:
            return "unknown"
        elif order_type == E_OrderType.ORDER_CANCEL:
            return "cancel"
        elif order_type == E_OrderType.ORDER_GENERIC:
            return "generic"
        elif order_type == E_OrderType.ORDER_LIMIT:
            return "limit"
        elif order_type == E_OrderType.ORDER_LIMIT_MAKER:
            return "limit_maker"
        elif order_type == E_OrderType.ORDER_MARKET:
            return "market"
        elif order_type == E_OrderType.ORDER_FOK:
            return "fok"
        elif order_type == E_OrderType.ORDER_FAK:
            return "fak"
        elif order_type == E_OrderType.ORDER_IOC:
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
        elif direction == Direction.DIRECTION_NEUTRAL:
            return "neutral"
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


@cython.freelist(128)
cdef class TransactionData:
    def __cinit__(self):
        self._data_ptr = <_MarketDataBuffer*> &self._data
        self._data_addr = <uintptr_t> self._data_ptr

    def __init__(self, *, ticker: str, double timestamp, double price, double volume, uint8_t side, double multiplier=1.0, double notional=0.0, object transaction_id=None, object buy_id=None, object sell_id=None, **kwargs):
        # Initialize base class fields
        cdef Py_ssize_t ticker_len
        cdef const char * ticker_ptr = PyUnicode_AsUTF8AndSize(ticker, &ticker_len)
        memcpy(<void *> &self._data.ticker, ticker_ptr, min(ticker_len, TICKER_SIZE - 1))
        self._data.timestamp = timestamp
        self._data.dtype = DataType.DTYPE_TRANSACTION
        if kwargs: self.__dict__.update(kwargs)

        # Initialize transaction-specific fields
        self._data.price = price
        self._data.volume = volume
        self._data.side = side
        self._data.multiplier = multiplier

        # Calculate notional if not provided
        if notional == 0.0:
            self._data.notional = price * volume * multiplier
        else:
            self._data.notional = notional

        # Initialize IDs
        TransactionHelper.set_id(id_ptr=&self._data.transaction_id, id_value=transaction_id)
        TransactionHelper.set_id(id_ptr=&self._data.buy_id, id_value=buy_id)
        TransactionHelper.set_id(id_ptr=&self._data.sell_id, id_value=sell_id)

    def __repr__(self) -> str:
        side_name = TransactionHelper.pyget_side_name(self._data.side)
        return f"<TransactionData>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker}, price={self.price}, volume={self.volume}, side={side_name})"

    def __reduce__(self):
        return self.__class__.from_bytes, (self.to_bytes(),), self.__dict__

    def __setstate__(self, state):
        if state:
            self.__dict__.update(state)

    def __copy__(self):
        cdef TransactionData instance = TransactionData.__new__(TransactionData)
        memcpy(<void*> &instance._data, <const char*> &self._data, sizeof(_TransactionDataBuffer))
        return instance

    @classmethod
    def buffer_size(cls):
        return sizeof(_TransactionDataBuffer)

    cdef bytes c_to_bytes(self):
        return PyBytes_FromStringAndSize(<char*> &self._data, sizeof(self._data))

    @staticmethod
    cdef TransactionData c_from_bytes(bytes data):
        cdef TransactionData instance = TransactionData.__new__(TransactionData)
        memcpy(<void*> &instance._data, <const char*> data, sizeof(_TransactionDataBuffer))
        return instance

    def to_bytes(self) -> bytes:
        return self.c_to_bytes()

    @classmethod
    def from_bytes(cls, bytes data):
        return TransactionData.c_from_bytes(data)

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
        trade_side = Side.SIDE_LONG if sum_volume > 0 else Side.SIDE_SHORT if sum_volume < 0 else Side.SIDE_NEUTRAL_OPEN
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

    @property
    def ticker(self) -> str:
        return PyUnicode_FromString(&self._data.ticker[0])

    @property
    def timestamp(self) -> float:
        return self._data.timestamp

    @property
    def dtype(self) -> int:
        return self._data.dtype

    @property
    def topic(self) -> str:
        cdef str ticker_str = PyUnicode_FromString(&self._data.ticker[0])
        return f'{ticker_str}.{self.__class__.__name__}'

    @property
    def market_time(self) :
        return _MarketDataVirtualBase.c_to_dt(self._data.timestamp)

    @property
    def price(self) -> float:
        return self._data.price

    @property
    def volume(self) -> float:
        return self._data.volume

    @property
    def side_int(self) -> int:
        return self._data.side

    @property
    def side_sign(self) -> Literal[-1, 0, 1]:
        return direction_to_sign(self._data.side)

    @property
    def side(self) -> TransactionSide:
        return TransactionSide(self.side_int)

    @property
    def multiplier(self) -> float:
        return self._data.multiplier

    @property
    def notional(self) -> float:
        return self._data.notional

    @property
    def transaction_id(self) -> object:
        return TransactionHelper.get_id(&self._data.transaction_id)

    @property
    def buy_id(self) -> object:
        return TransactionHelper.get_id(&self._data.buy_id)

    @property
    def sell_id(self) -> object:
        return TransactionHelper.get_id(&self._data.sell_id)

    @property
    def market_price(self) -> float:
        """
        Alias for the transaction price.
        """
        return self.price

    @property
    def volume_flow(self) -> float:
        cdef uint8_t sign = direction_to_sign(self._data.side)
        cdef double flow = self._data.volume * sign
        return flow

    @property
    def notional_flow(self) -> float:
        cdef uint8_t sign = direction_to_sign(self._data.side)
        cdef double notional = self._data.notional *  sign
        return notional


@cython.freelist(128)
cdef class OrderData:
    def __cinit__(self):
        self._data_ptr = <_MarketDataBuffer*> &self._data
        self._data_addr = <uintptr_t> self._data_ptr

    def __init__(self, *, str ticker, double timestamp, double price, double volume, uint8_t side, object order_id=None, uint8_t order_type=0, **kwargs):
        # Initialize base class fields
        cdef Py_ssize_t ticker_len
        cdef const char * ticker_ptr = PyUnicode_AsUTF8AndSize(ticker, &ticker_len)
        memcpy(<void *> &self._data.ticker, ticker_ptr, min(ticker_len, TICKER_SIZE - 1))
        self._data.timestamp = timestamp
        self._data.dtype = DataType.DTYPE_ORDER
        if kwargs: self.__dict__.update(kwargs)

        # Initialize order-specific fields
        self._data.price = price
        self._data.volume = volume
        self._data.side = side
        self._data.order_type = order_type

        # Initialize order_id
        TransactionHelper.set_id(id_ptr=&self._data.order_id, id_value=order_id)

    def __repr__(self) -> str:
        side_name = TransactionHelper.pyget_side_name(self._data.side)
        order_type_name = TransactionHelper.pyget_order_type_name(self._data.order_type)
        return f"<OrderData>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker}, price={self.price}, volume={self.volume}, side={side_name}, order_type={order_type_name})"

    def __reduce__(self):
        return self.__class__.from_bytes, (self.to_bytes(),), self.__dict__

    def __setstate__(self, state):
        if state:
            self.__dict__.update(state)

    def __copy__(self):
        cdef OrderData instance = OrderData.__new__(OrderData)
        memcpy(<void*> &instance._data, <const char*> &self._data, sizeof(_OrderDataBuffer))
        return instance

    @classmethod
    def buffer_size(cls):
        return sizeof(_OrderDataBuffer)

    cdef bytes c_to_bytes(self):
        return PyBytes_FromStringAndSize(<char*> &self._data, sizeof(self._data))

    @staticmethod
    cdef OrderData c_from_bytes(bytes data):
        cdef OrderData instance = OrderData.__new__(OrderData)
        memcpy(<void*> &instance._data, <const char*> data, sizeof(_OrderDataBuffer))
        return instance

    def to_bytes(self) -> bytes:
        return self.c_to_bytes()

    @classmethod
    def from_bytes(cls, bytes data):
        return OrderData.c_from_bytes(data)

    @property
    def ticker(self) -> str:
        return PyUnicode_FromString(&self._data.ticker[0])

    @property
    def timestamp(self) -> float:
        return self._data.timestamp

    @property
    def dtype(self) -> int:
        return self._data.dtype

    @property
    def topic(self) -> str:
        cdef str ticker_str = PyUnicode_FromString(&self._data.ticker[0])
        return f'{ticker_str}.{self.__class__.__name__}'

    @property
    def market_time(self) :
        return _MarketDataVirtualBase.c_to_dt(self._data.timestamp)

    @property
    def price(self) -> float:
        return self._data.price

    @property
    def volume(self) -> float:
        return self._data.volume

    @property
    def side_int(self) -> int:
        return self._data.side

    @property
    def side_sign(self) -> Literal[-1, 0, 1]:
        return direction_to_sign(self._data.side)

    @property
    def side(self) -> TransactionSide:
        return TransactionSide(self.side_int)

    @property
    def order_id(self) -> object:
        return TransactionHelper.get_id(&self._data.order_id)

    @property
    def order_type_int(self) -> int:
        return self._data.order_type

    @property
    def order_type(self) -> OrderType:
        return OrderType(self.order_type_int)

    @property
    def market_price(self) -> float:
        return self.price

    @property
    def flow(self) -> float:
        cdef int8_t sign = direction_to_sign(self._data.side)
        return sign * self._data.volume


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

    def __init__(self, *, ticker: str, double timestamp, double trade_price, double trade_volume, int side, object order_id=None, int order_type=0, **kwargs):
        TransactionData.__init__(self, ticker=ticker, timestamp=timestamp, price=trade_price, volume=trade_volume, side=side, order_id=order_id, order_type=order_type, **kwargs)

    @property
    def trade_price(self) -> float:
        return super().price

    @property
    def trade_volume(self) -> float:
        return super().volume
