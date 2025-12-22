import enum
import warnings

from cpython.list cimport PyList_Size, PyList_GET_ITEM
from cpython.unicode cimport PyUnicode_FromString, PyUnicode_AsUTF8
from libc.math cimport INFINITY, NAN, isnan, copysign, fabs
from libc.stdint cimport int8_t, uintptr_t

from .c_market_data cimport (
    data_type_t, order_type_t, direction_t, offset_t, side_t,
    transaction_data_t, c_md_side_opposite, c_md_side_sign, c_md_side_offset, c_md_side_direction,
    c_md_side_name, c_md_offset_name, c_md_direction_name, c_md_order_type_name,
    c_init_buffer, c_set_id, c_get_id
)


class OrderType(enum.IntEnum):
    ORDER_UNKNOWN = order_type_t.ORDER_UNKNOWN
    ORDER_CANCEL = order_type_t.ORDER_CANCEL
    ORDER_GENERIC = order_type_t.ORDER_GENERIC
    ORDER_LIMIT = order_type_t.ORDER_LIMIT
    ORDER_LIMIT_MAKER = order_type_t.ORDER_LIMIT_MAKER
    ORDER_MARKET = order_type_t.ORDER_MARKET
    ORDER_FOK = order_type_t.ORDER_FOK
    ORDER_FAK = order_type_t.ORDER_FAK
    ORDER_IOC = order_type_t.ORDER_IOC


class TransactionDirection(enum.IntEnum):
    DIRECTION_UNKNOWN = direction_t.DIRECTION_UNKNOWN
    DIRECTION_SHORT = direction_t.DIRECTION_SHORT
    DIRECTION_LONG = direction_t.DIRECTION_LONG
    DIRECTION_NEUTRAL = direction_t.DIRECTION_NEUTRAL

    def __or__(self, offset):
        if not isinstance(offset, TransactionOffset):
            raise TypeError(f'{self.__class__.__name__} Can only merge with a TransactionOffset.')

        combined_value = self.value + offset.value
        side = TransactionSide(combined_value)

        if side is TransactionSide.FAULTY:
            raise ValueError(f"Combination of {self.name} and {offset.name} doesn't correspond to a valid TransactionSide")

        return side

    @property
    def sign(self):
        return c_md_side_sign(self.value)


class TransactionOffset(enum.IntEnum):
    OFFSET_CANCEL = offset_t.OFFSET_CANCEL
    OFFSET_ORDER = offset_t.OFFSET_ORDER
    OFFSET_OPEN = offset_t.OFFSET_OPEN
    OFFSET_CLOSE = offset_t.OFFSET_CLOSE

    def __or__(self, direction):
        if not isinstance(direction, TransactionDirection):
            raise TypeError(f'{self.__class__.__name__} Can only merge with a TransactionDirection.')

        combined_value = self.value + direction.value
        side = TransactionSide(combined_value)

        if side is TransactionSide.FAULTY:
            raise ValueError(f"Combination of {self.name} and {direction.name} doesn't correspond to a valid TransactionSide")

        return side


class TransactionSide(enum.IntEnum):
    # Long Side
    SIDE_LONG_OPEN = side_t.SIDE_LONG_OPEN
    SIDE_LONG_CLOSE = side_t.SIDE_LONG_CLOSE
    SIDE_LONG_CANCEL = side_t.SIDE_LONG_CANCEL

    # Short Side
    SIDE_SHORT_OPEN = side_t.SIDE_SHORT_OPEN
    SIDE_SHORT_CLOSE = side_t.SIDE_SHORT_CLOSE
    SIDE_SHORT_CANCEL = side_t.SIDE_SHORT_CANCEL

    # Neutral Side
    SIDE_NEUTRAL_OPEN = side_t.SIDE_NEUTRAL_OPEN
    SIDE_NEUTRAL_CLOSE = side_t.SIDE_NEUTRAL_CLOSE

    # Order
    SIDE_BID = side_t.SIDE_BID
    SIDE_ASK = side_t.SIDE_ASK

    # Generic Cancel
    SIDE_CANCEL = side_t.SIDE_CANCEL

    # C Alias
    SIDE_UNKNOWN = side_t.SIDE_CANCEL
    SIDE_LONG = side_t.SIDE_LONG_OPEN
    SIDE_SHORT = side_t.SIDE_SHORT_OPEN

    # Backward compatibility
    ShortOrder = AskOrder = Ask = SIDE_ASK
    LongOrder = BidOrder = Bid = SIDE_BID

    ShortFilled = Unwind = Sell = SIDE_SHORT_CLOSE
    LongFilled = LongOpen = Buy = SIDE_LONG_OPEN

    ShortOpen = Short = SIDE_SHORT_OPEN
    Cover = SIDE_LONG_CLOSE

    UNKNOWN = CANCEL = SIDE_CANCEL
    FAULTY = 255

    def __neg__(self):
        return self.__class__(c_md_side_opposite(<side_t> self.value))

    @classmethod
    def _missing_(cls, value):
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
                warnings.warn(f'{value} is not recognized, return TransactionSide.FAULTY', RuntimeWarning, stacklevel=2)

        return trade_side

    @property
    def sign(self):
        return c_md_side_sign(<side_t> self.value)

    @property
    def offset(self):
        return TransactionOffset(c_md_side_offset(<side_t> self.value))

    @property
    def direction(self):
        return TransactionDirection(c_md_side_direction(<side_t> self.value))

    @property
    def side_name(self):
        return PyUnicode_FromString(c_md_side_name(<side_t> self.value))

    @property
    def offset_name(self):
        return PyUnicode_FromString(c_md_offset_name(<side_t> self.value))

    @property
    def direction_name(self):
        return PyUnicode_FromString(c_md_direction_name(<side_t> self.value))


cdef class TransactionData(MarketData):
    def __init__(
            self,
            *,
            ticker: str,
            double timestamp,
            double price,
            double volume,
            side_t side,
            double multiplier=1.0,
            double notional=NAN,
            object transaction_id=None,
            object buy_id=None,
            object sell_id=None,
            **kwargs
    ):
        self.header = c_init_buffer(
            data_type_t.DTYPE_TRANSACTION,
            PyUnicode_AsUTF8(ticker),
            timestamp
        )

        self.header.transaction_data.price = price
        self.header.transaction_data.volume = volume
        self.header.transaction_data.side = side
        self.header.transaction_data.multiplier = multiplier

        if isnan(notional):
            self.header.transaction_data.notional = price * volume * multiplier
        else:
            self.header.transaction_data.notional = notional

        c_set_id(&self.header.transaction_data.transaction_id, transaction_id)
        c_set_id(&self.header.transaction_data.buy_id, buy_id)
        c_set_id(&self.header.transaction_data.sell_id, sell_id)

        self.data_addr = <uintptr_t> self.header
        self.owner = True

        if kwargs:
            self.__dict__.update(kwargs)

    def __repr__(self):
        if not self.header:
            return f"<{self.__class__.__name__}>(Uninitialized)"
        cdef str side_name = PyUnicode_FromString(c_md_side_name(self.header.transaction_data.side))
        return f"<{self.__class__.__name__}>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker}, price={self.price}, volume={self.volume}, side={side_name})"

    @classmethod
    def merge(cls, list data_list):
        cdef Py_ssize_t n = PyList_Size(data_list)
        if n == 0:
            raise ValueError('Data list empty')

        cdef TransactionData first = <TransactionData> PyList_GET_ITEM(data_list, 0)
        cdef str ticker = first.ticker
        cdef double multiplier = first.multiplier
        cdef double timestamp = first.timestamp
        cdef double sum_volume = first.volume_flow
        cdef double sum_notional = first.notional_flow
        cdef Py_ssize_t i
        cdef TransactionData md
        cdef transaction_data_t td
        cdef int8_t sign

        for i in range(1, n):
            md = <TransactionData> PyList_GET_ITEM(data_list, i)
            td = md.header.transaction_data
            sign = c_md_side_sign(td.side)

            # Validation
            if md.ticker != ticker:
                raise AssertionError(f'Ticker mismatch, expect {ticker}, got {md.ticker}')

            if td.multiplier != multiplier:
                raise AssertionError(f'Multiplier mismatch, expect {multiplier}, got {td.multiplier}')

            # Calculations
            sum_volume += td.volume * sign
            sum_notional += td.notional * sign
            if td.meta_info.timestamp > timestamp:
                timestamp = td.meta_info.timestamp

        # Determine trade parameters using copysign and fabs
        cdef side_t trade_side = side_t.SIDE_LONG if sum_volume > 0 else side_t.SIDE_SHORT if sum_volume < 0 else side_t.SIDE_NEUTRAL_OPEN
        cdef double trade_volume = fabs(sum_volume)
        cdef double trade_notional = fabs(sum_notional)
        cdef double trade_price

        # Price calculation with copysign for infinity cases
        if sum_notional == 0.0:
            trade_price = 0.0
        elif sum_volume == 0.0:
            trade_price = copysign(INFINITY, sum_notional)
        else:
            trade_price = sum_notional / sum_volume

        return cls(
            ticker=ticker,
            timestamp=timestamp,
            price=trade_price,
            volume=trade_volume,
            side=trade_side,
            multiplier=multiplier,
            notional=trade_notional
        )

    property price:
        def __get__(self):
            return self.header.transaction_data.price

    property volume:
        def __get__(self):
            return self.header.transaction_data.volume

    property side:
        def __get__(self):
            return TransactionSide(self.header.transaction_data.side)

    property side_int:
        def __get__(self):
            return self.header.transaction_data.side

    property side_sign:
        def __get__(self):
            return c_md_side_sign(self.header.transaction_data.side)

    property multiplier:
        def __get__(self):
            return self.header.transaction_data.multiplier

    property notional:
        def __get__(self):
            return self.header.transaction_data.notional

    property transaction_id:
        def __get__(self):
            return c_get_id(&self.header.transaction_data.transaction_id)

    property buy_id:
        def __get__(self):
            return c_get_id(&self.header.transaction_data.buy_id)

    property sell_id:
        def __get__(self):
            return c_get_id(&self.header.transaction_data.sell_id)

    property volume_flow:
        def __get__(self):
            cdef int8_t sign = c_md_side_sign(self.header.transaction_data.side)
            return self.header.transaction_data.volume * sign

    property notional_flow:
        def __get__(self):
            cdef int8_t sign = c_md_side_sign(self.header.transaction_data.side)
            return self.header.transaction_data.notional * sign


cdef class OrderData(MarketData):
    def __init__(
            self,
            *,
            str ticker,
            double timestamp,
            double price,
            double volume,
            side_t side,
            object order_id=None,
            order_type_t order_type=order_type_t.ORDER_GENERIC,
            **kwargs
    ):
        self.header = c_init_buffer(
            data_type_t.DTYPE_ORDER,
            PyUnicode_AsUTF8(ticker),
            timestamp
        )

        self.header.order_data.price = price
        self.header.order_data.volume = volume
        self.header.order_data.side = side
        self.header.order_data.order_type = order_type

        c_set_id(id_ptr=&self.header.order_data.order_id, id_value=order_id)

        self.data_addr = <uintptr_t> self.header
        self.owner = True

        if kwargs:
            self.__dict__.update(kwargs)

    def __repr__(self):
        if not self.header:
            return f"<{self.__class__.__name__}>(Uninitialized)"
        cdef str side_name = PyUnicode_FromString(c_md_side_name(self.header.order_data.side))
        cdef str order_type_name = PyUnicode_FromString(c_md_order_type_name(self.header.order_data.order_type))
        return f"<{self.__class__.__name__}>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker}, price={self.price}, volume={self.volume}, side={side_name}, order_type={order_type_name})"

    property price:
        def __get__(self):
            return self.header.order_data.price

    property volume:
        def __get__(self):
            return self.header.order_data.volume

    property side:
        def __get__(self):
            return TransactionSide(self.side_int)

    property side_int:
        def __get__(self):
            return self.header.order_data.side

    property side_sign:
        def __get__(self):
            return c_md_side_sign(self.header.order_data.side)

    property order_id:
        def __get__(self):
            return c_get_id(&self.header.order_data.order_id)

    property order_type:
        def __get__(self):
            return OrderType(self.header.order_data.order_type)

    property order_type_int:
        def __get__(self):
            return self.header.order_data.order_type

    property flow:
        def __get__(self):
            cdef int8_t sign = c_md_side_sign(self.header.order_data.side)
            return self.header.order_data.volume * sign


cdef class TradeData(TransactionData):
    def __init__(
            self,
            *,
            ticker: str,
            double timestamp,
            double trade_price,
            double trade_volume,
            side_t trade_side,
            double multiplier=1.0,
            double notional=NAN,
            object transaction_id=None,
            object buy_id=None,
            object sell_id=None,
            **kwargs
    ):
        self.header = c_init_buffer(
            data_type_t.DTYPE_TRANSACTION,
            PyUnicode_AsUTF8(ticker),
            timestamp
        )

        self.header.transaction_data.price = trade_price
        self.header.transaction_data.volume = trade_volume
        self.header.transaction_data.side = trade_side
        self.header.transaction_data.multiplier = multiplier

        if isnan(notional):
            self.header.transaction_data.notional = trade_price * trade_volume * multiplier
        else:
            self.header.transaction_data.notional = notional

        c_set_id(&self.header.transaction_data.transaction_id, transaction_id)
        c_set_id(&self.header.transaction_data.buy_id, buy_id)
        c_set_id(&self.header.transaction_data.sell_id, sell_id)

        self.data_addr = <uintptr_t> self.header
        self.owner = True

        if kwargs:
            self.__dict__.update(kwargs)

    property trade_price:
        def __get__(self):
            return self.header.transaction_data.price

    property trade_volume:
        def __get__(self):
            return self.header.transaction_data.volume

    property trade_side:
        def __get__(self):
            return TransactionSide(self.header.transaction_data.side)


from . cimport c_market_data

c_market_data.transaction_from_header = transaction_from_header
c_market_data.order_from_header = order_from_header
