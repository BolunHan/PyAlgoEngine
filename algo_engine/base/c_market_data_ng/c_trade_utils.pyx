import enum
import json
import time
import uuid

from cpython.datetime cimport datetime
from cpython.unicode cimport PyUnicode_FromString, PyUnicode_AsUTF8
from libc.math cimport NAN, fabs, isnan
from libc.stdint cimport int8_t, uintptr_t

from .c_market_data cimport (
    C_PROFILE,
    md_data_type, md_order_type, md_side,
    c_md_side_sign, c_md_state_name, c_md_side_name, c_md_order_type_name,
    c_md_state_working, c_md_state_placed, c_md_state_done,
    c_init_buffer, c_set_long_id, c_get_long_id, c_md_compare_long_id
)
from .c_transaction cimport TransactionData
from .c_transaction import TransactionSide, OrderType

from algo_engine.base import LOGGER


cdef object NO_DEFAULT = object()


class OrderState(enum.IntEnum):
    STATE_UNKNOWN       = md_order_state.STATE_UNKNOWN
    STATE_REJECTED      = md_order_state.STATE_REJECTED      # order rejected
    STATE_INVALID       = md_order_state.STATE_INVALID       # invalid order
    STATE_PENDING       = md_order_state.STATE_PENDING       # order not sent
    STATE_SENT          = md_order_state.STATE_SENT          # order sent (to exchange)
    STATE_PLACED        = md_order_state.STATE_PLACED        # order placed in exchange
    STATE_PARTFILLED    = md_order_state.STATE_PARTFILLED    # order partial filled
    STATE_FILLED        = md_order_state.STATE_FILLED        # order fully filled
    STATE_CANCELING     = md_order_state.STATE_CANCELING     # order canceling
    STATE_CANCELED      = md_order_state.STATE_CANCELED      # order stopped and canceled

    # Alias for compatibility
    UNKNOWN             = STATE_UNKNOWN
    Rejected            = STATE_REJECTED
    Invalid             = STATE_INVALID
    Pending             = STATE_PENDING
    Sent                = STATE_SENT
    Placed              = STATE_PLACED
    PartFilled          = STATE_PARTFILLED
    Filled              = STATE_FILLED
    Canceling           = STATE_CANCELING
    Canceled            = STATE_CANCELED

    def __hash__(self):
        return self.value

    @property
    def is_working(self):
        return <bint> c_md_state_working(self.value)

    @property
    def is_placed(self):
        return <bint> c_md_state_placed(self.value)

    @property
    def is_done(self):
        return <bint> c_md_state_done(self.value)

    @property
    def state_name(self):
        return PyUnicode_FromString(c_md_state_name(<md_order_state> self.value))


cdef class TradeReport(MarketData):
    def __init__(
            self,
            *,
            str ticker,
            double timestamp,
            double price,
            double volume,
            md_side side,
            double notional=NAN,
            double multiplier=1.,
            double fee=0.,
            object order_id=None,
            object trade_id=NO_DEFAULT,
            **kwargs
    ):
        if volume < 0:
            raise ValueError("Volume must be non-negative.")

        self.header = c_init_buffer(
            md_data_type.DTYPE_REPORT,
            PyUnicode_AsUTF8(ticker),
            timestamp
        )

        self.header.trade_report.price = price
        self.header.trade_report.volume = volume
        self.header.trade_report.side = side
        self.header.trade_report.multiplier = multiplier
        self.header.trade_report.fee = fee

        if isnan(notional):
            self.header.trade_report.notional = price * volume * multiplier
        else:
            self.header.trade_report.notional = notional

        c_set_long_id(&self.header.trade_report.order_id, order_id)
        c_set_long_id(&self.header.trade_report.trade_id, uuid.uuid4() if trade_id is NO_DEFAULT else trade_id)

        self.data_addr = <uintptr_t> self.header
        self.owner = True

        if kwargs:
            self.__dict__.update(kwargs)

    cdef dict c_to_json(self):
        cdef dict json_dict = {
            'ticker': self.ticker,
            'timestamp': self.timestamp,
            'price': self.price,
            'volume': self.volume,
            'side': self.side_int,
            'notional': self.notional,
            'multiplier': self.multiplier,
            'fee': self.fee,
            'order_id': self.order_id,
            'trade_id': self.trade_id
        }
        return json_dict

    @staticmethod
    cdef TradeReport c_from_json(dict json_dict):
        cdef TradeReport instance = TradeReport(
            ticker=json_dict['ticker'],
            timestamp=json_dict['timestamp'],
            price=json_dict['price'],
            volume=json_dict['volume'],
            side=json_dict['side'],
            notional=json_dict.get('notional', 0.),
            multiplier=json_dict.get('multiplier', 1.),
            fee=json_dict.get('fee', 0.),
            order_id=json_dict.get('order_id', None),
            trade_id=json_dict.get('trade_id', None)
        )
        return instance

    # === Python Interfaces ===

    def __eq__(self, TradeReport other):
        if not self.order_id == other.order_id:
            return False
        elif not self.trade_id == other.trade_id:
            return False

        return True

    def __repr__(self):
        if not self.header:
            return f"<{self.__class__.__name__}>(Uninitialized)"
        cdef str side_name = PyUnicode_FromString(c_md_side_name(self.header.trade_report.side))
        return f"<{self.__class__.__name__} id={self.trade_id}>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker} {side_name} {self.volume} at {self.price})"

    cpdef TradeReport reset_order_id(self, object order_id=NO_DEFAULT):
        if order_id is NO_DEFAULT:
            order_id = uuid.uuid4()
        c_set_long_id(&self.header.trade_report.order_id, order_id)
        return self

    cpdef TradeReport reset_trade_id(self, object trade_id=NO_DEFAULT):
        if trade_id is None:
            trade_id = uuid.uuid4()
        c_set_long_id(&self.header.trade_report.trade_id, trade_id)
        return self

    def to_json(self, str fmt='str', **kwargs):
        cdef dict json_dict = self.c_to_json()

        if fmt == 'dict':
            return json_dict
        elif fmt == 'str':
            return json.dumps(json_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, object json_data):
        if isinstance(json_data, dict):
            return TradeReport.c_from_json(json_data)
        cdef dict json_dict = json.loads(json_data)
        return TradeReport.c_from_json(json_data)

    cpdef TransactionData to_trade(self):
        return TransactionData(
            ticker=self.ticker,
            timestamp=self.timestamp,
            price=self.price,
            volume=self.volume,
            notional=self.notional,
            side=self.side_int,
            transaction_id=self.trade_id,
            multiplier=self.multiplier
        )

    property price:
        def __get__(self):
            return self.header.trade_report.price

    property volume:
        def __get__(self):
            return self.header.trade_report.volume

    property side:
        def __get__(self):
            return TransactionSide(self.header.trade_report.side)

    property side_int:
        def __get__(self):
            return self.header.trade_report.side

    property side_sign:
        def __get__(self):
            return c_md_side_sign(<md_side> self.header.trade_report.side)

    property multiplier:
        def __get__(self):
            return self.header.trade_report.multiplier

    property notional:
        def __get__(self):
            return self.header.trade_report.notional

    property fee:
        def __get__(self):
            return self.header.trade_report.fee

    property trade_id:
        def __get__(self):
            return c_get_long_id(&self.header.trade_report.trade_id)

    property order_id:
        def __get__(self):
            return c_get_long_id(&self.header.trade_report.order_id)

    property volume_flow:
        def __get__(self):
            cdef int8_t sign = c_md_side_sign(<md_side> self.header.trade_report.side)
            return sign * self.header.trade_report.volume

    property notional_flow:
        def __get__(self):
            cdef int8_t sign = c_md_side_sign(<md_side> self.header.trade_report.side)
            return sign * self.header.trade_report.notional

    property trade_time:
        def __get__(self):
            return datetime.fromtimestamp(self.header.meta_info.timestamp, tz=C_PROFILE.time_zone)


cdef class TradeInstruction(MarketData):
    def __init__(
            self,
            *,
            str ticker,
            double timestamp,
            md_side side,
            double volume,
            md_order_type order_type=md_order_type.ORDER_GENERIC,
            double limit_price=NAN,
            double multiplier=1.,
            object order_id=NO_DEFAULT,
            **kwargs
    ):
        self.header = c_init_buffer(
            md_data_type.DTYPE_INSTRUCTION,
            PyUnicode_AsUTF8(ticker),
            timestamp
        )

        if volume <= 0:
            raise ValueError("Volume must be positive")

        self.header.trade_instruction.limit_price = limit_price
        self.header.trade_instruction.volume = volume
        self.header.trade_instruction.side = side
        self.header.trade_instruction.order_type = order_type
        self.header.trade_instruction.multiplier = multiplier

        c_set_long_id(&self.header.trade_instruction.order_id, uuid.uuid4() if order_id is NO_DEFAULT else order_id)

        self.header.trade_instruction.order_state = md_order_state.STATE_PENDING
        self.header.trade_instruction.filled_volume = 0.
        self.header.trade_instruction.filled_notional = 0.
        self.header.trade_instruction.fee = 0.
        self.header.trade_instruction.ts_placed = 0.
        self.header.trade_instruction.ts_canceled = 0.
        self.header.trade_instruction.ts_finished = 0.
        self.trades = {}

        self.data_addr = <uintptr_t> self.header
        self.owner = True

        if kwargs:
            self.__dict__.update(kwargs)

    cdef dict c_to_json(self):
        cdef dict json_dict = {
            'ticker': self.ticker,
            'timestamp': self.timestamp,
            'side': self.side_int,
            'volume': self.volume,
            'order_type': self.order_type_int,
            'limit_price': self.limit_price,
            'multiplier': self.multiplier,
            'order_id': self.order_id,
        }
        return json_dict

    @staticmethod
    cdef TradeInstruction c_from_json(dict json_dict):
        cdef TradeInstruction instance = TradeInstruction(
            ticker=json_dict['ticker'],
            timestamp=json_dict['timestamp'],
            side=json_dict['side'],
            volume=json_dict['volume'],
            order_type=json_dict.get('order_type', 0),
            limit_price=json_dict.get('limit_price', NAN),
            multiplier=json_dict.get('multiplier', 1.),
            order_id=json_dict['order_id']
        )
        return instance

    # === Python Interfaces ===

    def __eq__(self, TradeInstruction other):
        return self.order_id == other.order_id

    def __repr__(self):
        if not self.header:
            return f"<{self.__class__.__name__}>(Uninitialized)"
        cdef str side_name = PyUnicode_FromString(c_md_side_name(self.header.trade_instruction.side))
        cdef str order_type_name = PyUnicode_FromString(c_md_order_type_name(self.header.trade_instruction.order_type))
        if isnan(self.header.trade_instruction.limit_price) or self.header.trade_instruction.order_type == md_order_type.ORDER_MARKET:
            return f'<{self.__class__.__name__} id={self.order_id}>({self.ticker} {order_type_name} {side_name} {self.volume}; filled {self.filled_volume:.2f} @ {self.average_price:.2f} now {self.order_state.state_name})'
        else:
            return f'<{self.__class__.__name__} id={self.order_id}>({self.ticker} {order_type_name} {side_name} {self.volume} limit {self.limit_price:.2f}; filled {self.filled_volume:.2f} @ {self.average_price:.2f} now {self.order_state.state_name})'

    def to_json(self, str fmt='str', **kwargs):
        cdef dict json_dict = self.c_to_json()
        if fmt == 'dict':
            return json_dict
        elif fmt == 'str':
            return json.dumps(json_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, object json_data):
        if isinstance(json_data, dict):
            return TradeInstruction.c_from_json(json_data)
        cdef dict json_dict = json.loads(json_data)
        return TradeInstruction.c_from_json(json_dict)

    cpdef TradeInstruction reset(self):
        self.trades.clear()

        self.header.trade_instruction.order_state = md_order_state.STATE_PENDING
        self.header.trade_instruction.filled_volume = 0.
        self.header.trade_instruction.filled_notional = 0.
        self.header.trade_instruction.fee = 0.
        self.header.trade_instruction.ts_placed = 0.
        self.header.trade_instruction.ts_canceled = 0.
        self.header.trade_instruction.ts_finished = 0.

        return self

    cpdef TradeInstruction reset_order_id(self, object order_id=NO_DEFAULT):
        if order_id is NO_DEFAULT:
            order_id = uuid.uuid4()

        cdef TradeReport trade_report
        for trade_report in self.trades.values():
            c_set_long_id(&trade_report.header.trade_report.order_id, order_id)
        c_set_long_id(&self.header.trade_instruction.order_id, order_id)
        return self

    cpdef TradeInstruction set_order_state(self, md_order_state order_state, double timestamp=NAN):
        if isnan(timestamp):
            timestamp = time.time()

        self.header.trade_instruction.order_state = order_state

        if order_state == md_order_state.STATE_PLACED:
            self.header.trade_instruction.ts_placed = timestamp
        elif order_state == md_order_state.STATE_FILLED:
            self.header.trade_instruction.ts_finished = timestamp
        elif order_state == md_order_state.STATE_CANCELED:
            self.header.trade_instruction.ts_canceled = timestamp

        return self

    cpdef TradeInstruction fill(self, TradeReport trade_report):
        # Check order_id match by comparing Python objects
        if not c_md_compare_long_id(&self.header.trade_instruction.order_id, &trade_report.header.trade_report.order_id):
            LOGGER.warning(f'Order ID mismatch! Instruction: {self.order_id}, Report: {trade_report.order_id}')
            return self

        # Check for duplicate trade
        cdef object trade_id = c_get_long_id(&trade_report.header.trade_report.trade_id)
        if trade_id in self.trades:
            LOGGER.warning(f'Duplicated trade received!\nInstruction {self}.\nReport {trade_report}.')
            return self

        # Check multiplier consistency
        cdef double multiplier = trade_report.header.trade_report.multiplier
        if isnan(self.header.trade_instruction.multiplier):
            self.header.trade_instruction.multiplier = multiplier
        elif self.header.trade_instruction.multiplier != multiplier:
            raise ValueError(f'Multiplier not match for order {self} and report {trade_report}.')

        # Check volume doesn't exceed order volume
        cdef double trade_volume = trade_report.header.trade_report.volume
        cdef double trade_notional = trade_report.header.trade_report.notional
        cdef double trade_fee = trade_report.header.trade_report.fee
        cdef double timestamp = trade_report.header.meta_info.timestamp

        if trade_volume + self.header.trade_instruction.filled_volume > self.header.trade_instruction.volume:
            trades_str = '\n\t'.join([str(x) for x in self.trades.values()]) + f'\n\t<new> {trade_report}'
            LOGGER.warning('Fatal error!\nTradeInstruction: \n\t{}\nTradeReport:\n\t{}'.format(str(self), trades_str))
            raise ValueError('Fatal error! trade reports filled volume exceed order volume!')

        # Only process if there's volume
        self.header.trade_instruction.filled_volume += trade_volume
        self.header.trade_instruction.filled_notional += fabs(trade_notional)
        self.header.trade_instruction.fee += trade_fee

        # Update order state
        if self.header.trade_instruction.filled_volume == self.volume:
            self.set_order_state(md_order_state.STATE_FILLED, timestamp)
            self.header.trade_instruction.ts_finished = timestamp
        elif self.header.trade_instruction.filled_volume > 0:
            self.set_order_state(md_order_state.STATE_PARTFILLED)

        # Add to trades dictionary
        self.trades[trade_id] = trade_report
        return self

    cpdef TradeInstruction add_trade(self, TradeReport trade_report):
        cdef object trade_id = c_get_long_id(&trade_report.header.trade_report.trade_id)
        cdef double trade_volume = trade_report.header.trade_report.volume
        cdef double trade_notional = trade_report.header.trade_report.notional
        cdef double trade_fee = trade_report.header.trade_report.fee
        cdef double timestamp = trade_report.header.meta_info.timestamp

        self.header.trade_instruction.filled_volume += trade_volume
        self.header.trade_instruction.filled_notional += fabs(trade_notional)
        self.header.trade_instruction.fee += trade_fee

        if self.filled_volume == self.volume:
            self.set_order_state(md_order_state.STATE_FILLED, timestamp)
        elif self.filled_volume > 0:
            self.set_order_state(md_order_state.STATE_PARTFILLED)

        self.trades[trade_id] = trade_report
        return self

    cpdef TradeInstruction cancel_order(self, double timestamp=NAN):
        self.set_order_state(md_order_state.STATE_CANCELING, timestamp)
        return self

    cpdef TradeInstruction canceled(self, double timestamp=NAN):
        self.set_order_state(md_order_state.STATE_CANCELED, timestamp)
        return self

    property is_working:
        def __get__(self):
            return <bint> c_md_state_working(self.header.trade_instruction.order_state)

    property is_placed:
        def __get__(self):
            return <bint> c_md_state_placed(self.header.trade_instruction.order_state)

    property is_done:
        def __get__(self):
            return <bint> c_md_state_done(self.header.trade_instruction.order_state)

    property limit_price:
        def __get__(self):
            return self.header.trade_instruction.limit_price

    property volume:
        def __get__(self):
            return self.header.trade_instruction.volume

    property side:
        def __get__(self):
            return TransactionSide(self.header.trade_instruction.side)

    property side_int:
        def __get__(self):
            return self.header.trade_instruction.side

    property side_sign:
        def __get__(self):
            return c_md_side_sign(<md_side> self.header.trade_instruction.side)

    property order_type:
        def __get__(self):
            return OrderType(self.header.trade_instruction.order_type)

    property order_type_int:
        def __get__(self):
            return self.header.trade_instruction.order_type

    property order_state:
        def __get__(self):
            return OrderState(self.header.trade_instruction.order_state)

    property order_state_int:
        def __get__(self):
            return self.header.trade_instruction.order_state

    property multiplier:
        def __get__(self):
            return self.header.trade_instruction.multiplier

    property filled_volume:
        def __get__(self):
            return self.header.trade_instruction.filled_volume

    property working_volume:
        def __get__(self):
            cdef double total_volume = self.header.trade_instruction.volume
            cdef double filled_volume = self.header.trade_instruction.filled_volume
            return total_volume - filled_volume

    property filled_notional:
        def __get__(self):
            return self.header.trade_instruction.filled_notional

    property fee:
        def __get__(self):
            return self.header.trade_instruction.fee

    property order_id:
        def __get__(self):
            return c_get_long_id(&self.header.trade_instruction.order_id)

    property average_price:
        def __get__(self):
            cdef double filled_volume = self.header.trade_instruction.volume
            cdef double filled_notional = self.header.trade_instruction.filled_notional
            cdef double multiplier = self.header.trade_instruction.multiplier
            if filled_volume and multiplier:
                return filled_notional / filled_volume / multiplier
            return NAN

    property start_time:
        def __get__(self):
            return datetime.fromtimestamp(self.header.meta_info.timestamp, tz=C_PROFILE.time_zone)

    property placed_ts:
        def __get__(self):
            return self.header.trade_instruction.ts_placed

    property canceled_ts:
        def __get__(self):
            return self.header.trade_instruction.ts_canceled

    property finished_ts:
        def __get__(self):
            return self.header.trade_instruction.ts_finished

    property placed_time:
        def __get__(self):
            cdef double ts_placed = self.header.trade_instruction.ts_placed
            if ts_placed:
                return datetime.fromtimestamp(ts_placed, tz=C_PROFILE.time_zone)
            return None

    property canceled_time:
        def __get__(self):
            cdef double ts_canceled = self.header.trade_instruction.ts_canceled
            if ts_canceled:
                return datetime.fromtimestamp(ts_canceled, tz=C_PROFILE.time_zone)
            return None

    property finished_time:
        def __get__(self):
            cdef double ts_finished = self.header.trade_instruction.ts_finished
            if ts_finished:
                return datetime.fromtimestamp(ts_finished, tz=C_PROFILE.time_zone)
            return None


from . cimport c_market_data

c_market_data.report_from_header = report_from_header
c_market_data.instruction_from_header = instruction_from_header
