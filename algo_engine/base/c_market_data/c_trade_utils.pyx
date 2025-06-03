# cython: language_level=3
import enum
import time
import uuid
from typing import Literal

cimport cython
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.datetime cimport datetime
from libc.math cimport NAN, fabs, isnan
from libc.stdint cimport uint8_t
from libc.string cimport memcpy

from .c_market_data cimport _MarketDataVirtualBase, TICKER_SIZE, _MarketDataBuffer, DataType, OrderType, OrderState as OrderStateCython, _TradeReportBuffer, _TradeInstructionBuffer
from .c_transaction cimport TransactionData, TransactionHelper
from .c_transaction import TransactionSide

from algo_engine.base import LOGGER


# Python wrapper for Direction enum
class OrderState(enum.IntEnum):
    STATE_UNKNOWN = OrderStateCython.STATE_UNKNOWN
    STATE_REJECTED = OrderStateCython.STATE_REJECTED     # order rejected
    STATE_INVALID = OrderStateCython.STATE_INVALID       # invalid order
    STATE_PENDING = OrderStateCython.STATE_PENDING       # order not sent
    STATE_SENT = OrderStateCython.STATE_SENT             # order sent (to exchange)
    STATE_PLACED = OrderStateCython.STATE_PLACED         # order placed in exchange
    STATE_PARTFILLED = OrderStateCython.STATE_PARTFILLED # order partial filled
    STATE_FILLED = OrderStateCython.STATE_FILLED         # order fully filled
    STATE_CANCELING = OrderStateCython.STATE_CANCELING   # order canceling
    STATE_CANCELED = OrderStateCython.STATE_CANCELED     # order stopped and canceled

    # Alias for compatibility
    UNKNOWN = STATE_UNKNOWN
    Rejected = STATE_REJECTED
    Invalid = STATE_INVALID
    Pending = STATE_PENDING
    Sent = STATE_SENT
    Placed = STATE_PLACED
    PartFilled = STATE_PARTFILLED
    Filled = STATE_FILLED
    Canceling = STATE_CANCELING
    Canceled = STATE_CANCELED

    def __hash__(self):
        return self.value

    @property
    def is_working(self):
        return OrderStateHelper.is_working(self.value)

    @property
    def is_done(self):
        return OrderStateHelper.is_done(self.value)

    @property
    def state_name(self) -> str:
        if self.value == OrderStateCython.STATE_UNKNOWN:
            return 'unknown'
        elif self.value == OrderStateCython.STATE_REJECTED:
            return 'rejected'
        elif self.value == OrderStateCython.STATE_INVALID:
            return 'invalid'
        elif self.value == OrderStateCython.STATE_PENDING:
            return 'pending'
        elif self.value == OrderStateCython.STATE_SENT:
            return 'sent'
        elif self.value == OrderStateCython.STATE_PLACED:
            return 'placed'
        elif self.value == OrderStateCython.STATE_PARTFILLED:
            return 'part-filled'
        elif self.value == OrderStateCython.STATE_FILLED:
            return 'filled'
        elif self.value == OrderStateCython.STATE_CANCELING:
            return 'canceling'
        elif self.value == OrderStateCython.STATE_CANCELED:
            return 'canceled'


cdef class OrderStateHelper:
    @staticmethod
    cdef bint is_working(int order_state):
        if order_state == OrderStateCython.STATE_SENT or order_state == OrderStateCython.STATE_PLACED or order_state == OrderStateCython.STATE_PARTFILLED or order_state == OrderStateCython.STATE_CANCELING:
            return True

        return False

    @staticmethod
    cdef bint is_done(int order_state):
        if order_state == OrderStateCython.STATE_FILLED or order_state == OrderStateCython.STATE_CANCELED or order_state == OrderStateCython.STATE_REJECTED or order_state == OrderStateCython.STATE_INVALID:
            return True

        return False


@cython.freelist(128)
cdef class TradeReport:
    def __cinit__(self):
        self._data_ptr = <_MarketDataBuffer*> &self._data

    def __init__(self, *, ticker: str, double timestamp, double price, double volume, uint8_t side, double notional=0., double multiplier=1., double fee=0., object order_id=None, object trade_id=None, **kwargs):
        if volume < 0:
            raise ValueError("Volume must be non-negative.")

        # Initialize base class fields
        cdef bytes ticker_bytes = ticker.encode('utf-8')
        cdef size_t ticker_len = min(len(ticker_bytes), TICKER_SIZE - 1)
        memcpy(<void *> &self._data.ticker, <const char *> ticker_bytes, ticker_len)
        self._data.timestamp = timestamp
        self._data.dtype = DataType.DTYPE_REPORT
        if kwargs: self.__dict__.update(kwargs)
        
        # Initialize report-specific fields
        self._data.price = price
        self._data.volume = volume
        self._data.side = side
        self._data.multiplier = multiplier
        self._data.fee = fee

        # Calculate notional if not provided
        if notional == 0.0:
            self._data.notional = price * volume * multiplier
        else:
            self._data.notional = notional

        # Initialize IDs
        if trade_id is None:
            TransactionHelper.set_id(id_ptr=&self._data.trade_id, id_value=uuid.uuid4())
        else:
            TransactionHelper.set_id(id_ptr=&self._data.trade_id, id_value=trade_id)
        if order_id is None:
            raise ValueError('Must assign an order_id.')
        else:
            TransactionHelper.set_id(id_ptr=&self._data.order_id, id_value=order_id)

    def __eq__(self, other: TradeReport):
        assert isinstance(other, self.__class__), f'Can only compare with {self.__class__.__name__}'

        # Fast check: only check the order id and trade id.
        if not self.order_id == other.order_id:
            return False
        elif not self.trade_id == other.trade_id:
            return False

        return True

    def __repr__(self) -> str:
        side_name = TransactionHelper.get_side_name(self._data.side).decode('utf-8')

        return f"<TradeReport id={self.trade_id}>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker} {side_name} {self.volume} at {self.price})"

    def __reduce__(self):
        return self.__class__.from_bytes, (self.to_bytes(),), self.__dict__

    def __setstate__(self, state):
        if state:
            self.__dict__.update(state)

    def __copy__(self):
        cdef TradeReport instance = TradeReport.__new__(TradeReport)
        memcpy(<void*> &instance._data, <const char*> &self._data, sizeof(_TradeReportBuffer))
        return instance

    cpdef TradeReport reset_order_id(self, object order_id=None):
        if order_id is None:
            order_id = uuid.uuid4()
        TransactionHelper.set_id(id_ptr=&self._data.order_id, id_value=order_id)
        return self

    cpdef TradeReport reset_trade_id(self, object trade_id=None):
        if trade_id is None:
            trade_id = uuid.uuid4()
        TransactionHelper.set_id(id_ptr=&self._data.trade_id, id_value=trade_id)
        return self

    @classmethod
    def buffer_size(cls):
        return sizeof(_TradeReportBuffer)

    cdef bytes c_to_bytes(self):
        return PyBytes_FromStringAndSize(<char*> &self._data, sizeof(self._data))

    @staticmethod
    cdef TradeReport c_from_bytes(bytes data):
        cdef TradeReport instance = TradeReport.__new__(TradeReport)
        memcpy(<void*> &instance._data, <const char*> data, sizeof(_TradeReportBuffer))
        return instance

    def to_bytes(self) -> bytes:
        return self.c_to_bytes()

    @classmethod
    def from_bytes(cls, bytes data):
        return TradeReport.c_from_bytes(data)

    def copy(self):
        return self.__copy__()

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
        return TransactionHelper.get_sign(self._data.side)

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
    def fee(self) -> float:
        return self._data.fee

    @property
    def trade_id(self) -> object:
        return TransactionHelper.get_id(&self._data.trade_id)

    @property
    def order_id(self) -> object:
        return TransactionHelper.get_id(&self._data.order_id)

    @property
    def market_price(self) -> float:
        """
        Alias for the transaction price.
        """
        return self.price

    @property
    def volume_flow(self) -> float:
        cdef int sign = TransactionHelper.get_sign(self._data.side)
        return sign * self._data.volume

    @property
    def notional_flow(self) -> float:
        cdef int sign = TransactionHelper.get_sign(self._data.side)
        return sign * self._data.notional

    @property
    def trade_time(self) -> datetime:
        return _MarketDataVirtualBase.c_to_dt(self._data.timestamp)


@cython.freelist(128)
cdef class TradeInstruction:
    def __cinit__(self):
        self._data_ptr = <_MarketDataBuffer*> &self._data
        self.trades = {}

    def __init__(self, *, str ticker, double timestamp, uint8_t side, double volume, uint8_t order_type=0, double limit_price=NAN, double multiplier=1., object order_id=None, **kwargs):
        if not (volume > 0):
            raise ValueError("Volume must be positive")

        # Initialize base class fields
        cdef bytes ticker_bytes = ticker.encode('utf-8')
        cdef size_t ticker_len = min(len(ticker_bytes), TICKER_SIZE - 1)
        memcpy(<void*> &self._data.ticker, <const char*> ticker_bytes, ticker_len)
        self._data.timestamp = timestamp
        self._data.dtype = DataType.DTYPE_INSTRUCTION
        if kwargs: self.__dict__.update(kwargs)
        
        # Initialize instruction-specific fields
        self._data.limit_price = limit_price
        self._data.volume = volume
        self._data.side = side
        self._data.order_type = order_type
        self._data.multiplier = multiplier

        # Initialize IDs
        if order_id is None:
            TransactionHelper.set_id(id_ptr=&self._data.order_id, id_value=uuid.uuid4())
        else:
            TransactionHelper.set_id(id_ptr=&self._data.order_id, id_value=order_id)

        self._data.order_state = OrderStateCython.STATE_PENDING
        self._data.filled_volume = 0.
        self._data.filled_notional = 0.
        self._data.fee = 0.
        self._data.ts_placed = 0.
        self._data.ts_canceled = 0.
        self._data.ts_finished = 0.

    def __eq__(self, other: TradeInstruction):
        assert isinstance(other, self.__class__), f'Can only compare with {self.__class__.__name__}'

        # Fast check: only check the order id
        if not self.order_id == other.order_id:
            return False

        return True

    def __repr__(self) -> str:
        side_name = TransactionHelper.get_side_name(self._data.side).decode('utf-8')
        order_type_name = TransactionHelper.get_order_type_name(self._data.order_type).decode('utf-8')

        if self.limit_price is None or self.order_type_int == OrderType.ORDER_MARKET:
            return f'<TradeInstruction id={self.order_id}>({self.ticker} {order_type_name} {side_name} {self.volume}; filled {self.filled_volume:.2f} @ {self.average_price:.2f} now {self.order_state.state_name})'
        else:
            return f'<TradeInstruction id={self.order_id}>({self.ticker} {order_type_name} {side_name} {self.volume} limit {self.limit_price:.2f}; filled {self.filled_volume:.2f} @ {self.average_price:.2f} now {self.order_state.state_name})'

    def __reduce__(self):
        return self.__class__.from_bytes, (self.to_bytes(),), self.__dict__

    def __setstate__(self, state):
        if state:
            self.__dict__.update(state)

    def __copy__(self):
        cdef TradeInstruction instance = TradeInstruction.__new__(TradeInstruction)
        memcpy(<void*> &instance._data, <const char*> &self._data, sizeof(_TradeInstructionBuffer))
        return instance

    @classmethod
    def buffer_size(cls):
        return sizeof(_TradeInstructionBuffer)

    cdef bytes c_to_bytes(self):
        return PyBytes_FromStringAndSize(<char*> &self._data, sizeof(self._data))

    @staticmethod
    cdef TradeInstruction c_from_bytes(bytes data):
        cdef TradeInstruction instance = TradeInstruction.__new__(TradeInstruction)
        memcpy(<void*> &instance._data, <const char*> data, sizeof(_TradeInstructionBuffer))
        return instance

    def to_bytes(self) -> bytes:
        return self.c_to_bytes()

    @classmethod
    def from_bytes(cls, bytes data):
        return TradeInstruction.c_from_bytes(data)

    cpdef TradeInstruction reset(self):
        self.trades.clear()

        self._data.order_state = OrderStateCython.STATE_PENDING
        self._data.filled_volume = 0.
        self._data.filled_notional = 0.
        self._data.fee = 0.
        self._data.ts_placed = 0.
        self._data.ts_canceled = 0.
        self._data.ts_finished = 0.

        return self

    cpdef TradeInstruction reset_order_id(self, object order_id=None):
        if order_id is None:
            order_id = uuid.uuid4()

        for trade_report in self.trades.values():
            trade_report.reset_order_id(order_id=order_id)

        TransactionHelper.set_id(id_ptr=&self._data.order_id, id_value=order_id)
        return self

    cpdef TradeInstruction set_order_state(self, uint8_t order_state, double timestamp=NAN):
        if isnan(timestamp):
            timestamp = time.time()

        self._data.order_state = order_state

        if order_state == OrderStateCython.STATE_PLACED:
            self._data.ts_placed = timestamp
        elif order_state == OrderStateCython.STATE_FILLED:
            self._data.ts_finished = timestamp
        elif order_state == OrderStateCython.STATE_CANCELED:
            self._data.ts_canceled = timestamp

        return self

    cpdef TradeInstruction fill(self, TradeReport trade_report):
        # Check order_id match by comparing Python objects
        if not TransactionHelper.compare_id(&self._data.order_id, &trade_report._data.order_id):
            LOGGER.warning(f'Order ID mismatch! Instruction: {self.order_id}, Report: {trade_report.order_id}')
            return self

        # Check for duplicate trade
        if trade_report.trade_id in self.trades:
            LOGGER.warning(f'Duplicated trade received!\nInstruction {self}.\nReport {trade_report}.')
            return self

        # Check multiplier consistency
        if isnan(self._data.multiplier):
            self._data.multiplier = trade_report._data.multiplier
        elif self._data.multiplier != trade_report._data.multiplier:
            raise ValueError(f'Multiplier not match for order {self} and report {trade_report}.')

        # Check volume doesn't exceed order volume
        if trade_report._data.volume + self._data.filled_volume > self._data.volume:
            trades_str = '\n\t'.join([str(x) for x in self.trades.values()]) + f'\n\t<new> {trade_report}'
            LOGGER.warning('Fatal error!\nTradeInstruction: \n\t{}\nTradeReport:\n\t{}'.format(str(self), trades_str))
            raise ValueError('Fatal error! trade reports filled volume exceed order volume!')

        # Only process if there's volume
        if trade_report._data.volume:

            # Update filled values
            self._data.filled_volume += trade_report._data.volume
            self._data.filled_notional += trade_report._data.notional
            self._data.fee += trade_report._data.fee

        # Update order state
        if self._data.filled_volume == self.volume:
            self.set_order_state(order_state=OrderStateCython.STATE_FILLED, timestamp=trade_report._data.timestamp)
            self._data.ts_finished = trade_report._data.timestamp
        elif self._data.filled_volume > 0:
            self.set_order_state(order_state=OrderStateCython.STATE_PARTFILLED)

        # Add to trades dictionary
        self.trades[trade_report.trade_id] = trade_report

        return self

    cpdef TradeInstruction add_trade(self, TradeReport trade_report):
        self._data.fee += trade_report._data.fee
        self._data.filled_volume += trade_report._data.volume
        self._data.filled_notional += fabs(trade_report._data.notional)

        if self.filled_volume == self.volume:
            self.set_order_state(order_state=OrderStateCython.STATE_FILLED, timestamp=trade_report._data.timestamp)
        elif self.filled_volume > 0:
            self.set_order_state(order_state=OrderStateCython.STATE_PARTFILLED)

        self.trades[trade_report.trade_id] = trade_report
        return self

    cpdef TradeInstruction cancel_order(self, double timestamp=NAN):
        self.set_order_state(order_state=OrderStateCython.STATE_CANCELING, timestamp=timestamp)
        return self

    cpdef TradeInstruction canceled(self, double timestamp=NAN):
        self.set_order_state(order_state=OrderStateCython.STATE_CANCELED, timestamp=timestamp)
        return self

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
    def is_working(self) -> bool:
        return OrderStateHelper.is_working(order_state=self._data.order_state)

    @property
    def is_done(self) -> bool:
        return OrderStateHelper.is_done(order_state=self._data.order_state)

    @property
    def limit_price(self) -> float:
        return self._data.limit_price

    @property
    def volume(self) -> float:
        return self._data.volume

    @property
    def side_int(self) -> int:
        return self._data.side

    @property
    def side_sign(self) -> Literal[-1, 0, 1]:
        return TransactionHelper.get_sign(self._data.side)

    @property
    def side(self) -> TransactionSide:
        return TransactionSide(self.side_int)

    @property
    def order_type_int(self) -> int:
        return self._data.order_type

    @property
    def order_type(self) -> OrderType:
        return OrderType(self.order_type_int)

    @property
    def order_state_int(self) -> int:
        return self._data.order_state

    @property
    def order_state(self) -> OrderState:
        return OrderState(self.order_state_int)

    @property
    def multiplier(self) -> float:
        return self._data.multiplier

    @property
    def filled_volume(self) -> float:
        return self._data.filled_volume

    @property
    def working_volume(self):
        return self._data.volume - self._data.filled_volume

    @property
    def filled_notional(self) -> float:
        return self._data.filled_notional

    @property
    def fee(self) -> float:
        return self._data.fee

    @property
    def order_id(self) -> object:
        return TransactionHelper.get_id(&self._data.order_id)

    @property
    def average_price(self):
        if self.filled_volume != 0:
            return self._data.filled_notional / self._data.filled_volume / self._data.multiplier
        else:
            return NAN

    @property
    def start_time(self) -> datetime:
        return _MarketDataVirtualBase.c_to_dt(self._data.timestamp)

    @property
    def placed_time(self) -> datetime | None:
        cdef double ts_placed = self._data.ts_placed
        if ts_placed:
            return _MarketDataVirtualBase.c_to_dt(ts_placed)
        return None

    @property
    def canceled_time(self):
        cdef double ts_canceled = self._data.ts_canceled
        if ts_canceled:
            return _MarketDataVirtualBase.c_to_dt(ts_canceled)
        return None

    @property
    def finished_time(self):
        cdef double ts_finished = self._data.ts_finished
        if ts_finished:
            return _MarketDataVirtualBase.c_to_dt(ts_finished)
        return None
