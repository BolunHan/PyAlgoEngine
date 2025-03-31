# cython: language_level=3
import enum
import time
import uuid

from cpython.buffer cimport PyBuffer_FillInfo
from cpython.datetime cimport datetime
from cpython.mem cimport PyMem_Malloc
from libc.math cimport NAN, fabs, isnan
from libc.stdint cimport uint8_t
from libc.string cimport memcpy, memset

from .market_data cimport MarketData, _MarketDataBuffer, DataType, OrderType as OrderTypeCython, OrderState as OrderStateCython, _TradeReportBuffer, _TradeInstructionBuffer
from .transaction cimport TransactionData, TransactionHelper
from .transaction import TransactionSide, OrderType
from ..base import LOGGER
from ..profile import PROFILE


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


# Cython implementation of TradeReport
cdef class TradeReport(MarketData):
    _dtype = DataType.DTYPE_REPORT

    def __cinit__(self):
        """
        Allocate memory for the transaction data structure but don't initialize it.
        """
        self._owner = True

    def __init__(self, ticker: str, double timestamp, double price, double volume, uint8_t side, double notional=0., double multiplier=1., double fee=0., object order_id=None, object trade_id=None, **kwargs):
        if volume < 0:
            raise ValueError("Volume must be non-negative.")

        if not self._owner:
            raise MemoryError(f"Can not initialize a view of {self.__class__.__name__}.")

        self._data = <_MarketDataBuffer *> PyMem_Malloc(sizeof(_TradeReportBuffer))
        memset(self._data, 0, sizeof(_TradeReportBuffer))

        # Initialize base class fields
        MarketData.__init__(self, ticker=ticker, timestamp=timestamp, **kwargs)

        # Initialize report-specific fields
        self._data.TradeReport.price = price
        self._data.TradeReport.volume = volume
        self._data.TradeReport.side = side
        self._data.TradeReport.multiplier = multiplier
        self._data.TradeReport.fee = fee

        # Calculate notional if not provided
        if notional == 0.0:
            self._data.TradeReport.notional = price * volume * multiplier
        else:
            self._data.TradeReport.notional = notional

        # Initialize IDs
        if trade_id is None:
            TransactionData._set_id(id_ptr=&self._data.TradeReport.trade_id, id_value=uuid.uuid4())
        else:
            TransactionData._set_id(id_ptr=&self._data.TradeReport.trade_id, id_value=trade_id)
        if order_id is None:
            raise ValueError('Must assign an order_id.')
        else:
            TransactionData._set_id(id_ptr=&self._data.TradeReport.order_id, id_value=order_id)

    def __eq__(self, other: TradeReport):
        assert isinstance(other, self.__class__), f'Can only compare with {self.__class__.__name__}'

        # Fast check: only check the order id and trade id.
        if not self.order_id == other.order_id:
            return False
        elif not self.trade_id == other.trade_id:
            return False

        return True

    def __repr__(self) -> str:
        """
        String representation of the order data.
        """
        if self._data == NULL:
            return "TradeReport(uninitialized)"
        side_name = TransactionHelper.get_side_name(self._data.TradeReport.side).decode('utf-8')

        return f"<TradeReport id={self.trade_id}>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker} {side_name} {self.volume} at {self.price})"

    cpdef TradeReport reset_order_id(self, object order_id=None):
        if order_id is None:
            order_id = uuid.uuid4()
        TransactionData._set_id(id_ptr=&self._data.TradeReport.order_id, id_value=order_id)
        return self

    cpdef TradeReport reset_trade_id(self, object trade_id=None):
        if trade_id is None:
            trade_id = uuid.uuid4()
        TransactionData._set_id(id_ptr=&self._data.TradeReport.trade_id, id_value=trade_id)
        return self

    @classmethod
    def from_buffer(cls, const unsigned char[:] buffer):
        """
        Create a new instance from a buffer.
        """
        cdef TradeReport instance = cls.__new__(cls)

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
        cdef TradeReport instance = cls.__new__(cls)
        cdef const char* data_ptr = <const char*>data

        instance._owner = True
        instance._data = <_MarketDataBuffer*>PyMem_Malloc(sizeof(_TradeReportBuffer))

        if instance._data == NULL:
            raise MemoryError("Failed to allocate memory for TradeReport")

        memcpy(instance._data, data_ptr, sizeof(_TradeReportBuffer))

        return instance

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        """
        Implement the buffer protocol for read-only access.
        """
        if self._data == NULL:
            raise ValueError("Cannot get buffer from uninitialized data")
        PyBuffer_FillInfo(buffer, self, <_TradeReportBuffer*>self._data, sizeof(_TradeReportBuffer), 1, flags)

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
    def price(self) -> float:
        """
        Get the transaction price.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TradeReport.price

    @property
    def volume(self) -> float:
        """
        Get the transaction volume.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TradeReport.volume

    @property
    def side_int(self) -> int:
        """
        Get the transaction side.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TradeReport.side

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
        return self._data.TradeReport.multiplier

    @property
    def notional(self) -> float:
        """
        Get the transaction notional value.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TradeReport.notional

    @property
    def fee(self) -> float:
        """
        Get the transaction fee value.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TradeReport.fee

    @property
    def trade_id(self) -> object:
        """
        Get the transaction ID.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return TransactionData._get_id(&self._data.TradeReport.trade_id)

    @property
    def order_id(self) -> object:
        """
        Get the buy ID.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return TransactionData._get_id(&self._data.TradeReport.order_id)

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
        cdef int sign = TransactionHelper.get_sign(self._data.TradeReport.side)
        return sign * self._data.TradeReport.volume

    @property
    def notional_flow(self) -> float:
        """
        Calculate the flow of the transaction notional.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        cdef int sign = TransactionHelper.get_sign(self._data.TradeReport.side)
        return sign * self._data.TradeReport.notional

    @property
    def trade_time(self) -> datetime:
        return super().market_time


# Cython implementation of TradeInstruction
cdef class TradeInstruction(MarketData):
    _dtype = DataType.DTYPE_INSTRUCTION

    def __cinit__(self):
        self.trades = {}
        self._owner = True

    def __init__(self, str ticker, double timestamp, uint8_t side, double volume, uint8_t order_type=0, double limit_price=NAN, double multiplier=1., object order_id=None, **kwargs):
        if not self._owner:
            raise MemoryError(f"Can not initialize a view of {self.__class__.__name__}.")

        if not (volume > 0):
            raise ValueError("Volume must be positive")

        self._data = <_MarketDataBuffer *> PyMem_Malloc(sizeof(_TradeInstructionBuffer))
        memset(self._data, 0, sizeof(_TradeInstructionBuffer))

        # Initialize base class fields
        MarketData.__init__(self, ticker=ticker, timestamp=timestamp, **kwargs)

        # Initialize instruction-specific fields
        self._data.TradeInstruction.limit_price = limit_price
        self._data.TradeInstruction.volume = volume
        self._data.TradeInstruction.side = side
        self._data.TradeInstruction.order_type = order_type
        self._data.TradeInstruction.multiplier = multiplier

        # Initialize IDs
        if order_id is None:
            TransactionData._set_id(id_ptr=&self._data.TradeInstruction.order_id, id_value=uuid.uuid4())
        else:
            TransactionData._set_id(id_ptr=&self._data.TradeInstruction.order_id, id_value=order_id)

        self._data.TradeInstruction.order_state = OrderStateCython.STATE_PENDING
        self._data.TradeInstruction.filled_volume = 0.
        self._data.TradeInstruction.filled_notional = 0.
        self._data.TradeInstruction.fee = 0.
        self._data.TradeInstruction.ts_placed = 0.
        self._data.TradeInstruction.ts_canceled = 0.
        self._data.TradeInstruction.ts_finished = 0.

    def __eq__(self, other: TradeInstruction):
        assert isinstance(other, self.__class__), f'Can only compare with {self.__class__.__name__}'

        # Fast check: only check the order id
        if not self.order_id == other.order_id:
            return False

        return True

    def __repr__(self) -> str:
        """
        String representation of the Instruction data.
        """
        if self._data == NULL:
            return "<TradeInstruction>(uninitialized)"
        side_name = TransactionHelper.get_side_name(self._data.TradeInstruction.side).decode('utf-8')
        order_type_name = TransactionHelper.get_order_type_name(self._data.TradeInstruction.order_type).decode('utf-8')

        if self.limit_price is None or self.order_type_int == OrderTypeCython.ORDER_MARKET:
            return f'<TradeInstruction id={self.order_id}>({self.ticker} {order_type_name} {side_name} {self.volume}; filled {self.filled_volume:.2f} @ {self.average_price:.2f} now {self.order_state.state_name})'
        else:
            return f'<TradeInstruction id={self.order_id}>({self.ticker} {order_type_name} {side_name} {self.volume} limit {self.limit_price:.2f}; filled {self.filled_volume:.2f} @ {self.average_price:.2f} now {self.order_state.state_name})'

    @classmethod
    def from_buffer(cls, const unsigned char[:] buffer):
        """
        Create a new instance from a buffer.
        """
        cdef TradeInstruction instance = cls.__new__(cls)

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
        cdef TradeInstruction instance = cls.__new__(cls)
        cdef const char* data_ptr = <const char*>data

        instance._owner = True
        instance._data = <_MarketDataBuffer*>PyMem_Malloc(sizeof(_TradeInstructionBuffer))

        if instance._data == NULL:
            raise MemoryError("Failed to allocate memory for TradeInstruction")

        memcpy(instance._data, data_ptr, sizeof(_TradeInstructionBuffer))

        return instance

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        """
        Implement the buffer protocol for read-only access.
        """
        if self._data == NULL:
            raise ValueError("Cannot get buffer from uninitialized data")
        PyBuffer_FillInfo(buffer, self, <_TradeInstructionBuffer*>self._data, sizeof(_TradeInstructionBuffer), 1, flags)

    def __copy__(self):
        """
        Create a copy of this instance.
        """
        instance = super().__copy__()
        instance.trades = self.trades.copy()

        return instance

    cpdef TradeInstruction reset(self):
        self.trades.clear()

        self._data.TradeInstruction.order_state = OrderStateCython.STATE_PENDING
        self._data.TradeInstruction.filled_volume = 0.
        self._data.TradeInstruction.filled_notional = 0.
        self._data.TradeInstruction.fee = 0.
        self._data.TradeInstruction.ts_placed = 0.
        self._data.TradeInstruction.ts_canceled = 0.
        self._data.TradeInstruction.ts_finished = 0.

        return self

    cpdef TradeInstruction reset_order_id(self, object order_id=None):
        if order_id is None:
            order_id = uuid.uuid4()

        for trade_report in self.trades.values():
            trade_report.reset_order_id(order_id=order_id)

        TransactionData._set_id(id_ptr=&self._data.TradeInstruction.order_id, id_value=order_id)
        return self

    cpdef TradeInstruction set_order_state(self, uint8_t order_state, double timestamp=NAN):
        if isnan(timestamp):
            timestamp = time.time()

        self._data.TradeInstruction.order_state = order_state

        if order_state == OrderStateCython.STATE_PLACED:
            self._data.TradeInstruction.ts_placed = timestamp
        elif order_state == OrderStateCython.STATE_FILLED:
            self._data.TradeInstruction.ts_finished = timestamp
        elif order_state == OrderStateCython.STATE_CANCELED:
            self._data.TradeInstruction.ts_canceled = timestamp

        return self

    cpdef TradeInstruction fill(self, TradeReport trade_report):
        # Check for NULL pointers first
        if self._data == NULL or trade_report._data == NULL:
            raise ValueError("Uninitialized trade data")

        # Check order_id match by comparing Python objects
        if not TransactionData._id_equal(&self._data.TradeInstruction.order_id, &trade_report._data.TradeReport.order_id):
            LOGGER.warning(f'Order ID mismatch! Instruction: {self.order_id}, Report: {trade_report.order_id}')
            return self

        # Check for duplicate trade
        if trade_report.trade_id in self.trades:
            LOGGER.warning(f'Duplicated trade received!\nInstruction {self}.\nReport {trade_report}.')
            return self

        # Check multiplier consistency
        if isnan(self._data.TradeInstruction.multiplier):
            self._data.TradeInstruction.multiplier = trade_report._data.TradeReport.multiplier
        elif self._data.TradeInstruction.multiplier != trade_report._data.TradeReport.multiplier:
            raise ValueError(f'Multiplier not match for order {self} and report {trade_report}.')

        # Check volume doesn't exceed order volume
        if trade_report._data.TradeReport.volume + self._data.TradeInstruction.filled_volume > self._data.TradeInstruction.volume:
            trades_str = '\n\t'.join([str(x) for x in self.trades.values()]) + f'\n\t<new> {trade_report}'
            LOGGER.warning('Fatal error!\nTradeInstruction: \n\t{}\nTradeReport:\n\t{}'.format(str(self), trades_str))
            raise ValueError('Fatal error! trade reports filled volume exceed order volume!')

        # Only process if there's volume
        if trade_report._data.TradeReport.volume:

            # Update filled values
            self._data.TradeInstruction.filled_volume += trade_report._data.TradeReport.volume
            self._data.TradeInstruction.filled_notional += trade_report._data.TradeReport.notional
            self._data.TradeInstruction.fee += trade_report._data.TradeReport.fee

        # Update order state
        if self._data.TradeInstruction.filled_volume == self.volume:
            self.set_order_state(order_state=OrderStateCython.STATE_FILLED, timestamp=trade_report._data.TradeReport.timestamp)
            self._data.TradeInstruction.ts_finished = trade_report._data.TradeReport.timestamp
        elif self._data.TradeInstruction.filled_volume > 0:
            self.set_order_state(order_state=OrderStateCython.STATE_PARTFILLED)

        # Add to trades dictionary
        self.trades[trade_report.trade_id] = trade_report

        return self

    cpdef TradeInstruction add_trade(self, TradeReport trade_report):
        self._data.TradeInstruction.fee += trade_report._data.TradeReport.fee
        self._data.TradeInstruction.filled_volume += trade_report._data.TradeReport.volume
        self._data.TradeInstruction.filled_notional += fabs(trade_report._data.TradeReport.notional)

        if self.filled_volume == self.volume:
            self.set_order_state(order_state=OrderStateCython.STATE_FILLED, timestamp=trade_report.timestamp)
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
    def is_working(self) -> bool:
        return OrderStateHelper.is_working(order_state=self._data.TradeInstruction.order_state)

    @property
    def is_done(self) -> bool:
        return OrderStateHelper.is_done(order_state=self._data.TradeInstruction.order_state)

    @property
    def limit_price(self) -> float:
        """
        Get the transaction price.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TradeInstruction.limit_price

    @property
    def volume(self) -> float:
        """
        Get the transaction volume.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TradeInstruction.volume

    @property
    def side_int(self) -> int:
        """
        Get the transaction side.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TradeInstruction.side

    @property
    def side(self) -> TransactionSide:
        return TransactionSide(self.side_int)

    @property
    def order_type_int(self) -> int:
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TradeInstruction.order_type

    @property
    def order_type(self) -> OrderType:
        return OrderType(self.order_type_int)

    @property
    def order_state_int(self) -> int:
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TradeInstruction.order_state

    @property
    def order_state(self) -> OrderState:
        return OrderState(self.order_state_int)

    @property
    def multiplier(self) -> float:
        """
        Get the transaction multiplier.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TradeInstruction.multiplier

    @property
    def filled_volume(self) -> float:
        """
        Get the transaction notional value.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TradeInstruction.filled_volume

    @property
    def working_volume(self):
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TradeInstruction.volume - self._data.TradeInstruction.filled_volume

    @property
    def filled_notional(self) -> float:
        """
        Get the transaction notional value.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TradeInstruction.filled_notional

    @property
    def fee(self) -> float:
        """
        Get the transaction fee value.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return self._data.TradeInstruction.fee

    @property
    def order_id(self) -> object:
        """
        Get the buy ID.
        """
        if self._data == NULL:
            raise ValueError("Data not initialized")
        return TransactionData._get_id(&self._data.TradeInstruction.order_id)

    @property
    def average_price(self):
        if self.filled_volume != 0:
            return self._data.TradeInstruction.filled_notional / self._data.TradeInstruction.filled_volume / self._data.TradeInstruction.multiplier
        else:
            return NAN

    @property
    def start_time(self) -> datetime:
        return super().market_time

    @property
    def placed_time(self) -> datetime | None:
        cdef double ts_placed = self._data.TradeInstruction.ts_placed
        if ts_placed:
            return datetime.fromtimestamp(ts_placed, tz=PROFILE.time_zone)
        return None

    @property
    def canceled_time(self):
        cdef double ts_canceled = self._data.TradeInstruction.ts_canceled
        if ts_canceled:
            return datetime.fromtimestamp(ts_canceled, tz=PROFILE.time_zone)
        return None

    @property
    def finished_time(self):
        cdef double ts_finished = self._data.TradeInstruction.ts_finished
        if ts_finished:
            return datetime.fromtimestamp(ts_finished, tz=PROFILE.time_zone)
        return None
