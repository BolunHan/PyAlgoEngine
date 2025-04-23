import abc
import copy
import datetime
import enum
import json
import math
import time
import uuid
from typing import Self

from . import LOGGER, PROFILE, TransactionSide, TransactionData, OrderType

LOGGER = LOGGER.getChild('TradeUtils')
__all__ = ['OrderState', 'OrderType', 'TradeInstruction', 'TradeReport']


class OrderState(enum.IntEnum):
    UNKNOWN = -3
    Rejected = -2  # order rejected
    Invalid = -1  # invalid order
    Pending = 0  # order not sent. CAUTION pending order is not working nor done!
    Sent = 1  # order sent (to exchange)
    Placed = 2  # order placed in exchange
    PartFilled = 3  # order partial filled
    Filled = 4  # order fully filled
    Canceling = 5  # order canceling
    # PartCanceled = 5  # Deprecated
    Canceled = 6  # order stopped and canceled

    def __hash__(self):
        return self.value

    @property
    def is_working(self):
        """
        order in working status (ready to be filled),
        all non-working status are Pending / Filled / Cancelled / Rejected
        """
        if self.value == OrderState.Pending.value or \
                self.value == OrderState.Filled.value or \
                self.value == OrderState.Canceled.value or \
                self.value == OrderState.Invalid.value or \
                self.value == OrderState.Rejected.value:
            return False
        else:
            return True

    @property
    def is_done(self):
        if self.value == OrderState.Filled.value or \
                self.value == OrderState.Canceled.value or \
                self.value == OrderState.Rejected.value or \
                self.value == OrderState.Invalid.value:
            return True
        else:
            return False


class TradeBaseClass(dict, metaclass=abc.ABCMeta):
    def __init__(self, ticker: str, timestamp: float | None, **kwargs):
        super().__init__(ticker=ticker, timestamp=timestamp)

        if kwargs:
            self['additional'] = dict(kwargs)

    def __copy__(self):
        return self.__class__.__init__(**self)

    def copy(self):
        return self.__copy__()

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            dtype=self.__class__.__name__,
            **self
        )

        if 'additional' in data_dict:
            additional = data_dict.pop('additional')
            data_dict.update(additional)

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        dtype = json_dict.pop('dtype', None)
        if dtype == 'TradeReport':
            return TradeReport.from_json(json_dict)
        elif dtype == 'TickData':
            return TradeInstruction.from_json(json_dict)
        else:
            raise TypeError(f'Invalid dtype {dtype}')

    @abc.abstractmethod
    def to_list(self) -> list[float | int | str | bool]:
        ...

    @classmethod
    def from_list(cls, data_list: list[float | int | str | bool]) -> Self:
        dtype = data_list[0]

        if dtype == 'TradeReport':
            return TradeReport.from_list(data_list)
        elif dtype == 'TradeInstruction':
            return TradeInstruction.from_list(data_list)
        else:
            raise TypeError(f'Invalid dtype {dtype}')

    @property
    def ticker(self):
        return self['ticker']

    @property
    def timestamp(self):
        return self['timestamp']

    @property
    def additional(self):
        if 'additional' not in self:
            self['additional'] = {}
        return self['additional']

    @property
    def topic(self) -> str:
        return f'{self.ticker}.{self.__class__.__name__}'

    @property
    def market_time(self) -> datetime.datetime | datetime.date:
        return datetime.datetime.fromtimestamp(self.timestamp, tz=PROFILE.time_zone)


class TradeReport(TradeBaseClass):

    def __init__(
            self, *,
            ticker: str,
            side: int | float | str | TransactionSide,
            price: float,
            volume: float,
            timestamp: float,
            order_id: str,
            trade_id: str = None,
            notional: float = None,
            multiplier: float = None,
            fee: float = None,
            **kwargs
    ):
        assert volume >= 0, 'Trade volume must not be negative'
        assert notional >= 0, 'Trade notional must not be negative'

        super().__init__(ticker=ticker, timestamp=timestamp, **kwargs)

        self['price'] = price
        self['volume'] = volume
        self['side'] = int(side) if isinstance(side, (int, float)) else TransactionSide(side).value

        self['order_id'] = order_id

        if trade_id is not None:
            self['trade_id'] = trade_id

        if notional is not None and math.isfinite(notional):
            self['notional'] = notional

        if multiplier is not None and math.isfinite(multiplier):
            self['multiplier'] = multiplier

        if fee is not None and math.isfinite(fee):
            self['fee'] = fee

    def __eq__(self, other: Self):
        assert isinstance(other, self.__class__), f'Can only compare with {self.__class__.__name__}'

        # Fast check: only check the order id and trade id.
        if not self.order_id == other.order_id:
            return False
        elif not self.trade_id == other.trade_id:
            return False

        return True

    def __str__(self):
        return f'<TradeReport id={self.trade_id}>([{self.market_time:%Y-%m-%d %H:%M:%S}] {self.ticker} {TransactionSide(self.side).side_name} {self.volume} at {self.price})'

    def __reduce__(self):
        return self.__class__.from_json, (self.to_json(),)

    def reset_order_id(self, order_id: int | str = None, _ignore_warning: bool = False) -> Self:
        if not _ignore_warning:
            LOGGER.warning('TradeReport OrderID being reset manually! TradeInstruction.reset_order_id() is the recommended method to do so.')

        if order_id is not None:
            self['order_id'] = order_id
        else:
            self['order_id'] = uuid.uuid4().int

        return self

    def reset_trade_id(self, trade_id: int | str = None) -> Self:
        if trade_id is not None:
            self['trade_id'] = trade_id
        else:
            self['trade_id'] = uuid.uuid4().int

        return self

    def to_trade(self) -> TransactionData:
        trade = TransactionData(
            ticker=self.ticker,
            timestamp=self.timestamp,
            price=self.price,
            volume=self.volume,
            side=self.side,
            multiplier=self.multiplier
        )
        return trade

    def to_list(self) -> list[float | int | str | bool]:
        return [self.__class__.__name__,
                self.ticker,
                self.timestamp,
                self.price,
                self.volume,
                self['side'],
                self.get('multiplier'),
                self.get('notional'),
                self.get('fee'),
                self.get('trade_id'),
                self.order_id]

    def copy(self, **kwargs):
        new_trade = self.__class__(
            ticker=kwargs.pop('ticker', self.ticker),
            side=kwargs.pop('side', self.side),
            price=kwargs.pop('price', self.price),
            volume=kwargs.pop('volume', self.volume),
            notional=kwargs.pop('notional', self.notional),
            timestamp=kwargs.pop('timestamp', self.timestamp),
            order_id=kwargs.pop('order_id', self.order_id),
            trade_id=kwargs.pop('trade_id', f'{self.trade_id}.copy'),
            multiplier=kwargs.pop('multiplier', self.multiplier),
            fee=kwargs.pop('fee', self.fee)
        )

        return new_trade

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        dtype = json_dict.pop('dtype', None)
        if dtype is not None and dtype != cls.__name__:
            raise TypeError(f'dtype mismatch, expect {cls.__name__}, got {dtype}.')

        self = cls(**json_dict)
        return self

    @classmethod
    def from_list(cls, data_list: list[float | int | str | bool]) -> Self:
        (dtype, ticker, timestamp, price, volume, side,
         multiplier, notional, fee, trade_id, order_id) = data_list

        if dtype != cls.__name__:
            raise TypeError(f'dtype mismatch, expect {cls.__name__}, got {dtype}.')

        kwargs = {}

        if trade_id is not None:
            kwargs['trade_id'] = trade_id

        if notional is not None and math.isfinite(notional):
            kwargs['notional'] = notional

        if multiplier is not None and math.isfinite(multiplier):
            kwargs['multiplier'] = multiplier

        if fee is not None and math.isfinite(fee):
            kwargs['fee'] = fee

        return cls(
            ticker=ticker,
            timestamp=timestamp,
            price=price,
            volume=volume,
            side=side,
            order_id=order_id,
            **kwargs
        )

    @classmethod
    def from_trade(cls, trade_data: TransactionData, order_id: str, trade_id: str = None) -> Self:
        report = cls(
            ticker=trade_data.ticker,
            side=trade_data.side,
            volume=trade_data.volume,
            price=trade_data.price,
            notional=trade_data.notional,
            timestamp=trade_data.timestamp,
            order_id=order_id,
            trade_id=trade_id
        )
        return report

    @property
    def price(self) -> float:
        return self['price']

    @property
    def volume(self) -> float:
        return self['volume']

    @property
    def side(self) -> TransactionSide:
        return TransactionSide(self['side'])

    @property
    def multiplier(self) -> float:
        return self.get('multiplier', 1.)

    @property
    def fee(self) -> float:
        return self.get('fee', 0.)

    @property
    def order_id(self) -> int | str:
        return self['order_id']

    @property
    def trade_id(self) -> int | str:
        if 'trade_id' in self:
            trade_id = self['trade_id']
        else:
            trade_id = self['trade_id'] = uuid.uuid4().int

        return trade_id

    @property
    def notional(self) -> float:
        return self.get('notional', self.price * self.volume * self.multiplier)

    @property
    def market_price(self) -> float:
        return self.price

    @property
    def flow(self):
        return self.side.sign * self.volume

    @property
    def trade_time(self) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(self.timestamp, tz=PROFILE.time_zone)


class TradeInstruction(TradeBaseClass):
    def __init__(
            self, *,
            ticker: str,
            side: int | float | str | TransactionSide,
            volume: float,
            timestamp: float,
            order_type: int | float | str | OrderType = OrderType.Generic,
            limit_price: float = None,
            order_id: str = None,
            multiplier: float = None,
            **kwargs
    ):
        assert volume > 0, f'Invalid trade volume {volume}!'
        super().__init__(ticker=ticker, timestamp=timestamp, **kwargs)

        self['volume'] = volume
        self['side'] = int(side) if isinstance(side, (int, float)) else TransactionSide(side).value
        self['order_type'] = int(order_type) if isinstance(order_type, (int, float)) else OrderType(order_type).value
        self['order_id'] = order_id if order_id is not None else uuid.uuid4().int

        if limit_price is not None and math.isfinite(limit_price):
            self['limit_price'] = limit_price

        if multiplier is not None and math.isfinite(multiplier):
            self['multiplier'] = multiplier

        self['order_state'] = OrderState.Pending.value
        self['filled_volume'] = 0.
        self['filled_notional'] = 0.
        self['fee'] = 0.

        # note that 3 additional entries might be added to the TradeInstruction
        # self['ts_placed'] = timestamp
        # self['ts_canceled'] = timestamp
        # self['ts_finished'] = timestamp

        self.trades: dict[int | str, TradeReport] = {}

    def __eq__(self, other: Self):
        assert isinstance(other, self.__class__), f'Can only compare with {self.__class__.__name__}'

        # Fast check: only check the order id and trade id.
        if not self.order_id == other.order_id:
            return False

        return True

    def __str__(self):
        if self.limit_price is None or self.order_type == OrderType.MarketOrder:
            return f'<TradeInstruction id={self.order_id}>({self.ticker} {self.order_type.name} {self.side.name} {self.volume}; filled {self.filled_volume:.2f} @ {self.average_price:.2f} now {self.order_state.name})'
        else:
            return f'<TradeInstruction id={self.order_id}>({self.ticker} {self.order_type.name} {self.side.name} {self.volume} limit {self.limit_price:.2f}; filled {self.filled_volume:.2f} @ {self.average_price:.2f} now {self.order_state.name})'

    def __reduce__(self):
        return self.__class__.from_json, (self.to_json(),)

    def reset(self):
        self.trades.clear()
        self.order_state = OrderState.Pending

        self['filled_volume']: float = 0.0
        self['filled_notional']: float = 0.0
        self['fee'] = .0

        self.pop('ts_placed', None)
        self.pop('ts_canceled', None)
        self.pop('ts_finished', None)

    def reset_order_id(self, order_id: int | str = None, _ignore_warning: bool = False) -> Self:
        if not _ignore_warning:
            LOGGER.warning(f'{self.__class__.__name__} OrderID being reset manually! Position.reset_order_id() is the recommended method to do so.')

        if order_id is not None:
            self['order_id'] = order_id
        else:
            self['order_id'] = uuid.uuid4().int

        for trade_report in self.trades.values():
            trade_report.reset_order_id(order_id=self.order_id, _ignore_warning=True)

        return self

    def set_order_state(self, order_state: OrderState, timestamp: float = time.time()) -> Self:
        self.order_state = order_state

        # assign a start_datetime if order placed
        if order_state == OrderState.Placed:
            self['ts_placed'] = timestamp

        elif order_state == OrderState.Filled:
            self['ts_finished'] = timestamp

        if order_state == OrderState.Canceled:
            self['ts_canceled'] = timestamp
            self['ts_finished'] = timestamp

        return self

    def fill(self, trade_report: TradeReport) -> Self:
        if trade_report.order_id != self.order_id:
            LOGGER.warning(f'Order ID not match! Instruction ID {self.order_id}; Report ID {trade_report.order_id}')
            return self

        if trade_report.trade_id in self.trades:
            LOGGER.warning(f'Duplicated trade received!\nInstruction {self}.\nReport {trade_report}.')
            return self

        if trade_report.volume:
            # update multiplier
            if 'multiplier' in trade_report:
                if 'multiplier' not in self:
                    self['multiplier'] = trade_report.multiplier
                elif trade_report.multiplier != self['multiplier']:
                    raise ValueError(f'Multiplier not match for order {self} and report {trade_report}.')

            if trade_report.volume + self.filled_volume > self.volume:
                LOGGER.warning('Fatal error!\nTradeInstruction: \n\t{}\nTradeReport:\n\t{}'.format(str(TradeInstruction), '\n\t'.join([str(x) for x in self.trades.values()]) + f'\n\t<new> {trade_report}'))
                raise ValueError('Fatal error! trade reports filled volume exceed order volume!')

            self['filled_volume'] += abs(trade_report.volume)
            self['filled_notional'] += abs(trade_report.notional)

        if self.filled_volume == self.volume:
            self.set_order_state(order_state=OrderState.Filled, timestamp=trade_report.timestamp)
            self['ts_finished'] = trade_report.timestamp
        elif self.filled_volume > 0:
            self.set_order_state(order_state=OrderState.PartFilled)

        self.trades[trade_report.trade_id] = trade_report

        return self

    def cancel_order(self) -> Self:
        self.set_order_state(order_state=OrderState.Canceling)

        cancel_instruction = copy.copy(self)
        cancel_instruction.set_order_state(order_state=OrderState.Canceled)

        return cancel_instruction

    def canceled(self, timestamp: float) -> Self:
        LOGGER.warning(DeprecationWarning('[canceled] depreciated! Use [set_order_state] instead!'), stacklevel=2)

        self.set_order_state(order_state=OrderState.Canceled, timestamp=timestamp)
        return self

    def to_json(self, with_trade=True, fmt: str = 'str') -> str | dict:
        json_dict = super().to_json()

        if self.trades:
            json_dict['trade'] = [report.to_json(fmt='dict') for report in self.trades.values()]

        if fmt == 'dict':
            return json_dict
        else:
            return json.dumps(json_dict)

    def to_list(self) -> list[float | int | str | bool]:
        return [self.__class__.__name__,
                self.ticker,
                self.timestamp,
                self.limit_price,
                self.volume,
                self['side'],
                self['order_state'],
                self.get('multiplier'),
                self.get('notional'),
                self.filled_volume,
                self.filled_notional,
                self.fee,
                self.get('order_id')]

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        dtype = json_dict.pop('dtype', None)
        trades = json_dict.pop('trades', [])

        if dtype is not None and dtype != cls.__name__:
            raise TypeError(f'dtype mismatch, expect {cls.__name__}, got {dtype}.')

        self = cls(**json_dict)

        for trade_json in json_dict['trades']:
            report = TradeReport.from_json(trade_json)
            self.trades[report.trade_id] = report

        return self

    @classmethod
    def from_list(cls, data_list: list[float | int | str | bool]) -> Self:
        (dtype, ticker, timestamp, limit_price, volume, side,
         order_state, multiplier, notional, filled_volume, filled_notional, fee, order_id) = data_list

        if dtype != cls.__name__:
            raise TypeError(f'dtype mismatch, expect {cls.__name__}, got {dtype}.')

        kwargs = {}

        if notional is not None and math.isfinite(notional):
            kwargs['notional'] = notional

        if multiplier is not None and math.isfinite(multiplier):
            kwargs['multiplier'] = multiplier

        if fee is not None and math.isfinite(fee):
            kwargs['fee'] = fee

        self = cls(
            ticker=ticker,
            timestamp=timestamp,
            limit_price=limit_price,
            volume=volume,
            side=side,
            order_id=order_id,
            **kwargs
        )

        self['filled_volume'] = filled_volume
        self['filled_notional'] = filled_notional
        self['fee'] = fee

        return self

    @property
    def is_working(self):
        return self.order_state.is_working

    @property
    def is_done(self):
        return self.order_state.is_done

    @property
    def limit_price(self) -> float:
        return self['limit_price']

    @property
    def volume(self) -> float:
        return self['volume']

    @property
    def side(self) -> TransactionSide:
        return TransactionSide(self['side'])

    @property
    def multiplier(self) -> float:
        return self.get('multiplier', 1.)

    @property
    def fee(self) -> float:
        return self['fee']

    @fee.setter
    def fee(self, value: float):
        self['fee'] = value

    @property
    def order_id(self) -> int | str:
        return self['order_id']

    @property
    def order_type(self) -> OrderType:
        return OrderType(self['order_type'])

    @property
    def order_state(self) -> OrderState:
        return OrderState(self['order_state'])

    @order_state.setter
    def order_state(self, value: OrderState):
        if isinstance(value, int):
            self['order_state'] = value
        else:
            self['order_state'] = OrderState(value).value

    @property
    def filled_volume(self) -> float:
        return self['filled_volume']

    @property
    def working_volume(self) -> float:
        return self.volume - self.filled_volume

    @property
    def filled_notional(self) -> float:
        return self['filled_notional']

    @property
    def average_price(self) -> float:
        if self.filled_volume != 0:
            return self.filled_notional / self.filled_volume / self.multiplier
        else:
            return float('NaN')

    @property
    def start_time(self) -> datetime.datetime | None:
        return datetime.datetime.fromtimestamp(self.timestamp, tz=PROFILE.time_zone)

    @property
    def placed_time(self) -> datetime.datetime | None:
        if 'ts_placed' in self:
            return datetime.datetime.fromtimestamp(self['ts_placed'], tz=PROFILE.time_zone)

        return None

    @property
    def canceled_time(self) -> datetime.datetime | None:
        if 'ts_canceled' in self:
            return datetime.datetime.fromtimestamp(self['ts_canceled'], tz=PROFILE.time_zone)

        return None

    @property
    def finished_time(self) -> datetime.datetime | None:
        if 'ts_finished' in self:
            return datetime.datetime.fromtimestamp(self['ts_finished'], tz=PROFILE.time_zone)

        return None


class TradeOrder(TradeInstruction):
    pass