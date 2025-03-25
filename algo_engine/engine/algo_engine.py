import abc
import datetime
import enum
import functools
import json
import threading
import uuid
from typing import Type, TYPE_CHECKING

import numpy as np

from . import LOGGER
from .market_engine import MDS, Singleton
from ..base import TransactionSide, TradeInstruction, MarketData, TradeReport, OrderState, OrderType

LOGGER = LOGGER.getChild('AlgoEngine')
if TYPE_CHECKING:
    from .trade_engine import DirectMarketAccess

__all__ = ['AlgoTemplate', 'AlgoRegistry', 'AlgoEngine', 'ALGO_ENGINE', 'ALGO_REGISTRY']


class AlgoStatus(enum.Enum):
    idle = 'idle'  # init state
    preparing = 'preparing'  # preparing
    ready = 'ready'  # ready to launch order
    working = 'working'  # order launched
    done = 'done'  # transaction complete!
    closed = 'closed'  # transaction failed and close
    stopping = 'stopping'  # trying to stop transaction
    rejected = 'rejected'  # internal / external rejected
    error = 'error'  # internal / external error


class AlgoTemplate(object, metaclass=abc.ABCMeta):
    Status = AlgoStatus

    def __init__(self, dma: 'DirectMarketAccess', ticker: str, target_volume: float, side: TransactionSide, **kwargs):
        """ Template for trading algorithm
        an abstract class to create a trading algorithm

        :param dma: direct market access
        :param ticker: the given symbol of the underlying to trade
        :param target_volume: the given volume to trade
        :param side: the given TransactionSide
        :keyword algo_engine: the algo_engine instance, default is ALGO_ENGINE
        :keyword logger: the logger instance, default is LOGGER
        :keyword algo_id: the id of the algo, default is uuid4()
        """
        self.dma = dma
        self.ticker = ticker
        self.side = side
        self.target_volume = target_volume
        self.algo_engine = kwargs.pop('algo_engine', ALGO_ENGINE)
        self.algo_type = kwargs.get('algo_type', self.algo_engine.registry.reversed_registry[self.__class__.__name__])
        self.logger = kwargs.pop('logger', LOGGER)
        self.algo_id = kwargs.pop('algo_id', uuid.uuid4().hex)

        self.status: AlgoStatus = self.Status.idle
        self._target_progress = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self.work)

        self.working_order: dict[str, TradeInstruction] = {}
        self.order: dict[str, TradeInstruction] = {}

        self.is_active = False

        self.ts_started = None
        self.ts_finished = None

    def __repr__(self):
        return f'<TradeAlgo>(ticker={self.ticker}, target={self.side.sign * self.target_volume}, done={self.side.sign * self.exposure_volume}, algo={self.__class__.__name__}, status={self.status.value}, id={id(self)})'

    def on_sync_progress(self, progress: float, **kwargs):
        self._target_progress = max(min(progress, 1), 0)
        self._sync(progress=progress, **kwargs)

    def on_market_data(self, market_data: MarketData, **kwargs):
        pass

    def on_filled(self, report: TradeReport, **kwargs):
        if report.order_id in self.working_order:
            self._filled(order=self.working_order[report.order_id], report=report, **kwargs)
            return 1
        else:
            self.logger.warning(f'[Failed to fill] {self} has no matching for working order {report.order_id}')
            return 0

    def on_canceled(self, order_id: str = None, **kwargs):
        if order_id in self.working_order:
            self._canceled(order=self.working_order[order_id], **kwargs)
            return 1
        else:
            self.logger.warning(f'[Failed to cancel] {self} has no matching for working order {order_id}')
            return 0

    def on_rejected(self, order: TradeInstruction, **kwargs):
        if order.order_id in self.working_order:
            self._rejected(order=order, **kwargs)
            return 1
        else:
            self.logger.warning(f'[Failed to reject] {self} has no matching for working order {order.order_id}')
            return 0

    def recover(self):
        self._update_working_order()

        if not self.working_volume:
            if self.exposure_volume:
                self._assign_status(status=self.Status.done)
            else:
                self._assign_status(status=self.Status.closed)
            LOGGER.info(f'{self} recovery successful! status {self.status}')
        else:
            LOGGER.warning(f'Caution! Recovering WORKING trade handler {self} may cause unexpected error!')
            self._assign_status(status=self.Status.working)

    def _update_working_order(self):
        """
        refresh working order, to remove the finished orders
        :return: a dict of working orders
        """
        for order_id in list(self.working_order):
            order = self.working_order.get(order_id)

            if order is None:
                continue

            if order.is_done:
                self.working_order.pop(order_id, None)

        return self.working_order

    def _assign_status(self, status: Status, timestamp: float = None, **kwargs):
        if not isinstance(status, self.Status):
            raise TypeError(f'Invalid status {status}! Expect {self.Status}, got {type(status)}.')

        state_0, state_1 = self.status, status

        if timestamp is None:
            timestamp = self.timestamp

        if 'market_time' in kwargs:
            LOGGER.warning(DeprecationWarning('Assigning market_time deprecated, use timestamp instead!'))

        if state_0 == self.Status.idle and state_1 == self.Status.working:
            self.ts_started = timestamp
        elif state_1 == self.Status.done or state_1 == self.Status.closed:
            self.ts_finished = timestamp

        self.status = state_1

    def _update_status(self, status=None, sync_pos=True, **kwargs):
        """
        ._update_status provides a method to clear working orders and auto assign status.
        ._update_status DOES NOT call .on_filled, .on_rejected nor .on_canceled, these method is triggered by position management service.
        ._update_status should be called in .on_filled .on_rejected and .on_canceled.

        as the result of concurrency, assigning status while sync_pos may cause unexpected result, use with caution

        :param status: the given status
        :param sync_pos: whether to auto-clear working orders
        :param kwargs: market_time to assign the exact time when status is changed, used in backtesting
        """
        if sync_pos:
            self._update_working_order()

        # if 'market_time' in kwargs:
        #     market_time: datetime.datetime = kwargs['market_time']
        #     self.ts_started = market_time.timestamp()

        # assign status with given status
        if status is not None:
            return self._assign_status(status=status)

        # update status with self info
        if self.working_order:
            if self.status == self.Status.idle:
                self.status = self.Status.working
                self.ts_started = self.timestamp
        else:
            if self.filled_volume == self.target_volume:
                self.status = self.Status.done
                self.ts_finished = self.timestamp

        return self.status

    def _launch(self, order, **kwargs):
        self.dma.launch_order(order=order, **kwargs)
        # order launched, order state can be pending, placed, or rejected (by internal on_order risk control)
        # DO NOT assume order state is_working, it may be rejected!
        # DO NOT assume order is in .working_order, it may be rejected!
        # DO NOT assume algo state is working, it may be rejected!
        # therefor calling _update_status is recommended but still optional.
        # self._update_status(sync_pos=False)

    def _cancel_order(self, order, **kwargs):
        self.dma.cancel_order(order=order, **kwargs)
        # self._update_status(sync_pos=False)

    def _filled(self, order: TradeInstruction, report: TradeReport, **kwargs):
        """
        callback on order filled / part-filled

        this callback will REMOVE filled order from working order dict and update algo status
        :param order: the given filled order
        :param kwargs: keyword args for updating status. e.g. timestamp
        """
        if report.trade_id not in order.trades:
            order.fill(trade_report=report)

        kwargs['sync_pos'] = True
        self._update_status(**kwargs)

    def _canceled(self, order: TradeInstruction, **kwargs):
        """
        callback on order canceled

        this callback will REMOVE cancelled order from working order dict and update algo status
        :param order: the given canceled order
        :param kwargs: keyword args for updating status. e.g. timestamp
        """
        self._update_working_order()

        if self.working_order:
            self._assign_status(status=self.Status.working, **kwargs)
        elif self.exposure_volume:
            self._assign_status(status=self.Status.done, **kwargs)
        else:
            self._assign_status(status=self.Status.closed, **kwargs)

    def _rejected(self, order: TradeInstruction, **kwargs):
        self._assign_status(status=self.Status.rejected, **kwargs)

    def _sync(self, progress, **kwargs):
        ...

    def to_json(self, fmt='str') -> str | dict:
        json_dict = {
            'algo_type': self.algo_type,
            'ticker': self.ticker,
            'side': self.side.name,
            'target_volume': self.target_volume,
            'algo_id': self.algo_id,
            'status': self.status.name,
            'target_progress': self._target_progress,
            'ts_started': self.ts_started,
            'ts_finished': self.ts_finished,
            'order': {_: self.order[_].to_json(fmt='dict') for _ in self.order},
        }

        if fmt == 'dict':
            return json_dict
        else:
            return json.dumps(json_dict)

    def from_json(self, json_str: str | dict):
        if isinstance(json_str, (str, bytes)):
            json_dict = json.loads(json_str)
        elif isinstance(json_str, dict):
            json_dict = json_str
        else:
            raise TypeError(f'Invalid type {type(json_str)}, expect [str, bytes, dict]')

        self.ticker = json_dict['ticker']
        self.side = TransactionSide(json_dict['side'])
        self.target_volume = json_dict['target_volume']
        self.algo_id = json_dict['algo_id']
        self.status = self.Status[json_dict['status']]
        self._target_progress = json_dict['target_progress']
        self.ts_started = json_dict['ts_started']
        self.ts_finished = json_dict['ts_finished']
        self.order = {_: TradeInstruction.from_json(json_dict['order'][_]) for _ in json_dict['order']}
        self.working_order = {order_id: order for order_id, order in self.order.items() if not order.is_done}

        return self

    @abc.abstractmethod
    def work(self):
        ...

    @abc.abstractmethod
    def launch(self, **kwargs) -> list[TradeInstruction]:
        """
        launch is a method to initiate the algo and launching orders.
        this method will set the algo is_active = true
        this method will set a new algo state, usually idle -> working
        launch method is designed to be called by strategy / position management service.

        :param kwargs: other keywords needed to launch an algo
        :return: a list of working orders. Noted, that not all working order is returned by this method, for example, TWAP algo will init a sequence of order and return later.
        """
        ...

    @abc.abstractmethod
    def cancel(self, **kwargs):
        """
        cancel is a method to cancel / stop ALL working orders
        this method will set the algo is_active = false
        this method may set a new algo state, usually working -> stopping
        launch method is designed to be called by strategy / position management service.

        :param kwargs: other keywords needed to cancel an algo
        :return: None
        """
        ...

    @property
    def trades(self) -> dict[str, TradeReport]:
        trades = {}

        for order in list(self.order.values()):
            for trade_id in list(order.trades):
                trade_report = order.trades.get(trade_id)

                if trade_report is None:
                    continue

                trades[trade_report.trade_id] = trade_report

        return trades

    @property
    def average_price(self) -> float:
        adjust_volume = 0.
        notional = 0.

        for report in list(self.trades.values()):
            if report.price == 0:
                adjust_volume += report.volume
            else:
                adjust_volume += report.notional / report.price
            notional += report.notional

        if adjust_volume == 0:
            return np.nan
        else:
            return notional / adjust_volume

    @property
    def exposure_volume(self) -> float:
        """
        <WITH SIGN> net exposed VOLUME indicating the exposure of the pos
        :return: float
        """
        exposure = 0.

        for report in list(self.trades.values()):
            exposure += report.volume * report.side.sign

        return exposure

    @property
    def working_volume(self) -> float:
        """
        <WITHOUT SIGN> net working VOLUME indicating the working status of the pos
        :return: float
        """
        working = 0.

        for order_id in list(self.working_order):
            working_order = self.working_order.get(order_id)

            if working_order is None:
                continue

            working += working_order.working_volume  # should be all positive

        return working

    @property
    def filled_volume(self) -> float:
        """
        <WITHOUT SIGN> filled VOLUME
        :return: float
        """
        volume = 0.

        for report in list(self.trades.values()):
            volume += report.volume

        return volume

    @property
    def filled_notional(self) -> float:
        """
        <POSSIBLY WITH SIGN> total filled Notional
        :return: float
        """
        notional = 0.

        for report in list(self.trades.values()):
            notional += report.notional  # which should be a POSITIVE number in normal cases.

        return notional

    @property
    def fee(self) -> float:
        """
        <POSSIBLY WITH SIGN> total transaction fee
        :return: float
        """
        total_fee = 0.

        for report in list(self.trades.values()):
            total_fee += report.fee

        return total_fee

    @property
    def cash_flow(self) -> float:
        """
        <WITH SIGN> total cash flow
        :return: float
        """
        cash_flow = -self.filled_notional * self.side.sign
        return cash_flow

    @property
    def multiplier(self) -> float:
        if self.order:
            return self.order[list(self.order)[0]].multiplier
        else:
            return 1.0

    @property
    def filled_progress(self):
        return self.filled_volume / self.target_volume

    @property
    def placed_progress(self):
        return abs(self.working_volume / self.target_volume) + self.filled_progress

    @property
    def target_progress(self):
        return self._target_progress

    @property
    def market_price(self):
        return self.algo_engine.mds.market_price.get(self.ticker)

    @property
    def market_time(self) -> datetime.datetime:
        return self.algo_engine.mds.market_time

    @property
    def timestamp(self) -> float:
        return self.algo_engine.mds.timestamp

    @property
    def start_time(self) -> datetime.datetime | None:
        if self.ts_started is None:
            return None

        return datetime.datetime.fromtimestamp(self.ts_started, tz=self.algo_engine.mds.profile.time_zone)

    @property
    def finish_time(self) -> datetime.datetime | None:
        if self.ts_finished is None:
            return None

        return datetime.datetime.fromtimestamp(self.ts_finished, tz=self.algo_engine.mds.profile.time_zone)


class Passive(AlgoTemplate):
    """ Passive trading algorithm
    Passive is a basic trading algo which trades all target volume into one single LIMIT order.
    Algo will stop after order get filled or canceled.
    no additional order will be launched except the initial one

    a limit price can be set by keyword arguments, see also in doc: algo_engine.calculate_limit

    """

    def __init__(self, **kwargs):
        """
        init a Passive trade algo

        requires all params from AlgoTemplate and additional following 4
        :keyword limit_price: the absolute limit price of the order
        :keyword limit_adjust_factor: limit price = market_price * (1 + factor) for long order else limit price = market_price * (1 - factor) for short order
        :keyword limit_adjust_level:  for long order, limit price = bid[lvl] if lvl > 0 else ask[lvl] for lvl < 0.
        :keyword limit_mode: if multiple limit price standard is provided, use "strict" to select strictest limit price or "loose" to select loosest one. Default is None, which is "strict".
        """
        self.limit_price = kwargs.pop('limit_price', None)
        self.limit_adjust_factor = kwargs.pop('limit_adjust_factor', None)
        self.limit_adjust_level = kwargs.pop('limit_adjust_level', None)
        self.limit_mode = kwargs.pop('limit_mode', None)

        super().__init__(**kwargs)

    def work(self):
        pass

    def launch(self, **kwargs):
        if self.is_active:
            raise RuntimeError(f'{self} is working already')

        self.is_active = True

        limit_price = kwargs.pop('limit_price', self.limit_price)
        limit_adjust_factor = kwargs.pop('limit_adjust_factor', self.limit_adjust_factor)
        limit_adjust_level = kwargs.pop('limit_adjust_level', self.limit_adjust_level)
        limit_mode = kwargs.pop('limit_mode', self.limit_mode)

        limit = self.algo_engine.calculate_limit(
            algo=self,
            limit_price=limit_price,
            limit_adjust_factor=limit_adjust_factor,
            limit_adjust_level=limit_adjust_level,
            mode=limit_mode
        )
        order_type = OrderType.ORDER_LIMIT
        volume = self.target_volume - self.filled_volume - self.working_volume

        LOGGER.info(f'{self} launching {order_type} {self.ticker} {self.side.name} {volume}')

        if volume:
            order = TradeInstruction(
                ticker=self.ticker,
                side=self.side,
                order_type=order_type,
                volume=volume,
                limit_price=limit,
                order_id=f'{self.__class__.__name__}.{self.ticker}.{self.side.side_name}.{uuid.uuid4().hex}',
                timestamp=self.dma.timestamp
            )

            self.working_order[order.order_id] = order
            self.order[order.order_id] = order
            self.ts_started = self.dma.timestamp
            self._launch(order=order, **kwargs)

    def cancel(self, **kwargs):
        self.status = self.Status.stopping
        self.is_active = False
        self._cancel_all_order(**kwargs)

    def _cancel_all_order(self, **kwargs):
        for order_id in list(self.working_order):
            order = self.working_order.get(order_id)

            if order is None:
                continue

            if order.order_state in [OrderState.Pending, OrderState.Placed, OrderState.PartFilled]:
                LOGGER.info(f'{self} canceling {order}')
                self.dma.cancel_order(order=order, **kwargs)

    def _rejected(self, order: TradeInstruction, **kwargs):
        super()._rejected(order=order)

        if not self.exposure_volume:
            self._assign_status(status=self.Status.closed)
        else:
            self._assign_status(status=self.Status.done)

    def _filled(self, order: TradeInstruction, report: TradeReport, **kwargs):
        super()._filled(order=order, report=report, **kwargs)

        if order.order_id not in self.working_order:
            if self.status == self.Status.working:
                if self.filled_volume:
                    self._assign_status(status=self.Status.done)
                else:
                    self._assign_status(status=self.Status.closed)

    def _canceled(self, order: TradeInstruction, **kwargs):
        super()._canceled(order=order, **kwargs)

        if order.order_id not in self.working_order:
            if self.status == self.Status.working:
                if self.filled_volume:
                    self._assign_status(status=self.Status.done)
                else:
                    self._assign_status(status=self.Status.closed)

        if not self.is_active:
            self._assign_status(status=self.Status.done)

    def to_json(self, fmt='str') -> str | dict:
        json_dict = super().to_json(fmt='dict')

        additional_dict = dict(
            limit_price=self.limit_price,
            limit_adjust_factor=self.limit_adjust_factor,
            limit_adjust_level=self.limit_adjust_level,
            limit_mode=self.limit_mode
        )

        json_dict.update(additional_dict)

        if fmt == 'dict':
            return json_dict
        else:
            return json.dumps(json_dict)

    def from_json(self, json_str: str | dict):
        if isinstance(json_str, (str, bytes)):
            json_dict = json.loads(json_str)
        elif isinstance(json_str, dict):
            json_dict = json_str
        else:
            raise TypeError(f'Invalid type {type(json_str)}, expect [str, bytes, dict]')

        super().from_json(json_dict)

        self.limit_price = json_dict['limit_price']
        self.limit_adjust_factor = json_dict['limit_adjust_factor']
        self.limit_adjust_level = json_dict['limit_adjust_level']
        self.limit_mode = json_dict['limit_mode']

        return self


class PassiveTimeout(Passive):
    """ Passive handler with timeout function
    PassiveTimeout is similar to Passive, with a timeout value (in seconds) and cancel working order after that

    Default timeout is 0, which is no timeout (same as passive).
    """

    def __init__(self, **kwargs):
        self.timeout = kwargs.pop('timeout', 0)

        super().__init__(**kwargs)

    def on_market_data(self, market_data: MarketData, **kwargs):
        if self.is_active:
            self.work()

    def work(self):
        ts = self.algo_engine.mds.trade_time_between(start_time=self.ts_started, end_time=self.timestamp).total_seconds()
        if self.status == self.Status.working and self.timeout and ts > self.timeout:
            self.cancel()
            self.logger.debug(f'{self} canceling. status={self.status}, ts={ts:.3f}s')
        else:
            self.logger.debug(f'{self} working. status={self.status}, ts={ts:.3f}s, timeout={self.timeout:.3f}s')

    def to_json(self, fmt='str') -> str | dict:
        json_dict = super().to_json(fmt='dict')

        additional_dict = dict(
            timeout=self.timeout
        )

        json_dict.update(additional_dict)

        if fmt == 'dict':
            return json_dict
        else:
            return json.dumps(json_dict)

    def from_json(self, json_str: str | dict):
        if isinstance(json_str, (str, bytes)):
            json_dict = json.loads(json_str)
        elif isinstance(json_str, dict):
            json_dict = json_str
        else:
            raise TypeError(f'Invalid type {type(json_str)}, expect [str, bytes, dict]')

        super().from_json(json_dict)

        self.timeout = json_dict['timeout']

        return self


class Aggressive(Passive):
    """ Aggressive trading algorithm
    Aggressive is similar as Passive.
    Aggressive will re-launch a "fixing" order immediately
    after working order got canceled or filled, if there is any un-filled volume.

    USE WITH CAUTION
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _filled(self, order: TradeInstruction, report: TradeReport, **kwargs):
        super()._filled(order=order, report=report, **kwargs)

        if not self.is_active:
            self._assign_status(status=self.Status.done)
        elif order.order_id not in self.working_order:
            if self.status == self.Status.working:
                self.launch()

    def _canceled(self, order: TradeInstruction, **kwargs):
        super()._canceled(order=order, **kwargs)

        if not self.is_active:
            self._assign_status(status=self.Status.done)
        elif order.order_id not in self.working_order:
            if self.status == self.Status.working:
                self.launch()


class AggressiveTimeout(PassiveTimeout, Aggressive):
    """ Similar to PassiveTimeout, AggressiveTimeout cancel working order after timeout and re-launch "fixing" order after canceled or filled.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _filled(self, order: TradeInstruction, report: TradeReport, **kwargs):
        return Aggressive._filled(self=self, order=order, report=report, **kwargs)

    def _canceled(self, order: TradeInstruction, **kwargs):
        return Aggressive._canceled(self=self, order=order, **kwargs)


class AlgoEngine(object, metaclass=Singleton):
    def __init__(self, mds=None, registry=None):
        self.mds = mds if mds is not None else MDS
        self.registry = registry if registry is not None else ALGO_REGISTRY

    @classmethod
    def _compare_price(cls, side: TransactionSide, limit_price: float = None, original_limit: float = None, mode='strict') -> float:
        calculated_limit = original_limit

        if limit_price is None:
            return calculated_limit
        elif calculated_limit is None:
            return limit_price
        if mode is None or mode == 'strict':
            if side.sign > 0:
                calculated_limit = min(calculated_limit, limit_price)
            else:
                calculated_limit = max(calculated_limit, limit_price)
        elif mode == 'loose':
            if side.sign > 0:
                calculated_limit = max(calculated_limit, limit_price)
            else:
                calculated_limit = min(calculated_limit, limit_price)
        else:
            LOGGER.error(f'Invalid compare mode {mode}!')
            return limit_price

        return calculated_limit

    def get_algo(self, name: str):
        algo = self.registry.to_algo(name=name.lower(), algo_engine=self)
        return algo

    def calculate_limit(
            self,
            algo: AlgoTemplate,
            limit_price: float = None,
            limit_adjust_factor: float = None,
            limit_adjust_level: float = None,
            mode: str = 'loose'
    ) -> float | None:
        """Calculate limit price

        :param algo: given algo
        :param limit_price: absolute limit_price
        :param limit_adjust_factor: limit_price = market_price * (1 + factor) for long order else limit price = market_price * (1 - factor) for short order
        :param limit_adjust_level:  for long order, limit price = bid[lvl] if lvl > 0 else ask[lvl] for lvl < 0.
        :param mode: "strict" to select strictest limit price or "loose" to select loosest one. Default is None, which is "strict".
        :return: the calculated limit price, if there is any
        """
        ticker = algo.ticker
        side = algo.side
        market_price = self.mds.market_price.get(ticker)

        # validate side
        if side.sign == 0:
            LOGGER.error(f'Invalid side {side}')
            return None

        # market data not available
        if market_price is None:
            LOGGER.error(f'{ticker} market data not available')
            return None

        calculated_limit: float | None = None
        limit_abs = None
        limit_adj = None
        limit_lvl = None

        # compare with absolute limit_price
        if limit_price is not None:
            limit_abs = limit_price

        if limit_adjust_factor is not None:
            limit_adj = market_price * (1 + limit_adjust_factor * side.sign)

        if limit_adjust_level is not None:
            order_book = self.mds.get_order_book(ticker=ticker)

            if order_book is not None:
                lvl = abs(limit_adjust_level)

                if limit_adjust_level > 0:
                    if side.sign > 0:
                        book = order_book.bid.price
                    else:
                        book = order_book.ask.price

                    limit_lvl = book[min(lvl, len(book) - 1)]
                elif limit_adjust_level < 0:
                    if side.sign > 0:
                        book = order_book.ask.price
                    else:
                        book = order_book.bid.price

                    limit_lvl = book[min(lvl, len(book) - 1)]

        calculated_limit = self._compare_price(limit_price=limit_abs, original_limit=calculated_limit, side=side, mode=mode)
        calculated_limit = self._compare_price(limit_price=limit_adj, original_limit=calculated_limit, side=side, mode=mode)
        calculated_limit = self._compare_price(limit_price=limit_lvl, original_limit=calculated_limit, side=side, mode=mode)
        calculated_limit = self._compare_price(limit_price=market_price, original_limit=calculated_limit, side=side, mode=mode)

        LOGGER.info(f'BBA limits {ticker} market_price={market_price}, lmt_abs={limit_price}, lmt_adj={limit_adj}, lmt_lvl={limit_lvl}, mode={mode}, cal_lmt={calculated_limit}')
        return calculated_limit

    def from_json(self, json_str, dma) -> AlgoTemplate:
        if isinstance(json_str, (str, bytes)):
            json_dict = json.loads(json_str)
        elif isinstance(json_str, dict):
            json_dict = json_str
        else:
            raise TypeError(f'Invalid type {type(json_str)}, expect [str, bytes, dict]')

        algo: AlgoTemplate = self.get_algo(json_dict['algo_type'])(
            ticker=json_dict['ticker'],
            side=TransactionSide(json_dict['side']),
            target_volume=json_dict['target_volume'],
            dma=dma,
            algo_id=json_dict['algo_id']
        )
        algo.from_json(json_dict)

        return algo


class AlgoRegistry(object, metaclass=Singleton):
    """
    registry for trade algos

    to add a new algo, add name to __init__ method, add handler to .cast() method

    DO NOT add any other value to __init__.
    """

    def __init__(self):
        self.alias = {}
        self.registry = {}

        # pre-defined algo name for easy access
        self.aggressive = 'aggressive'
        self.passive = 'passive'
        self.aggressive_timeout = 'aggressive_timeout'
        self.passive_timeout = 'passive_timeout'
        self.limit_range = 'limit_range'

    def add_algo(self, name: str, *alias, handler: Type[AlgoTemplate]):
        self.registry[name] = handler

        for _alias in alias:
            self.alias[_alias] = name

    def cast(self, value: str):
        name = value.lower()

        # check alias
        if name in self.alias:
            name = self.alias[name]

        # init from storage
        if name in self.registry:
            return self.registry[name]
        else:
            raise ValueError(f'Invalid name {value}')

    @property
    def reversed_registry(self) -> dict[str, str]:
        reversed_registry = {algo.__name__: name for name, algo in self.registry.items()}
        return reversed_registry

    def to_algo(self, name: str, algo_engine: AlgoEngine = None):
        if algo_engine is None:
            algo_engine = ALGO_ENGINE

        algo = self.registry.get(name.lower())
        return functools.partial(algo, algo_engine=algo_engine)


ALGO_REGISTRY = AlgoRegistry()

ALGO_REGISTRY.add_algo('aggressive', 'aggr', handler=Aggressive)
ALGO_REGISTRY.add_algo('passive', 'pass', handler=Passive)
ALGO_REGISTRY.add_algo('aggressive_timeout', 'aggr_timeout', handler=AggressiveTimeout)
ALGO_REGISTRY.add_algo('passive_timeout', 'pass_timeout', handler=PassiveTimeout)

ALGO_ENGINE = AlgoEngine(mds=MDS, registry=ALGO_REGISTRY)
