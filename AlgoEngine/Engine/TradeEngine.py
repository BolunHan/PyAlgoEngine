import abc
import datetime
import json
import os
import pathlib
import queue
import threading
import time
import uuid
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from PyQuantKit import TransactionSide, TradeInstruction, OrderType, MarketData, BarData, TradeData, TickData, OrderState, OrderBook, TradeReport

from . import LOGGER
from .AlgoEngine import ALGO_ENGINE, AlgoTemplate
from .EventEngine import TOPIC, EVENT_ENGINE
from .MarketEngine import MarketDataService

LOGGER = LOGGER.getChild('TradeEngine')
__all__ = ['DirectMarketAccess', 'Balance', 'TradeHandler', 'TradePos', 'PositionTracker', 'Inventory', 'RiskProfile', 'SimMatch']


class NameSpace(dict):
    def __init__(self, name: str = None, **kwargs):
        self.name = name
        super().__init__(**kwargs)

    def __getattr__(self, entry):
        if entry in self:
            return self[entry]

        raise KeyError(f'Entry {entry} not exist!')

    def __repr__(self):
        if self.name:
            repr_str = f'<{self.name}>'
        else:
            repr_str = f'<NameSpace>'

        repr_str += f'({super().__repr__()})'
        return repr_str

    def unpack(self):
        return list(self.values())


class DirectMarketAccess(object, metaclass=abc.ABCMeta):
    """
    order buff designed to process order and control risk
    """

    def __init__(self, mds: MarketDataService, risk_profile: 'RiskProfile', cool_down: float = None):
        self.mds = mds
        self.risk_profile = risk_profile
        self.cool_down = cool_down

        self.order_queue = queue.Queue()
        self.worker = threading.Thread(target=self._run)
        self._is_done = False

    def __repr__(self):
        return f'<OrderHandler>(cd={self.cool_down}, id={id(self)})'

    @abc.abstractmethod
    def _launch_order_handler(self, order: TradeInstruction, **kwargs):
        ...

    @abc.abstractmethod
    def _cancel_order_handler(self, order: TradeInstruction, **kwargs):
        ...

    @abc.abstractmethod
    def _reject_order_handler(self, order: TradeInstruction, **kwargs):
        ...

    def trade_time_between(self, start_time: Union[datetime.datetime, float], end_time: Union[datetime.datetime, float], **kwargs) -> datetime.timedelta:
        return self.mds.trade_time_between(start_time, end_time, **kwargs)

    def _launch_order_buffed(self, order: TradeInstruction, **kwargs):
        self.order_queue.put(('launch', order, kwargs))

    def _cancel_order_buffed(self, order: TradeInstruction, **kwargs):
        self.order_queue.put(('cancel', order, kwargs))

    def _launch_order_no_wait(self, order: TradeInstruction, **kwargs):
        LOGGER.info(f'{self} sent a LAUNCH signal of {order}')
        is_pass = self.risk_profile.check(order=order)

        if not is_pass:
            LOGGER.warning(f'{order} Rejected by risk control! Invalid action {order.ticker} {order.side.name} {order.volume}!')
            order.set_order_state(order_state=OrderState.Rejected)
            self._reject_order_handler(order=order, **kwargs)
            return

        order.set_order_state(order_state=OrderState.Sent)
        self._launch_order_handler(order=order, **kwargs)

    def _cancel_order_no_wait(self, order: TradeInstruction, **kwargs):
        LOGGER.info(f'{self} sent a CANCEL signal of {order}')

        order.set_order_state(order_state=OrderState.Canceling)
        self._cancel_order_handler(order=order, **kwargs)

    def launch_order(self, order: TradeInstruction, **kwargs):
        LOGGER.info(f'{self} launching order {order}')
        if self.cool_down:
            self._launch_order_buffed(order=order, **kwargs)
        else:
            self._launch_order_no_wait(order=order, **kwargs)

    def cancel_order(self, order: TradeInstruction, **kwargs):
        LOGGER.info(f'{self} canceling order {order}')
        if self.cool_down:
            self._cancel_order_buffed(order=order, **kwargs)
        else:
            self._cancel_order_no_wait(order=order, **kwargs)

    def _process_order(self):
        try:
            flag, order, kwargs = self.order_queue.get(block=False)

            if flag == 'launch':
                self._launch_order_no_wait(order=order, **kwargs)
            elif flag == 'cancel':
                self._cancel_order_no_wait(order=order, **kwargs)
            else:
                LOGGER.info(f'Invalid order type {flag}')
        except queue.Empty:
            pass

    def _run(self):
        while True:
            self._process_order()

            time.sleep(self.cool_down)

            if self._is_done:
                break
        LOGGER.info('DMA successfully shutdown!')

    def run(self):
        self.worker.run()

    @property
    def is_done(self):
        return self._is_done

    def shut_down(self):
        self._is_done = True
        LOGGER.info(f'Order buff shutting down!')

    @property
    def market_price(self):
        return self.mds.market_price

    @property
    def market_time(self):
        return self.mds.market_time


class TradeHandler(object):
    """
    TradeHandler handles trades for a single given ticker. It requires 3 parameter:
    - ticker: the given ticker
    - side: trade side
    - target_volume: the target total volume, must be positive

    TradeHandler provides various trading algo:
    - Aggressive: to ignore sync_progress, list all volume and reopen if got canceled.
    - Passive: to ignore sync_progress, list all volume and stop if got canceled.
    - TWAP: to follow sync_progress and place order by twap <NotImplemented>
    - VWAP: to follow sync_progress and place order by vwap <NotImplemented>
    - Iceberg: try to follow sync_progress and place order by iceberg / bbo <NotImplemented>
    ...
    """

    class _TradeHandlerStatus(Enum):
        idle = 'idle'
        winding = 'winding'
        positioning = 'positioning'
        unwinding = 'unwinding'
        closed = 'closed'
        error = 'error'

    Status = _TradeHandlerStatus

    def __init__(
            self,
            ticker: str,
            side: TransactionSide,
            target_volume: float,
            dma: DirectMarketAccess,
            default_algo: str = None,
            logger=None,
            trade_id: str = None,
    ):
        self.ticker = ticker
        self.side = side
        self.target_volume = target_volume
        self.dma = dma
        self.algo_registry = ALGO_ENGINE.registry
        self.default_algo = self.algo_registry.passive if default_algo is None else default_algo
        self.logger = LOGGER if logger is None else logger
        self.trade_id = f'{self.__class__.__name__}.{self.ticker}.{self.side.side_name}.{uuid.uuid4().hex}' if trade_id is None else trade_id

        self.status = self.Status.idle
        self._open_time = None
        self._unwind_time = None
        self._closed_time = None

        self.working_algo: Dict[str, AlgoTemplate] = {}
        self.open_algo: Dict[str, AlgoTemplate] = {}
        self.unwind_algo: Dict[str, AlgoTemplate] = {}

    def __call__(
            self,
            filled: List[TradeReport] = None,
            canceled: List[str] = None,
            market_data: List[MarketData] = None,
            progress: float = None
    ):
        if filled:
            for _ in filled:
                self.on_fill(report=_)

        if canceled:
            for _ in canceled:
                self.on_cancel(order_id=_)

        if market_data:
            for _ in market_data:
                self.on_market_data(market_data=_)

        if progress:
            self.on_sync_progress(progress=progress)

    def __repr__(self):
        return f'<GroupTradeHandler>({self.ticker}.{self.status.name}.{self.side.side_name}.{id(self)})'

    def _update_status(self, status: Status = None):
        for algo_id in list(self.working_algo):
            algo = self.working_algo.get(algo_id, None)

            if algo is not None:
                if algo.status in [algo.Status.done, algo.Status.closed]:
                    self.working_algo.pop(algo_id, None)

        if status is not None:
            self.status = status
        else:
            if any([algo.status in [algo.Status.rejected, algo.Status.error] for algo in list(self.working_algo.values())]):
                self.status = self.Status.error
            elif self.status == self.Status.idle:
                if not self.working_algo:
                    if self.exposure_volume:
                        self.status = self.Status.positioning
                    else:
                        self.status = self.Status.closed

                        if self._closed_time is None:
                            self._closed_time = self.market_time
                elif self.open_algo:
                    self.status = self.Status.winding
            elif self.status == self.Status.winding:
                if not self.working_algo:
                    if self.exposure_volume:
                        self.status = self.Status.positioning
                    else:
                        self.status = self.Status.closed

                        if self._closed_time is None:
                            self._closed_time = self.market_time
            elif self.status == self.Status.positioning:
                if not self.working_algo:
                    if self.exposure_volume:
                        self.status = self.Status.positioning
                    else:
                        self.status = self.Status.closed

                        if self._closed_time is None:
                            self._closed_time = self.market_time
                elif self.unwind_algo:
                    self.status = self.Status.unwinding
            elif self.status == self.Status.unwinding:
                if not self.working_algo:
                    if self.exposure_volume:
                        self.status = self.Status.positioning
                    else:
                        self.status = self.Status.closed

                        if self._closed_time is None:
                            self._closed_time = self.market_time

    def launch_order(self, order: TradeInstruction, **kwargs):
        self.dma.launch_order(order=order, **kwargs)
        self._update_status()

    def cancel_order(self, order: TradeInstruction, **kwargs):
        order.set_order_state(order_state=OrderState.Canceling)
        self.dma.cancel_order(order=order, **kwargs)
        self._update_status()

    def on_fill(self, report: TradeReport, **kwargs):
        for algo_id in list(self.working_algo):
            algo = self.working_algo.get(algo_id)

            if algo is None:
                continue

            if report.order_id in algo.working_order:
                _ = algo.on_fill(report=report, **kwargs)
                self._update_status()
                return _

        return 0

    def on_cancel(self, order_id: Optional[str] = None, **kwargs):
        for algo_id in list(self.working_algo):
            algo = self.working_algo.get(algo_id)

            if algo is None:
                continue

            if order_id in algo.working_order:
                _ = algo.on_cancel(order_id=order_id, **kwargs)
                self._update_status()
                return _
        return 0

    def on_reject(self, order: TradeInstruction, **kwargs):
        for algo_id in list(self.working_algo):
            algo = self.working_algo.get(algo_id)

            if algo is None:
                continue

            if order.order_id in algo.working_order:
                _ = algo.on_reject(order=order, **kwargs)

                if algo.algo_id in self.unwind_algo and self.exposure_volume:
                    LOGGER.error(f'{self} unwinding order rejected! {self.exposure_volume} outstanding, Manual intervention required!')
                    algo.status = algo.Status.error

                self._update_status()
                return _
        return 0

    def on_market_data(self, market_data: MarketData, **kwargs):
        for algo_id in list(self.working_algo):
            algo = self.working_algo.get(algo_id)

            if algo is None:
                continue

            algo.on_market_data(market_data=market_data, **kwargs)

    def on_sync_progress(self, progress: float, **kwargs):
        for algo_id in list(self.working_algo):
            algo = self.working_algo.get(algo_id)

            if algo is None:
                continue

            algo.on_sync_progress(progress=progress, **kwargs)

    def open(self, algo: str = None, target_volume: float = None, **kwargs):
        if algo is None:
            algo = self.default_algo

        if target_volume is None:
            target_volume = max(0., self.target_volume - abs(self.exposure_volume) - abs(self.working_volume_open))

        if target_volume:
            algo_handler = self.algo_registry.to_algo(name=algo)(
                handler=self,
                ticker=self.ticker,
                side=self.side,
                target_volume=target_volume,
                dma=self.dma,
                **kwargs
            )

            self.logger.debug(f'{algo_handler} opening {self.ticker} {self.side} position!')
            self.working_algo[algo_handler.algo_id] = algo_handler
            self.open_algo[algo_handler.algo_id] = algo_handler

            if self._open_time is None:
                self._open_time = self.market_time

            algo_handler.launch(**kwargs)
            self._update_status()

    def unwind(self, algo: str = None, target_volume: float = None, **kwargs):
        if algo is None:
            algo = self.default_algo

        if self.working_volume_open:
            self.cancel()

        if target_volume is None:
            target_volume = abs(self.exposure_volume) - abs(self.working_volume_unwind)

            if target_volume < 0:
                LOGGER.error(f'Invalid trade_handler {self}, unwinding {self.working_volume_unwind}, exposed {abs(self.exposure_volume)}, diff={target_volume}')
                self.cancel()
                target_volume = 0.

        if target_volume:
            algo_handler = self.algo_registry.to_algo(name=algo)(
                handler=self,
                ticker=self.ticker,
                side=-self.side,
                target_volume=target_volume,
                dma=self.dma,
                **kwargs
            )

            self.logger.debug(f'{algo_handler} unwinding {self.ticker} {self.side} position!')
            self.working_algo[algo_handler.algo_id] = algo_handler
            self.unwind_algo[algo_handler.algo_id] = algo_handler

            if self._unwind_time is None:
                self._unwind_time = self.dma.market_time

            algo_handler.launch(**kwargs)
            self._update_status()

    def cancel(self, **kwargs):
        for algo_id in list(self.working_algo):
            algo = self.working_algo.get(algo_id)

            if algo is None:
                continue

            self.logger.debug(f'Canceling {algo.ticker} {algo.side} order!')
            algo.cancel(**kwargs)

    def recover(self):
        for algo_id in list(self.working_algo):
            algo = self.working_algo.get(algo_id)

            if algo is None:
                continue

            algo.recover()

            if algo.status == algo.Status.closed:
                self.working_algo.pop(algo_id, None)

        if self.working_volume:
            status = None

            for algo_id in list(self.working_algo):
                if algo_id in self.open_algo:
                    _status = self.Status.winding
                elif algo_id in self.unwind_algo:
                    _status = self.Status.unwinding
                else:
                    break

                if status is None:
                    status = _status
                elif status != _status:
                    status = None
                    break

            if status is not None:
                self._update_status(status=status)
            else:
                LOGGER.info(f'{self} failed to recover from error')
                return
        else:
            if self.exposure_volume:
                self._update_status(status=self.Status.positioning)
            else:
                self._update_status(status=self.Status.closed)

        LOGGER.info(f'{self} recovery successful! status {self.status}')

    def average_price(self, flag: str):
        adjust_volume = 0.
        notional = 0.
        trades = {}

        if flag in ['open', 'wind']:
            handlers = self.open_algo
        elif flag in ['close', 'unwind']:
            handlers = self.unwind_algo
        else:
            raise ValueError(f'Invalid flag {flag}')

        for handler in list(handlers.values()):
            trades.update(handler.trades)

        for report in list(trades.values()):
            if report.price == 0:
                adjust_volume += report.volume
            else:
                adjust_volume += report.notional / report.price
            notional += report.notional

        if adjust_volume == 0:
            return np.nan
        else:
            return notional / adjust_volume

    def to_json(self, fmt='str') -> Union[str, dict]:
        json_dict = {
            'ticker': self.ticker,
            'side': self.side.name,
            'target_volume': self.target_volume,
            'default_algo': self.default_algo,
            'trade_id': self.trade_id,
            'status': self.status.name,
            'open_time': datetime.datetime.timestamp(self._open_time) if self._open_time else None,
            'unwind_time': datetime.datetime.timestamp(self._unwind_time) if self._unwind_time else None,
            'closed_time': datetime.datetime.timestamp(self._closed_time) if self._closed_time else None,
            'open_algo': {_: self.open_algo[_].to_json(fmt='dict') for _ in self.open_algo},
            'unwind_algo': {_: self.unwind_algo[_].to_json(fmt='dict') for _ in self.unwind_algo},
        }

        if fmt == 'dict':
            return json_dict
        else:
            return json.dumps(json_dict)

    def from_json(self, json_str: Union[str, dict]):
        if isinstance(json_str, (str, bytes)):
            json_dict = json.loads(json_str)
        elif isinstance(json_str, dict):
            json_dict = json_str
        else:
            raise TypeError(f'Invalid type {type(json_str)}, expect [str, bytes, dict]')

        self.ticker = json_dict['ticker']
        self.side = TransactionSide(json_dict['side'])
        self.target_volume = json_dict['target_volume']
        self.default_algo = json_dict['default_algo']
        self.status = self.Status[json_dict['status']]
        self.trade_id = json_dict['trade_id']
        self._open_time = None if json_dict['open_time'] is None else datetime.datetime.fromtimestamp(json_dict['open_time'])
        self._unwind_time = None if json_dict['unwind_time'] is None else datetime.datetime.fromtimestamp(json_dict['unwind_time'])
        self._closed_time = None if json_dict['closed_time'] is None else datetime.datetime.fromtimestamp(json_dict['closed_time'])

        for algo_id in json_dict['open_algo']:
            algo = ALGO_ENGINE.from_json(json_dict['open_algo'][algo_id], handler=self)

            if algo.status == algo.Status.working:
                self.working_algo[algo_id] = algo
            self.open_algo[algo_id] = algo

        for algo_id in json_dict['unwind_algo']:
            algo = ALGO_ENGINE.from_json(json_dict['unwind_algo'][algo_id], handler=self)

            if algo.status == algo.Status.working:
                self.working_algo[algo_id] = algo
            self.unwind_algo[algo_id] = algo

        return self

    @property
    def open_time(self) -> Optional[datetime.datetime]:
        return self._open_time

    @property
    def close_time(self) -> Optional[datetime.datetime]:
        return self._closed_time

    @property
    def working_order(self) -> Dict[str, TradeInstruction]:
        working_order = {}
        for algo_id in list(self.working_algo):
            algo = self.working_algo.get(algo_id)

            if algo is None:
                continue

            working_order.update(algo.working_order)
        return working_order

    @property
    def exposure_volume(self) -> float:
        """
        a float with sign indicating exposure (filled) volume
        """
        exposure = 0.

        for handler in list(self.open_algo.values()):
            exposure += handler.exposure_volume

        for handler in list(self.unwind_algo.values()):
            exposure += handler.exposure_volume

        return exposure

    @property
    def working_volume(self) -> float:
        """
        a positive float indicating working (listed but not filled) volume of the orders
        """
        return self.working_volume_open + self.working_volume_unwind

    @property
    def working_volume_open(self) -> float:
        """
        a positive float indicating working (listed but not filled) volume of the OPEN handlers
        """
        working_volume = 0.
        for handler in list(self.open_algo.values()):
            working_volume += handler.working_volume
        return working_volume

    @property
    def working_volume_unwind(self) -> float:
        """
        a positive float indicating working (listed but not filled) volume of the UNWINDING handlers
        """
        working_volume = 0.
        for handler in list(self.unwind_algo.values()):
            working_volume += handler.working_volume
        return working_volume

    @property
    def target_volume_open(self) -> float:
        """
        a positive float indicating target volume of the OPEN handlers
        """
        target_volume = 0.
        for handler in list(self.open_algo.values()):
            target_volume += handler.target_volume
        return target_volume

    @property
    def target_volume_unwind(self) -> float:
        """
        a positive float indicating target volume of the UNWINDING handlers
        """
        target_volume = 0.
        for handler in list(self.unwind_algo.values()):
            target_volume += handler.target_volume
        return target_volume

    @property
    def inventory_volume(self) -> float:
        """
        a positive float indicating exposure which has not been unwinding
        """
        ttl_inv = abs(self.exposure_volume)
        target_unwind = self.target_volume_unwind
        return ttl_inv - target_unwind

    @property
    def open_price(self) -> float:
        return self.average_price(flag='open')

    @property
    def close_price(self) -> float:
        return self.average_price(flag='unwind')

    @property
    def open_notional(self) -> float:
        notional = 0.
        for handler in list(self.open_algo.values()):
            notional += handler.filled_notional
        return notional

    @property
    def close_notional(self) -> float:
        notional = 0.
        for handler in list(self.unwind_algo.values()):
            notional += handler.filled_notional
        return notional

    @property
    def holding_cost(self):
        if self.exposure_volume:
            return (self.open_notional - self.close_notional) / self.exposure_volume
        else:
            return 0

    @property
    def open_volume(self) -> float:
        volume = 0.
        for handler in list(self.open_algo.values()):
            volume += handler.filled_volume
        return volume

    @property
    def close_volume(self) -> float:
        volume = 0.
        for handler in list(self.unwind_algo.values()):
            volume += handler.filled_volume
        return volume

    @property
    def cash_flow(self) -> float:
        cash_flow = 0.

        for handler in list(self.open_algo.values()):
            cash_flow += handler.cash_flow

        for handler in list(self.unwind_algo.values()):
            cash_flow += handler.cash_flow

        return cash_flow

    @property
    def fee(self) -> float:
        fee = 0.

        for handler in list(self.open_algo.values()):
            fee += handler.fee

        for handler in list(self.unwind_algo.values()):
            fee += handler.fee

        return fee

    @property
    def algo(self) -> Dict[str, AlgoTemplate]:
        algo = {}

        algo.update(self.open_algo)
        algo.update(self.unwind_algo)

        return algo

    @property
    def orders(self):
        orders = {}

        for handler in list(self.algo.values()):
            orders.update(handler.order)

        return orders

    @property
    def trades(self) -> Dict[str, TradeReport]:
        trades = {}

        for handler in list(self.algo.values()):
            trades.update(handler.trades)

        return trades

    @property
    def market_price(self):
        return self.dma.market_price.get(self.ticker)

    @property
    def market_time(self):
        return self.dma.market_time

    def pnl(self, pct=False):
        if self.exposure_volume:
            if self.market_price is not None:
                pnl = self.market_price * self.exposure_volume * self.multiplier + self.cash_flow
            else:
                pnl = np.nan
        else:
            pnl = self.cash_flow

        if pct:
            if self.open_notional:
                pnl = pnl / self.open_notional
            else:
                pnl = np.inf * np.sign(pnl)

        return pnl

    @property
    def multiplier(self) -> float:
        if self.open_algo:
            return list(self.open_algo.values())[0].multiplier
        else:
            return 1

    @property
    def notional(self):
        LOGGER.warning(DeprecationWarning('use .open_notional instead!'))
        return self.open_notional


class TradePos(object):
    """
    TradePos handles all TradeHandler for a given signal.
    """

    class _TradePosStatus(Enum):
        idle = 'idle'
        winding = 'winding'
        positioning = 'positioning'
        unwinding = 'unwinding'
        closed = 'closed'
        error = 'error'
        canceling = 'canceling'  # deprecated!

    Status = _TradePosStatus

    def __init__(
            self,
            dma: DirectMarketAccess,
            logger=None,
            pos_id: str = None,
            **kwargs
    ):
        self.dma = dma
        self.logger = LOGGER if logger is None else logger
        self.pos_id = f'{self.__class__.__name__}.{uuid.uuid4().hex}' if pos_id is None else pos_id
        self.__dict__.update(kwargs)

        self.status = self.Status.idle
        self.trade_handler: Dict[str, TradeHandler] = {}

        self.open_time = None
        self.close_time = None
        self.max_gain = 0.
        self.max_loss = 0.

    def __repr__(self):
        tickers = self.tickers

        if tickers:
            return f'<TradePos>({ {_: self.exposure_volume.get(_, 0) for _ in self.tickers} }.{self.status.name}.{id(self)})'
        else:
            return f'<TradePos>({self.status.name}.{id(self)})'

    def __iter__(self):
        return self.trade_handler.__iter__()

    def __getitem__(self, entry) -> TradeHandler:
        return self.get(entry)

    def get(self, entry) -> Optional[Union[TradeHandler, List[TradeHandler]]]:
        if entry in self.trade_handler:
            return self.trade_handler[entry]

        handlers = []

        for handler_id in list(self.trade_handler):
            trade_handler = self.trade_handler.get(handler_id)

            if trade_handler is None:
                continue

            if trade_handler.ticker == entry:
                handlers.append(trade_handler)

        if len(handlers) == 1:
            return handlers[0]
        elif not handlers:
            raise KeyError(f'Entry {entry} not found!')
        else:
            raise KeyError(f'multiple Entry {entry} TradeHandler found in {self}')

    def add_target(self, ticker: str, side: TransactionSide, target_volume: float, **kwargs):
        trade_handler = TradeHandler(
            ticker=ticker,
            side=side,
            target_volume=target_volume,
            dma=self.dma,
            default_algo=kwargs.pop('algo', None),
            logger=self.logger,
            trade_id=kwargs.pop('trade_id', None)
        )

        self.trade_handler[trade_handler.trade_id] = trade_handler
        return trade_handler.trade_id

    def open(self, additional_kwargs: Dict[str, dict] = None, **kwargs):
        if additional_kwargs is None:
            additional_kwargs = {}

        additional_kwargs.update(kwargs)

        for trade_id in self.trade_handler:
            trade_handler = self.trade_handler[trade_id]

            kwargs = additional_kwargs.get(trade_id, None)

            if kwargs is None:
                LOGGER.warning(f'kwargs for TradePos.open does not specify a trade_id, use ticker {trade_handler.ticker} to index.')
                kwargs = additional_kwargs.get(trade_handler.ticker, {})

            trade_handler.open(**kwargs)
            self.open_time = self.market_time

        self._update_status()

    def unwind(self, trade_id: str, additional_kwargs: Dict[str, dict] = None, **kwargs):
        if additional_kwargs is None:
            additional_kwargs = {}

        additional_kwargs.update(kwargs)
        trade_handler = self.trade_handler[trade_id]
        trade_handler.unwind(**additional_kwargs)
        self.close_time = self.market_time

        self._update_status()

    def unwind_all(self, additional_kwargs: Dict[str, dict] = None, **kwargs):
        if additional_kwargs is None:
            additional_kwargs = {}

        additional_kwargs.update(kwargs)

        for trade_id in self.trade_handler:
            trade_handler = self.trade_handler[trade_id]
            ticker = trade_handler.ticker

            if trade_id in additional_kwargs:
                kwargs = additional_kwargs.get(trade_id, {})
            elif ticker in additional_kwargs:
                LOGGER.warning(f'kwargs for TradePos.unwind does not specify a trade_id, use ticker {trade_handler.ticker} to index.')
                kwargs = additional_kwargs.get(ticker, {})
            else:
                kwargs = {}

            trade_handler.unwind(**kwargs)
            self.close_time = self.market_time

        self._update_status()

    def unwind_auto(self, ticker: str, side: TransactionSide, volume: float, additional_kwargs: Dict[str, dict] = None, **kwargs) -> float:
        """
        auto distribute unwind instruction to trade_handler and return undistributed volume
        """
        to_unwind = abs(volume)

        for _trade_id in list(self.trade_handler):
            trade_handler = self.trade_handler.get(_trade_id)
            _additional_kwargs = {}

            if trade_handler is None:
                continue

            if trade_handler.ticker != ticker:
                continue

            if side.sign == trade_handler.side.sign:
                LOGGER.debug(f'Invalid TransactionSide of the Unwind Instruction! Expect {-trade_handler.side}, got {side.sign}. Error ignored and execute as unwind!')
                continue
            elif not to_unwind:
                break

            LOGGER.debug(f'Close signal have no trade_id, unwinding TradeHandler {_trade_id}!')
            handler_inventory = trade_handler.inventory_volume

            # _additional_kwargs['default_algo'] = entry.default_algo
            # _additional_kwargs['limit_price'] = entry.limit

            if additional_kwargs is not None:
                if isinstance(additional_kwargs, dict):
                    _additional_kwargs.update(additional_kwargs)
                else:
                    raise ValueError(f'Invalid additional_kwargs {additional_kwargs}')

            # unwind opposite side
            if handler_inventory > 0:
                if to_unwind <= handler_inventory:
                    _additional_kwargs['target_volume'] = to_unwind
                    to_unwind = 0.
                else:
                    _additional_kwargs['target_volume'] = handler_inventory
                    to_unwind -= handler_inventory
            else:
                continue

            trade_handler.unwind(**_additional_kwargs, **kwargs)
            self.close_time = self.market_time
            self._update_status()

        return to_unwind

    def cancel(self, trade_id: str, additional_kwargs: Dict[str, dict] = None, **kwargs):
        if additional_kwargs is None:
            additional_kwargs = {}

        additional_kwargs.update(kwargs)
        trade_handler = self.trade_handler[trade_id]
        trade_handler.cancel(**kwargs)

        self._update_status()

    def cancel_all(self, additional_kwargs: Dict[str, dict] = None, **kwargs):
        if additional_kwargs is None:
            additional_kwargs = {}

        additional_kwargs.update(kwargs)

        for trade_id in self.trade_handler:
            trade_handler = self.trade_handler[trade_id]
            ticker = trade_handler.ticker

            if trade_id in additional_kwargs:
                kwargs = additional_kwargs.get(trade_id, {})
            elif ticker in additional_kwargs:
                LOGGER.warning(f'kwargs for TradePos.cancel does not specify a trade_id, use ticker {ticker} to index.')
                kwargs = additional_kwargs.get(ticker, {})
            else:
                kwargs = {}

            trade_handler.cancel(**kwargs)

        self._update_status()

    def sync_progress(self, step: float = None, progress: float = None):
        if progress and step:
            LOGGER.warning('Can not sync trade_handler with both progress and step')
        elif progress:
            pass
        elif step:
            sync_progress = max([algo.target_progress for trade_handler in self.trade_handler.values() for algo in trade_handler.algo.values()])
            progress = sync_progress + step
        else:
            progress = max([algo.filled_progress for trade_handler in self.trade_handler.values() for algo in trade_handler.algo.values()])

        for trade_id in list(self.trade_handler):
            trade_handler = self.trade_handler.get(trade_id)

            if trade_handler is None:
                continue

            trade_handler.on_sync_progress(progress=progress)

    def on_market_data(self, market_data: MarketData):
        for trade_id in list(self.trade_handler):
            trade_handler = self.trade_handler.get(trade_id)

            if trade_handler is None:
                continue

            trade_handler.on_market_data(market_data=market_data)

    def on_fill(self, report: TradeReport, **kwargs):
        order_id = report.order_id
        trade_handler = self.reversed_order_mapping.get(order_id)

        if trade_handler is None:
            return 0

        result = trade_handler.on_fill(report=report, **kwargs)
        self._update_status()
        return result

    def on_cancel(self, order_id: str, **kwargs):
        trade_handler = self.reversed_order_mapping.get(order_id)

        if trade_handler is None:
            return 0

        result = trade_handler.on_cancel(order_id=order_id, **kwargs)
        self._update_status()
        return result

    def on_reject(self, order: TradeInstruction, **kwargs):
        order_id = order.order_id
        trade_handler = self.reversed_order_mapping.get(order_id)

        if trade_handler is None:
            return 0

        result = trade_handler.on_reject(order=order, **kwargs)

        self.cancel_all()
        self.unwind_all()
        self._update_status()
        return result

    def recover(self):
        if self.status != self.Status.error:
            LOGGER.debug(f'{self} status normal')
            return

        for handler_id in list(self.trade_handler):
            trade_handler = self.trade_handler.get(handler_id)

            if trade_handler is None:
                continue

            trade_handler.recover()

        if self.working_volume_summed:
            status = None

            for handler_id in list(self.trade_handler):
                trade_handler = self.trade_handler.get(handler_id)

                if trade_handler is None:
                    continue

                if trade_handler.status in [trade_handler.Status.positioning, trade_handler.Status.closed, trade_handler.Status.idle]:
                    continue
                elif trade_handler.status == trade_handler.Status.winding:
                    _status = self.Status.winding
                elif trade_handler.status == trade_handler.Status.unwinding:
                    _status = self.Status.unwinding
                else:
                    break

                if status is None:
                    status = _status
                elif status != _status:
                    status = None
                    break

            if status is not None:
                self._update_status(status=status)
            else:
                LOGGER.info(f'{self} failed to recover from error')
                return
        else:
            if self.exposure_volume:
                self._update_status(status=self.Status.positioning)
            else:
                self._update_status(status=self.Status.closed)

    def _update_status(self, status: 'TradePos.Status' = None, **kwargs):
        if 'open_time' in kwargs:
            self.open_time = kwargs['open_time']
        elif 'close_time' in kwargs:
            self.close_time = kwargs['close_time']

        all_status = []
        for trade_id in list(self.trade_handler):
            trade_handler = self.trade_handler.get(trade_id)

            if trade_handler is None:
                continue

            handler_status = trade_handler.status

            if handler_status == TradeHandler.Status.closed or handler_status == TradeHandler.Status.idle:
                continue

            all_status.append(handler_status)

        if status:
            if isinstance(status, self.Status):
                self.status = status
            else:
                raise TypeError(f'Invalid type {type(status)}. Can not assign state {status}')
        else:
            if any([_ == TradeHandler.Status.error for _ in all_status]):
                self.status = self.Status.error
            elif self.status == self.Status.idle:
                if not all_status:
                    self.status = self.Status.closed
                elif all([_ == TradeHandler.Status.winding for _ in all_status]):
                    self.status = self.Status.winding
                elif all([_ == TradeHandler.Status.positioning for _ in all_status]):
                    self.status = self.Status.positioning
                elif all([_ == TradeHandler.Status.unwinding for _ in all_status]):
                    self.status = self.Status.unwinding
                elif len(set(all_status)) > 2:
                    self.status = self.Status.error
                    LOGGER.warning(f'{self} synchronization lost!')
            elif self.status == self.Status.winding:
                if not all_status:
                    self.status = self.Status.closed
                elif all([_ == TradeHandler.Status.positioning for _ in all_status]):
                    self.status = self.Status.positioning
                elif all([_ == TradeHandler.Status.unwinding for _ in all_status]):
                    self.status = self.Status.unwinding
                elif len(set(all_status)) > 2:
                    self.status = self.Status.error
                    LOGGER.warning(f'{self} synchronization lost!')
            elif self.status == self.Status.positioning:
                if not all_status:
                    self.status = self.Status.closed
                elif all([_ == TradeHandler.Status.unwinding for _ in all_status]):
                    self.status = self.Status.unwinding
                elif len(set(all_status)) > 2:
                    self.status = self.Status.error
                    LOGGER.warning(f'{self} synchronization lost!')
            elif self.status == self.Status.unwinding:
                if not all_status:
                    self.status = self.Status.closed
                elif all([_ == TradeHandler.Status.positioning for _ in all_status]):
                    self.status = self.Status.positioning
                elif len(set(all_status)) > 2:
                    self.status = self.Status.error
                    LOGGER.warning(f'{self} synchronization lost!')
            elif self.status == self.Status.canceling:
                if not all_status:
                    self.status = self.Status.closed
                elif all([_ == TradeHandler.Status.idle for _ in all_status]):
                    self.status = self.Status.idle
                elif all([_ in [TradeHandler.Status.positioning, TradeHandler.Status.closed, TradeHandler.Status.idle] for _ in all_status]):
                    self.status = self.Status.positioning

    def pnl(self, pct=True):
        pnl = 0.
        notional = 0.
        for trade_id in list(self.trade_handler):
            trade_handler = self.trade_handler.get(trade_id)

            if trade_handler is None:
                continue

            pnl = trade_handler.pnl(pct=False)
            notional = trade_handler.open_notional

        if pct:
            if notional:
                pnl = pnl / notional
            else:
                pnl = np.inf * np.sign(pnl)

        return pnl

    def to_json(self, fmt='str') -> Union[str, dict]:
        json_dict = {
            'pos_id': self.pos_id,
            'status': self.status.name,
            'open_time': datetime.datetime.timestamp(self.open_time) if self.open_time else None,
            'close_time': datetime.datetime.timestamp(self.close_time) if self.close_time else None,
            'max_gain': self.max_gain,
            'max_loss': self.max_loss,
            'trade_handler': {_: self.trade_handler[_].to_json(fmt='dict') for _ in self.trade_handler},
        }

        if fmt == 'dict':
            return json_dict
        else:
            return json.dumps(json_dict)

    def from_json(self, json_str: Union[str, dict]):
        if isinstance(json_str, (str, bytes)):
            json_dict = json.loads(json_str)
        elif isinstance(json_str, dict):
            json_dict = json_str
        else:
            raise TypeError(f'Invalid type {type(json_str)}, expect [str, bytes, dict]')

        self.pos_id = json_dict['pos_id']
        self.status = self.Status[json_dict['status']]
        self.open_time = None if json_dict['open_time'] is None else datetime.datetime.fromtimestamp(json_dict['open_time'])
        self.close_time = None if json_dict['close_time'] is None else datetime.datetime.fromtimestamp(json_dict['close_time'])
        self.max_gain = json_dict['max_gain']
        self.max_loss = json_dict['max_loss']

        for handler_id in json_dict['trade_handler']:
            handler_json = json_dict['trade_handler'][handler_id]
            trade_handler = TradeHandler(
                ticker=handler_json['ticker'],
                side=TransactionSide(handler_json['side']),
                target_volume=handler_json['target_volume'],
                dma=self.dma,
                default_algo=handler_json['default_algo'],
                logger=self.logger,
                trade_id=handler_json['trade_id']
            )

            trade_handler.from_json(handler_json)
            self.trade_handler[trade_handler.trade_id] = trade_handler

        return self

    @property
    def notional(self):
        notional = 0.
        for trade_id in list(self.trade_handler):
            trade_handler = self.trade_handler.get(trade_id)

            if trade_handler is None:
                continue

            notional += trade_handler.open_notional
        return notional

    @property
    def working_volume_summed(self) -> Dict[str, float]:
        """
        a dict[ticker, float = summed working volume of given ticker]
        """
        working = {}
        for trade_id in list(self.trade_handler):
            trade_handler = self.trade_handler.get(trade_id)

            if trade_handler is None:
                continue

            ticker = trade_handler.ticker
            working[ticker] = working.get(ticker, 0.) + trade_handler.working_volume

        for ticker in list(working):
            if not working[ticker]:
                working.pop(ticker)

        return working

    @property
    def working_volume(self) -> Dict[str, Dict[str, float]]:
        """
        a dict[ticker, dict[side, summed working volume of given ticker]]
        """

        working_long = {}
        working_short = {}
        working = {'Long': working_long, 'Short': working_short}

        for trade_id in list(self.trade_handler):
            trade_handler = self.trade_handler.get(trade_id)

            if trade_handler is None:
                continue

            side = trade_handler.side.sign
            ticker = trade_handler.ticker

            if side > 0:
                working_long[ticker] = working_long.get(ticker, 0.) + trade_handler.working_volume_open
                working_short[ticker] = working_short.get(ticker, 0.) + trade_handler.working_volume_unwind
            elif side < 0:
                working_short[ticker] = working_short.get(ticker, 0.) + trade_handler.working_volume_open
                working_long[ticker] = working_long.get(ticker, 0.) + trade_handler.working_volume_unwind

        for side in working:
            _ = working[side]

            for ticker in list(_):
                if not _[ticker]:
                    _.pop(ticker)

        return working

    @property
    def exposure_volume(self) -> Dict[str, float]:
        """
        a dict[ticker, summed exposed volume of given ticker]
        """
        exposure = {}

        for trade_id in list(self.trade_handler):
            trade_handler = self.trade_handler.get(trade_id)

            if trade_handler is None:
                continue

            ticker = trade_handler.ticker
            exposure[ticker] = trade_handler.exposure_volume + exposure.get(ticker, 0.)

        for ticker in list(exposure):
            if not exposure[ticker]:
                exposure.pop(ticker)

        return exposure

    @property
    def inventory_volume(self) -> Dict[str, Dict[str, float]]:
        """
        a dict[ticker, dict[side, summed inventory volume of given ticker]]
        """
        inventory = {'Long': {}, 'Short': {}}

        for trade_id in list(self.trade_handler):
            trade_handler = self.trade_handler.get(trade_id)

            if trade_handler is None:
                continue

            ticker = trade_handler.ticker

            inv = inventory[trade_handler.side.side_name]
            inv[ticker] = trade_handler.inventory_volume + inv.get(ticker, 0.)

        for side in inventory:
            _ = inventory[side]

            for ticker in list(_):
                if not _[ticker]:
                    _.pop(ticker)

        return inventory

    @property
    def cash_flow(self) -> float:
        cash_flow = 0.
        for trade_id in list(self.trade_handler):
            trade_handler = self.trade_handler.get(trade_id)

            if trade_handler is None:
                continue

            cash_flow += trade_handler.cash_flow
        return cash_flow

    @property
    def fee(self) -> float:
        fee = 0.
        for trade_id in list(self.trade_handler):
            trade_handler = self.trade_handler.get(trade_id)

            if trade_handler is None:
                continue

            fee += trade_handler.fee
        return fee

    @property
    def market_price(self):
        return self.dma.market_price

    @property
    def market_time(self):
        return self.dma.market_time

    @property
    def orders(self) -> Dict[str, TradeInstruction]:
        orders = {}

        for handler_id in list(self.trade_handler):
            trade_handler = self.trade_handler.get(handler_id)

            if trade_handler is None:
                continue

            orders.update(trade_handler.orders)

        return orders

    @property
    def working_order(self) -> Dict[str, TradeInstruction]:
        working_order = {}

        for handler_id in list(self.trade_handler):
            trade_handler = self.trade_handler.get(handler_id)

            if trade_handler is None:
                continue

            working_order.update(trade_handler.working_order)

        return working_order

    @property
    def trades(self) -> Dict[str, TradeReport]:
        trades = {}

        for handler_id in list(self.trade_handler):
            trade_handler = self.trade_handler.get(handler_id)

            if trade_handler is None:
                continue

            trades.update(trade_handler.trades)

        return trades

    @property
    def order_mapping(self) -> Dict[str, Dict[str, TradeInstruction]]:
        order_mapping = {}

        for handler_id in list(self.trade_handler):
            trade_handler = self.trade_handler.get(handler_id)

            if trade_handler is None:
                continue

            order_mapping[trade_handler.trade_id] = trade_handler.orders

        return order_mapping

    @property
    def reversed_order_mapping(self) -> Dict[str, TradeHandler]:
        reversed_order_mapping = {}

        for handler_id in list(self.trade_handler):
            trade_handler = self.trade_handler.get(handler_id)

            if trade_handler is None:
                continue

            for order_id in trade_handler.orders:
                reversed_order_mapping[order_id] = trade_handler

        return reversed_order_mapping

    @property
    def tickers(self):
        tickers = set()

        for handler_id in list(self.trade_handler):
            trade_handler = self.trade_handler.get(handler_id)

            if trade_handler is None:
                continue

            tickers.add(trade_handler.ticker)

        return tickers

    @property
    def holding_time(self):
        if self.open_time is None:
            return 0.
        else:
            open_time = self.open_time
            if self.close_time is None:
                close_time = self.dma.market_time
            else:
                close_time = self.close_time

        holding_time = self.dma.trade_time_between(start_time=open_time, end_time=close_time).total_seconds()
        return holding_time


class PositionTracker(object):
    """
    PositionTracker handles all TradePos from a given strategy / factor

    PositionTracker is bounded to a specific strategy, it can track multiple TradePos for multiple tickers.

    PositionTracker is designed to manage TradePos with accuracy and ease
    It provides several feature for TradePos
    :param stop_loss: a NEGATIVE float, indicates how much loss (percentage to notional) to trigger an auto-unwind for any TradePos. e.g. -0.02 -> 2% loss compared to notional
    :param stop_gain: a POSITIVE float, similar to stop_loss, indicates how much percentage gain to trigger auto-unwind.
    :param timeout: a POSITIVE float, indicates how long, after the first fills, to auto-unwind the TradePos, in seconds

    All 3 features above only works on POSITIONING TradePos. working or idle TradePos is unaffected. These auto-unwind features respects rules in risk control.
    """

    def __init__(
            self,
            dma: DirectMarketAccess,
            **kwargs
    ):
        self.dma = dma
        self.tracker_id = kwargs.pop('tracker_id', uuid.uuid4().hex)
        self.logger = kwargs.pop('logger', LOGGER)
        self.stop_loss = kwargs.pop('stop_loss', None)
        self.stop_gain = kwargs.pop('stop_gain', None)
        self.timeout = kwargs.pop('timeout', None)
        self.position_limit = kwargs.pop('position_limit', 1)

        self._trade_position: Dict[str, TradePos] = {}
        self._trade_history: Dict[str, TradePos] = {}
        self._trade_manual: Dict[str, TradePos] = {}

    def __call__(self, market_data: MarketData):
        self.on_market_data(market_data=market_data)

    def add_pos(self, **kwargs) -> TradePos:
        pos = TradePos(
            logger=LOGGER,
            dma=self.dma,
            **kwargs
        )
        self._trade_position[pos.pos_id] = pos
        return pos

    def pop_pos(self, pos_id: str):
        self._trade_position.pop(pos_id, None)

    def pos_done(self, trade_pos: TradePos = None, pos_id: str = None):
        if pos_id is None:
            pos_id = trade_pos.pos_id

        _ = self._trade_position.pop(pos_id, trade_pos)
        self._trade_history[pos_id] = _
        LOGGER.info(f'TradePos {_} complete!')

    def pos_error(self, trade_pos: TradePos = None, pos_id: str = None):
        if pos_id is None:
            pos_id = trade_pos.pos_id

        _ = self._trade_position.pop(pos_id, trade_pos)
        self._trade_manual[pos_id] = _
        LOGGER.warning(f'TradePos {_} report error!')

    def on_fill(self, report: TradeReport, **kwargs):
        order_id = report.order_id
        trade_pos = self.reversed_order_mapping.get(order_id)

        if trade_pos is None:
            return 0

        result = trade_pos.on_fill(report=report, **kwargs)
        self._update_status()
        return result

    def on_cancel(self, order_id: str, **kwargs):
        trade_pos = self.reversed_order_mapping.get(order_id)

        if trade_pos is None:
            return 0

        result = trade_pos.on_cancel(order_id=order_id, **kwargs)
        self._update_status()
        return result

    def on_reject(self, order: TradeInstruction, **kwargs):
        order_id = order.order_id
        trade_pos = self.reversed_order_mapping.get(order_id)

        if trade_pos is None:
            return 0

        result = trade_pos.on_reject(order=order, **kwargs)
        self._update_status()
        return result

    def unwind_all(self, **kwargs):
        no_cross = kwargs.pop('no_cross', False)

        if not no_cross:
            exposure = self.exposure_volume
            additional_kwargs = kwargs.copy()

            for ticker in exposure:
                self.unwind_ticker(ticker, **additional_kwargs)

            return 0.
        else:
            # EMERGENCY ONLY
            LOGGER.warning('emergency fully unwinding triggered!')

            for pos_id in list(self.working_trade_pos):
                trade_pos = self.working_trade_pos.get(pos_id)

                if trade_pos is not None:
                    trade_pos.unwind_all(**kwargs)

            return 0

    def unwind_ticker(self, ticker: str, **kwargs):
        LOGGER.info(f'fully cancel and unwind {ticker} position!')

        # cancel all
        for pos_id in list(self.working_trade_pos):
            trade_pos = self.working_trade_pos.get(pos_id)

            if trade_pos is not None and ticker in trade_pos.exposure_volume:
                trade_pos.cancel_all(**kwargs)

        # calculate exposure
        exposure = self.exposure_volume.get(ticker)

        if not exposure:
            return

        to_unwind = abs(exposure)
        side = TransactionSide.Sell_to_Unwind if exposure > 0 else TransactionSide.Buy_to_Cover

        for pos_id in list(self.working_trade_pos):
            trade_pos = self.working_trade_pos.get(pos_id)

            if trade_pos is None:
                continue

            remains = trade_pos.unwind_auto(ticker=ticker, side=side, volume=to_unwind, **kwargs)
            to_unwind = remains

            if not remains:
                break

        if to_unwind:
            trade_pos = self.add_pos(source='fully_unwind', signal='fully_unwind')

            _trade_id = trade_pos.add_target(ticker=ticker, side=side, target_volume=to_unwind)
            trade_pos.open(**kwargs)

    def cancel_all(self, **kwargs):
        # EMERGENCY ONLY
        for pos_id in list(self.working_trade_pos):
            trade_pos = self.working_trade_pos.get(pos_id)

            if trade_pos is not None:
                trade_pos.cancel_all(**kwargs)

        return 0

    def on_market_data(self, market_data: MarketData):
        for pos_id in list(self.working_trade_pos):
            trade_pos = self._trade_position.get(pos_id)

            if trade_pos is None:
                continue

            trade_pos.on_market_data(market_data=market_data)

            # skip auto-unwind check
            if self.stop_gain is None \
                    and self.stop_loss is None \
                    and self.timeout is None:
                continue

            if trade_pos.status == trade_pos.Status.positioning:
                try:
                    pnl = trade_pos.pnl(pct=True)
                except (KeyError, ValueError, ZeroDivisionError) as _:
                    pnl = np.nan

                trade_pos.max_gain = np.nanmax([pnl, trade_pos.max_gain])
                trade_pos.max_loss = np.nanmin([pnl, trade_pos.max_loss])

                to_close = False
                if self.stop_gain is not None and pnl > self.stop_gain > 0:
                    LOGGER.info(f'STOP GAIN! {self} unwind position {trade_pos.exposure_volume}')
                    to_close = True
                elif self.stop_loss is not None and pnl < self.stop_loss < 0:
                    LOGGER.info(f'STOP LOSS! {self} unwind position {trade_pos.exposure_volume}')
                    to_close = True
                elif self.timeout is not None and self.dma.trade_time_between(start_time=trade_pos.open_time, end_time=self.market_time).total_seconds() > self.timeout > 0:
                    LOGGER.info(f'TIME OUT! {self} unwind position {trade_pos.exposure_volume}')
                    to_close = True

                if to_close:
                    trade_pos.unwind_all()

    def recover(self):
        for pos_id in list(self._trade_manual):
            trade_pos = self._trade_manual.get(pos_id)

            if trade_pos is None:
                continue

            trade_pos.recover()

            if trade_pos.status != trade_pos.Status.error:
                self._trade_manual.pop(pos_id, None)

                if trade_pos.exposure_volume:
                    self._trade_position[pos_id] = trade_pos
                else:
                    self._trade_history[pos_id] = trade_pos

    def to_json(self, fmt='str') -> Union[str, dict]:
        json_dict = {}
        map_id = self.tracker_id

        json_dict[map_id] = {}

        # dump working trade pos
        for pos_id in list(self.working_trade_pos):
            trade_pos = self.working_trade_pos.get(pos_id)

            if trade_pos is not None:
                json_dict[map_id][pos_id] = trade_pos.to_json(fmt='dict')

        # dump manual trade pos
        for pos_id in list(self.manual_trade_pos):
            trade_pos = self.manual_trade_pos.get(pos_id)

            if trade_pos is not None:
                json_dict[map_id][pos_id] = trade_pos.to_json(fmt='dict')

        if fmt == 'dict':
            return json_dict
        else:
            return json.dumps(json_dict)

    def _update_status(self):
        for pos_id in list(self._trade_position):
            trade_pos = self._trade_position.get(pos_id)

            if trade_pos is None:
                continue

            if trade_pos.status == trade_pos.Status.closed:
                self.pos_done(trade_pos=trade_pos)
            elif trade_pos.status == trade_pos.Status.error:
                self.pos_error(trade_pos=trade_pos)

    @property
    def pnl(self) -> Dict[str, float]:
        pnl = {}
        for pos_id in list(self.all_trade_pos):
            trade_pos = self.all_trade_pos.get(pos_id)

            if trade_pos is None:
                continue

            for trade_id in list(trade_pos.trade_handler):
                trade_handler = trade_pos.trade_handler.get(trade_id)

                if trade_handler is None:
                    continue

                ticker = trade_handler.ticker
                pnl[ticker] = trade_handler.pnl(pct=False) + pnl.get(ticker, 0)

        return pnl

    @property
    def notional(self) -> Dict[str, float]:
        notional = {}
        for pos_id in list(self.all_trade_pos):
            trade_pos = self.all_trade_pos.get(pos_id)

            if trade_pos is None:
                continue

            for trade_id in list(trade_pos.trade_handler):
                trade_handler = trade_pos.trade_handler.get(trade_id)

                if trade_handler is None:
                    continue

                ticker = trade_handler.ticker
                notional[ticker] = trade_handler.notional + notional.get(ticker, 0)

        return notional

    @property
    def working_volume(self) -> Dict[str, Dict[str, float]]:
        """
        a dictionary indicating current working volume of all orders

        {'Long': +float, 'Short': +float}

        :return: a Dict with non-negative numbers
        """
        working_long = {}
        working_short = {}
        working = {'Long': working_long, 'Short': working_short}

        for pos_id in list(self.working_trade_pos):
            trade_pos = self.working_trade_pos.get(pos_id)

            if trade_pos is not None:
                pos_working = trade_pos.working_volume

                for ticker in (_ := pos_working['Long']):
                    working_long[ticker] = working_long.get(ticker, 0.) + _.get(ticker, 0.)

                for ticker in (_ := pos_working['Short']):
                    working_short[ticker] = working_short.get(ticker, 0.) + _.get(ticker, 0.)

        for side in working:
            _ = working[side]

            for ticker in list(_):
                if not _[ticker]:
                    _.pop(ticker)

        return working

    @property
    def exposure_volume(self) -> Dict[str, float]:
        exposure = {}

        for pos_id in list(self.all_trade_pos):
            trade_pos = self.all_trade_pos.get(pos_id)

            if trade_pos is not None:
                for ticker in (pos_exposure := trade_pos.exposure_volume):
                    exposure[ticker] = exposure.get(ticker, 0.) + pos_exposure.get(ticker, 0.)

                    if exposure[ticker] == 0:
                        exposure.pop(ticker)

        for ticker in list(exposure):
            if not exposure[ticker]:
                exposure.pop(ticker)

        return exposure

    @property
    def working_volume_summed(self) -> Dict[str, float]:
        working = {}

        for pos_id in list(self.working_trade_pos):
            trade_pos = self.working_trade_pos.get(pos_id)

            if trade_pos is not None:
                for ticker in (pos_working := trade_pos.working_volume_summed):
                    working[ticker] = working.get(ticker, 0.) + pos_working[ticker]

                    if working[ticker] == 0:
                        working.pop(ticker)

        for ticker in list(working):
            if not working[ticker]:
                working.pop(ticker)

        return working

    @property
    def market_price(self):
        return self.dma.market_price

    @property
    def market_time(self):
        return self.dma.market_time

    @property
    def orders(self) -> Dict[str, TradeInstruction]:
        orders = {}

        for pos_id in list(self.all_trade_pos):
            trade_pos = self.all_trade_pos.get(pos_id)

            if trade_pos is None:
                continue

            orders.update(trade_pos.orders)

        return orders

    @property
    def working_order(self) -> Dict[str, TradeInstruction]:
        working_order = {}

        for pos_id in list(self.working_trade_pos):
            trade_pos = self.working_trade_pos.get(pos_id)

            if trade_pos is None:
                continue

            working_order.update(trade_pos.working_order)

        return working_order

    @property
    def trades(self) -> Dict[str, TradeReport]:
        trades = {}

        for pos_id in list(self.all_trade_pos):
            trade_pos = self.all_trade_pos.get(pos_id)

            if trade_pos is None:
                continue

            trades.update(trade_pos.trades)

        return trades

    @property
    def order_mapping(self) -> Dict[str, Dict[str, TradeInstruction]]:
        order_mapping = {}

        for pos_id in list(self.all_trade_pos):
            trade_pos = self.all_trade_pos.get(pos_id)

            if trade_pos is None:
                continue

            order_mapping[trade_pos.pos_id] = trade_pos.orders

        return order_mapping

    @property
    def handler_mapping(self) -> Dict[str, Dict[str, TradeHandler]]:
        handler_mapping = {}

        for pos_id in list(self.all_trade_pos):
            trade_pos = self.all_trade_pos.get(pos_id)

            if trade_pos is None:
                continue

            handler_mapping[trade_pos.pos_id] = trade_pos.trade_handler

        return handler_mapping

    @property
    def reversed_order_mapping(self) -> Dict[str, TradePos]:
        reversed_order_mapping = {}

        for pos_id in list(self.all_trade_pos):
            trade_pos = self.all_trade_pos.get(pos_id)

            if trade_pos is None:
                continue

            for order_id in trade_pos.orders:
                reversed_order_mapping[order_id] = trade_pos

        return reversed_order_mapping

    @property
    def reversed_handler_mapping(self) -> Dict[str, TradePos]:
        reversed_handler_mapping = {}

        for pos_id in list(self.all_trade_pos):
            trade_pos = self.all_trade_pos.get(pos_id)

            if trade_pos is None:
                continue

            for trade_id in trade_pos.trade_handler:
                reversed_handler_mapping[trade_id] = trade_pos

        return reversed_handler_mapping

    @property
    def trade_pos(self) -> Dict[str, TradePos]:
        LOGGER.warning(DeprecationWarning(f'{self.__class__.__name__} .trade_pos deprecated, use .all_trade_pos instead!'))
        return self.all_trade_pos

    @property
    def trade_position(self):
        LOGGER.warning(DeprecationWarning(f'{self.__class__.__name__} .trade_position deprecated, use .working_trade_pos instead!'))
        return self._trade_position

    @property
    def manual_trade_pos(self):
        return self._trade_manual

    @property
    def working_trade_pos(self) -> Dict[str, TradePos]:
        for pos_id in list(self._trade_position):
            trade_pos = self._trade_position.get(pos_id)

            if trade_pos is None:
                continue

            if trade_pos.status == trade_pos.Status.closed:
                self.pos_done(trade_pos=trade_pos)
            elif trade_pos.status == trade_pos.Status.error:
                self.pos_error(trade_pos=trade_pos)

        return self._trade_position

    @property
    def trade_history(self) -> Dict[str, TradePos]:
        return self._trade_history

    @property
    def all_trade_pos(self) -> Dict[str, TradePos]:
        all_trade_pos = {}
        all_trade_pos.update(self._trade_position)
        all_trade_pos.update(self._trade_history)
        all_trade_pos.update(self._trade_manual)
        return all_trade_pos


class Balance(object):
    """
    Balance handles mapping of PositionTracker <-> Strategy
    """

    class _BookTracker(object):
        def __init__(self, pos_tracker=None, **kwargs):
            self.pos_tracker: PositionTracker = pos_tracker
            self._working = kwargs.pop('working', {'Long': {}, 'Short': {}})
            self._exposure = kwargs.pop('exposure', {})
            self._cash = kwargs.pop('cash', 0.)
            self.lock = threading.Lock()

        def add_working(self, ticker: str, volume: float, side: Union[str, int, TransactionSide]):
            if not isinstance(side, TransactionSide):
                side = TransactionSide(side)

            self.lock.acquire()
            if volume:
                _ = self._working[side.side_name]
                _[ticker] = _.get(ticker, 0.) + volume
            self.lock.release()

        def add_exposure(self, ticker: str, volume: float):
            self.lock.acquire()
            if volume:
                if ticker in self._exposure:
                    self._exposure[ticker] += volume
                else:
                    self._exposure[ticker] = volume
            self.lock.release()

        def clear(self):
            self._working['Long'].clear()
            self._working['Short'].clear()
            self._exposure.clear()
            self._cash = 0.

        def update(self, position_tracker: PositionTracker = None, reset=True):
            if reset:
                self.clear()

            if position_tracker is None:
                if self.pos_tracker is None:
                    return
                position_tracker = self.pos_tracker

            for side in (working := position_tracker.working_volume):
                _ = working[side]
                for ticker in _:
                    volume = _.get(ticker)

                    if np.isfinite(volume):
                        self.add_working(ticker=ticker, volume=volume, side=side)
                    else:
                        LOGGER.error(f'Invalid working volume {volume} for {ticker} in {position_tracker}')

            for ticker in (exposure := position_tracker.exposure_volume):
                volume = exposure.get(ticker)

                if np.isfinite(volume):
                    self.add_exposure(ticker=ticker, volume=volume)
                else:
                    LOGGER.error(f'Invalid exposure volume {volume} for {ticker} in {position_tracker}')

        @property
        def cash(self) -> float:
            self.lock.acquire()
            result = self._cash
            self.lock.release()
            return result

        @property
        def working(self) -> Dict[str, float]:
            self.lock.acquire()
            result = self._working
            self.lock.release()
            return result

        @property
        def exposure(self) -> Dict[str, float]:
            self.lock.acquire()
            result = self._exposure
            self.lock.release()
            return result

    Book = _BookTracker

    def __init__(self, **kwargs):
        self.book = kwargs.pop('book', self.Book())
        self.inventory: Optional[Inventory] = kwargs.pop('inventory', None)

        self.strategy = {}
        self.trade_logs: List[TradeReport] = []
        self.position_tracker: Dict[str, PositionTracker] = {}
        self.book_tracker: Dict[str, Balance._BookTracker] = {}

        self.last_update_timestamp = None

    def __repr__(self):
        return f'<Balance>{{id={id(self)}}}'

    def add(self, map_id: str = None, strategy=None, position_tracker: PositionTracker = None):
        if map_id is None:
            map_id = uuid.uuid4().hex

        if strategy is not None:
            self.strategy[map_id] = strategy

        if position_tracker is not None:
            self.position_tracker[map_id] = position_tracker

    def pop(self, map_id: str):
        self.strategy.pop(map_id, None)
        self.position_tracker.pop(map_id, None)
        self.book_tracker.pop(map_id, None)

    def get(self, **kwargs) -> Optional[Book]:
        map_id = kwargs.pop('map_id', None)
        strategy = kwargs.pop('strategy', None)
        position_tracker = kwargs.pop('position_tracker', None)

        if map_id is None:
            if strategy is not None:
                map_id = self.reversed_strategy_mapping.get(id(strategy))

                if map_id is None:
                    raise KeyError(f'Can not found strategy {strategy}')
            elif position_tracker is not None:
                map_id = self.reversed_tracker_mapping.get(position_tracker.tracker_id)

                if map_id is None:
                    raise KeyError(f'Can not found PositionTracker {position_tracker}')
            else:
                raise TypeError('Must assign one value of map_id, strategy or position_tracker')

        if map_id not in self.book_tracker:

            if map_id not in self.position_tracker:
                return None

            position_tracker = self.position_tracker.get(map_id)
            book = self.book_tracker[map_id] = self.Book(pos_tracker=position_tracker)
            book.update(reset=True)
        else:
            book = self.book_tracker[map_id]

        return book

    def get_strategy(self, strategy_name: str = None, strategy_id=None):
        match = None

        if strategy_name is not None:
            for _ in self.strategy.values():
                if _.name == strategy_name:
                    match = _
                    break
        elif strategy_id is not None:
            for _ in self.strategy.values():
                if _.strategy_id == strategy_id:
                    match = _
                    break
        else:
            LOGGER.error(ValueError('Must assign ether a strategy_name or a strategy_id'))

        return match

    def get_tracker(self, strategy_name: str = None, strategy_id=None) -> Optional[PositionTracker]:
        strategy = self.get_strategy(strategy_name=strategy_name, strategy_id=strategy_id)

        if strategy is None:
            return None

        map_id = self.reversed_strategy_mapping.get(id(strategy))
        tracker = self.position_tracker.get(map_id)
        return tracker

    def on_update(self, market_time=None):
        if market_time is None:
            market_time = time.time()

        # step 0: update market time
        self.last_update_timestamp = market_time

        # step 1: write balance file
        # self.dump(file_path=pathlib.Path(WORKING_DIRECTORY).joinpath('Dumps', 'balance.updated.json'))

        # step 2: write trade file
        # self.dump_trades(file_path=pathlib.Path(WORKING_DIRECTORY).joinpath('Dumps', 'trades.updated.csv'))

    def on_order(self, order: TradeInstruction, **kwargs):
        order_id = order.order_id
        order_state = order.order_state
        status_code = 0

        for tracker_name in list(self.position_tracker):
            position_tracker = self.position_tracker.get(tracker_name)

            if position_tracker is None:
                continue

            if order_id in position_tracker.working_order:
                if position_tracker.working_order[order_id] is not order:
                    LOGGER.error(f'Order object not static! stored id {id(position_tracker.working_order[order_id])}, updated id {id(order)}')

                if order_state == OrderState.Canceled:
                    position_tracker.on_cancel(order_id=order_id, **kwargs)
                elif order_state == OrderState.Rejected:
                    position_tracker.on_reject(order=order, **kwargs)

                self.update(position_tracker)
                status_code = 1
                break

        if not status_code:
            if order_state == OrderState.Filled:
                LOGGER.debug(f'No match for filled order {order}, perhaps the Algo.on_filled called before Balance.on_order. This is not an error.')
            else:
                LOGGER.error(f'No match for {order.side} order {order}')

        self.on_update()
        return status_code

    def on_report(self, report: TradeReport, **kwargs):
        order_id = report.order_id
        status_code = 0

        for tracker_name in list(self.position_tracker):
            position_tracker = self.position_tracker.get(tracker_name)

            if position_tracker is None:
                continue

            if order_id in position_tracker.working_order:
                position_tracker.on_fill(report=report, **kwargs)

                self.update(position_tracker)
                status_code = 1
                break

        if not status_code:
            LOGGER.warning(f'No match for report {report}')

        self.on_update()
        self.trade_logs.append(report)
        return status_code

    def update(self, *trackers: PositionTracker):
        # update given tracker
        for tracker in trackers:
            map_id = self.reversed_tracker_mapping.get(tracker.tracker_id)

            if map_id is None:
                continue

            if map_id not in self.position_tracker:
                continue

            if map_id not in self.book_tracker:
                book = self.book_tracker[map_id] = self.Book(pos_tracker=tracker)
            else:
                book = self.book_tracker[map_id]

            book.update(reset=True)

        # update portfolio tracker
        self.book.clear()
        for tracker_name in list(self.position_tracker):
            position_tracker = self.position_tracker.get(tracker_name)

            if position_tracker is None:
                continue

            self.book.update(position_tracker=position_tracker, reset=False)

    def reset(self):
        self.book.clear()
        self.position_tracker.clear()
        self.book_tracker.clear()

    def to_json(self, fmt='str') -> Union[str, dict]:
        json_dict = {}

        for map_id in self.position_tracker:
            tracker = self.position_tracker.get(map_id)

            if tracker is not None:
                json_dict.update(tracker.to_json(fmt='dict'))

        if fmt == 'dict':
            return json_dict
        else:
            return json.dumps(json_dict)

    def from_json(self, json_str: Union[str, dict]):
        if isinstance(json_str, (str, bytes)):
            json_dict = json.loads(json_str)
        elif isinstance(json_str, dict):
            json_dict = json_str
        else:
            raise TypeError(f'Invalid type {type(json_str)}, expect [str, bytes, dict]')

        for map_id in json_dict:
            if map_id not in self.strategy:
                LOGGER.error(f'No strategy with key {map_id} found! Must register strategy before loading balance!')
                continue

            pos_tracker = self.position_tracker[map_id]
            pos_json = json_dict[map_id]

            for pos_id in pos_json:
                pos_json_dict = pos_json[pos_id]
                trade_pos = pos_tracker.add_pos(
                    pos_id=pos_json_dict['pos_id']
                )

                trade_pos.from_json(pos_json_dict)

        return self

    def dump(self, file_path: Union[str, pathlib.Path]):
        file_path = pathlib.Path(file_path)
        dump_dir = file_path.parent

        os.makedirs(dump_dir, exist_ok=True)

        with open(file_path, 'w') as f:
            f.write(json.dumps(self.to_json(fmt='dict'), indent=4, sort_keys=True))

    def dump_trades(self, file_path: Union[pathlib.Path, str], ts_from: float = None, ts_to: float = None):
        trades_dict = {}

        for strategy_id in self.position_tracker:
            tracker = self.position_tracker[strategy_id]
            trades = tracker.trades

            for trade_id in trades:
                report = trades[trade_id]
                trade_time = report.trade_time
                ts = trade_time.timestamp()

                if ts_from is not None and ts < ts_from:
                    continue
                elif ts_to is not None and ts > ts_to:
                    continue

                trades_dict[trade_id] = dict(
                    strategy=strategy_id,
                    ticker=report.ticker,
                    side=report.side.side_name,
                    volume=report.volume,
                    price=report.price,
                    notional=report.notional,
                    time=report.trade_time,
                    ts=report.timestamp,
                )

        trades_df = pd.DataFrame(trades_dict).T

        if not trades_df.empty:
            trades_df.sort_values('ts')

        trades_df.to_csv(file_path)
        return trades_dict

    def dump_trades_all(self, file_path: Union[pathlib.Path, str], ts_from: float = None, ts_to: float = None):
        trade_logs = []

        for report in self.trade_logs:  # type: TradeReport
            trade_time = report.trade_time
            ts = trade_time.timestamp()

            if ts_from is not None and ts < ts_from:
                continue
            elif ts_to is not None and ts > ts_to:
                continue

            trade_logs.append(dict(
                trade_id=report.trade_id,
                ticker=report.ticker,
                side=report.side.side_name,
                volume=report.volume,
                price=report.price,
                notional=report.notional,
                time=report.trade_time,
                ts=report.timestamp,
            ))

        trades_df = pd.DataFrame(trade_logs)

        if not trades_df.empty:
            trades_df.sort_values('ts')

        trades_df.to_csv(file_path)
        return trades_df

    def load(self, file_path: Union[str, pathlib.Path]):
        if not os.path.isfile(file_path):
            LOGGER.error(f'No such file {file_path}')
            return

        with open(file_path, 'r') as f:
            json_str = f.read()

        self.from_json(json_str)

    @property
    def tracker_mapping(self) -> Dict[str, str]:
        mapping = {}

        for map_id in self.position_tracker:
            tracker = self.position_tracker.get(map_id)

            if tracker is None:
                continue

            mapping[map_id] = tracker.tracker_id

        return mapping

    @property
    def reversed_tracker_mapping(self) -> Dict[str, str]:
        mapping = {}

        for id_0, id_1 in self.tracker_mapping.items():
            mapping[id_1] = id_0

        return mapping

    @property
    def strategy_mapping(self) -> Dict[str, int]:
        mapping = {}

        for map_id in self.strategy:
            strategy = self.strategy.get(map_id)

            if strategy is None:
                continue

            mapping[map_id] = id(strategy)

        return mapping

    @property
    def reversed_strategy_mapping(self) -> Dict[int, str]:
        mapping = {}

        for id_0, id_1 in self.strategy_mapping.items():
            mapping[id_1] = id_0

        return mapping

    @property
    def working_volume_summed(self) -> Dict[str, float]:
        working = {}

        for tracker_id in list(self.position_tracker):
            tracker = self.position_tracker.get(tracker_id)

            if tracker is not None:
                for ticker in tracker.working_volume_summed:
                    working[ticker] = working.get(ticker, 0.) + tracker.working_volume_summed.get(ticker, 0.)

                    if working[ticker] == 0:
                        working.pop(ticker)

        return working

    @property
    def exposure_volume(self) -> Dict[str, float]:
        exposure = {}

        for tracker_id in list(self.position_tracker):
            tracker = self.position_tracker.get(tracker_id)

            if tracker is not None:
                for ticker in tracker.exposure_volume:
                    exposure[ticker] = exposure.get(ticker, 0.) + tracker.exposure_volume[ticker]

                    if exposure[ticker] == 0:
                        exposure.pop(ticker)

        return exposure

    @property
    def working_volume(self) -> Dict[str, Dict[str, float]]:

        working_long = {}
        working_short = {}
        working = {'Long': working_long, 'Short': working_short}

        for tracker_id in list(self.position_tracker):
            tracker = self.position_tracker.get(tracker_id)

            if tracker is not None:
                tracker_working = tracker.working_volume

                for ticker in (_ := tracker_working['Long']):
                    working_long[ticker] = working_long.get(ticker, 0.) + _.get(ticker, 0.)

                for ticker in (_ := tracker_working['Short']):
                    working_short[ticker] = working_short.get(ticker, 0.) + _.get(ticker, 0.)

        for side in working:
            _ = working[side]

            for ticker in list(_):
                if not _[ticker]:
                    _.pop(ticker)

        return working

    def exposure_notional(self, mds) -> Dict[str, float]:
        notional = {}

        for ticker in self.exposure_volume:
            notional[ticker] = self.exposure_volume.get(ticker, 0.) * mds.market_price.get(ticker, 0)

        return notional

    def working_notional(self, mds) -> Dict[str, float]:
        notional = {}

        for ticker in (tracker_working := self.working_volume_summed):
            notional[ticker] = tracker_working[ticker] * mds.market_price.get(ticker, 0)

        return notional

    @property
    def orders(self) -> Dict[str, TradeInstruction]:
        orders = {}

        for tracker_id in list(self.position_tracker):
            tracker = self.position_tracker.get(tracker_id)

            if tracker is None:
                continue

            orders.update(tracker.orders)

        return orders

    @property
    def working_order(self) -> Dict[str, TradeInstruction]:
        working_order = {}

        for tracker_id in list(self.position_tracker):
            tracker = self.position_tracker.get(tracker_id)

            if tracker is None:
                continue

            working_order.update(tracker.working_order)

        return working_order

    @property
    def trades_today(self):
        trades = {}
        from .MarketEngine import MDS

        market_date = MDS.market_date
        if market_date is None:
            return {}

        for tracker_id in list(self.position_tracker):
            tracker = self.position_tracker.get(tracker_id)

            if tracker is None:
                continue

            for trade in tracker.trades.values():
                if trade.trade_time.date() == market_date:
                    trades[trade.trade_id] = trade

            # trades.update(tracker.trades)

        return trades

    @property
    def trades_session(self) -> Dict[str, TradeReport]:
        trades = {_.trade_id: _ for _ in self.trade_logs}

        return trades

    @property
    def trades(self) -> Dict[str, TradeReport]:
        return self.trades_today

    @property
    def cash(self):
        return self.book.cash

    @property
    def info(self) -> pd.DataFrame:
        info_dict = {
            'exposure': self.exposure_volume,
            'working_lone': self.working_volume['Long'],
            'working_short': self.working_volume['Short'],
        }

        return pd.DataFrame(info_dict).fillna(0)


class Inventory(object):
    """
    Inventory stores the info of security lending
    """

    class SecurityType(Enum):
        Commodity = 'Commodity'
        CurrencySwap = 'CurrencySwap'
        Crypto = 'Crypto'
        IndexFuture = 'IndexFuture'
        Stock = 'Stock'

    class CashDividend(object):
        def __init__(self, market_date: datetime.date, dividend_per_share: float):
            self.market_date = market_date
            self.dividend_per_share = dividend_per_share

    class StockDividend(object):
        def __init__(self, market_date: datetime.date, dividend_per_share: float):
            self.market_date = market_date
            self.dividend_per_share = dividend_per_share

    class StockSplit(object):
        def __init__(self, market_date: datetime.date, multiplier: float):
            self.market_date = market_date
            self.multiplier = multiplier

    class StockConversion(object):
        def __init__(self, market_date: datetime.date, convert_to: str, multiplier: float):
            self.convert_to = convert_to
            self.market_date = market_date
            self.multiplier = multiplier

    class Entry(object):
        def __init__(self, ticker: str, volume: float, price: float, security_type: 'Inventory.SecurityType', direction: TransactionSide, **kwargs):
            if volume < 0:
                LOGGER.warning('volume of Inventory.Entry normally should be positive!')

            self.ticker = ticker
            self.volume = volume
            self.price = price
            self.security_type = security_type
            self.direction = direction

            self.notional = kwargs.pop('notional', volume * price)
            self.fee = kwargs.pop('fee', 0.)
            self.recalled = kwargs.pop('recalled', 0.)

        def __repr__(self):
            return f'<Inventory.Entry>(ticker={self.ticker}, side={self.direction.side_name}, volume={self.volume:,}, fee={self.fee:.2f})'

        def __add__(self, other):
            if isinstance(other, self.__class__):
                return self.merge(other)

            raise TypeError(f'Can only merge type {self.__class__.__name__}')

        def __bool__(self):
            return self.volume.__bool__()

        def apply_cash_dividend(self, dividend: 'Inventory.CashDividend'):
            raise NotImplementedError()

        def apply_stock_dividend(self, dividend: 'Inventory.StockDividend'):
            raise NotImplementedError()

        def apply_conversion(self, stock_conversion: 'Inventory.StockConversion'):
            raise NotImplementedError()

        def apply_split(self, stock_split: 'Inventory.StockSplit'):
            raise NotImplementedError()

        def merge(self, entry: 'Inventory.Entry', inplace=False, **kwargs):
            if entry.ticker != self.ticker:
                raise ValueError(f'<ticker> not match! Expect {self.ticker}, got {entry.ticker}')

            if entry.direction.sign != self.direction.sign:
                raise ValueError(f'<direction> not match! Expect {self.direction}, got {entry.direction}')

            if entry.security_type != self.security_type:
                raise ValueError(f'<security_type> not match! Expect {self.security_type}, got {entry.security_type}')

            volume = kwargs.pop('volume', self.volume + entry.volume)
            notional = kwargs.pop('notional', self.notional + entry.notional)
            price = kwargs.pop('price', (self.price * self.volume + entry.price * entry.volume) / (self.volume + entry.volume))
            fee = kwargs.pop('fee', self.fee + entry.fee)
            recalled = kwargs.pop('recalled', self.recalled + entry.recalled)

            if inplace:
                self.volume = volume
                self.notional = notional
                self.price = price
                self.fee = fee
                self.recalled = recalled

                return self
            else:
                new_entry = self.__class__(
                    ticker=self.ticker,
                    volume=volume,
                    price=price,
                    security_type=self.security_type,
                    direction=self.direction,
                    notional=notional,
                    fee=fee,
                    recalled=recalled
                )

                return new_entry

        def to_json(self, fmt='str') -> Union[str, dict]:
            json_dict = dict(
                ticker=self.ticker,
                volume=self.volume,
                price=self.price,
                security_type=self.security_type.name,
                direction=self.direction.side_name,
                notional=self.notional,
                fee=self.fee,
                recalled=self.recalled
            )

            if fmt == 'dict':
                return json_dict
            else:
                return json.dumps(json_dict)

        @classmethod
        def from_json(cls, json_str: Union[str, dict]):
            if isinstance(json_str, (str, bytes)):
                json_dict = json.loads(json_str)
            elif isinstance(json_str, dict):
                json_dict = json_str
            else:
                raise TypeError(f'Invalid type {type(json_str)}, expect [str, bytes, dict]')

            entry = cls(
                ticker=json_dict['ticker'],
                volume=json_dict['volume'],
                price=json_dict['price'],
                security_type=Inventory.SecurityType[json_dict['security_type']],
                direction=TransactionSide(json_dict['direction']),
                notional=json_dict['notional'],
                fee=json_dict.get('fee', 0.),
                recalled=json_dict.get('recalled', 0.),
            )

            return entry

        @property
        def available(self):
            return max(self.volume - self.recalled, 0.)

    def __init__(self):
        self._inv: Dict[str, List[Inventory.Entry]] = {}
        self._traded: Dict[str, float] = {}
        self._tickers = set()

    def __repr__(self):
        return f'<Inventory>{{id={id(self)}}}'

    def __call__(self, ticker: str):
        return dict(
            Long=self.available_volume(ticker=ticker, direction=TransactionSide.LongOpen),
            Short=self.available_volume(ticker=ticker, direction=TransactionSide.ShortOpen)
        )

    def recall(self, ticker: str, volume: float, direction: TransactionSide = TransactionSide.LongOpen):
        key = f'{ticker}.{direction.side_name}'
        _ = self._inv.get(key, [])
        to_recall = volume

        for entry in _[:]:
            recalled = max(entry.volume, to_recall)
            entry.recalled += recalled

            if not entry.available:
                _.remove(entry)
                LOGGER.info(f'{entry} fully recalled!')
            else:
                LOGGER.info(f'{entry} recalled {recalled}, {entry.available} remains!')

            if not to_recall:
                break

        if not _:
            self._inv.pop(key)

    def add_inv(self, entry: Entry):
        self._tickers.add(entry.ticker)
        key = f'{entry.ticker}.{entry.direction.side_name}'
        _ = self._inv.get(key, [])

        _.append(entry)

        self._inv[key] = _

    def get_inv(self, ticker: str, direction: TransactionSide = TransactionSide.LongOpen) -> Optional[Entry]:
        key = f'{ticker}.{direction.side_name}'
        _ = self._inv.get(key, [])

        merged_entry = None
        for entry in _:
            if merged_entry is None:
                merged_entry = entry
            else:
                merged_entry = merged_entry + entry

        return merged_entry

    def use_inv(self, ticker: str, volume: float, direction: TransactionSide = TransactionSide.LongOpen):
        key = f'{ticker}.{direction.side_name}'

        self._traded[key] = self._traded.get(key, 0.) + volume

    def available_volume(self, ticker: str, direction: TransactionSide = TransactionSide.LongOpen) -> float:
        inv = self.get_inv(ticker=ticker, direction=direction)

        if inv is None:
            return 0.

        used = self._traded.get(ticker, 0.)
        return inv.available - used

    def clear(self):
        self._inv.clear()
        self._traded.clear()
        self._tickers.clear()

    def to_json(self, fmt='str') -> Union[str, dict]:
        json_dict = {}

        for name in self._inv:
            json_dict[name] = {
                'used': 0.,
                'inv': []
            }
            _ = self._inv[name]

            for entry in _:
                json_dict[name]['inv'].append(entry.to_json(fmt=fmt))

            json_dict[name]['used'] = self._traded.get(name, 0.)

        if fmt == 'dict':
            return json_dict
        else:
            return json.dumps(json_dict)

    def from_json(self, json_str: Union[str, dict], with_used=False):
        if isinstance(json_str, (str, bytes)):
            json_dict = json.loads(json_str)
        elif isinstance(json_str, dict):
            json_dict = json_str
        else:
            raise TypeError(f'Invalid type {type(json_str)}, expect [str, bytes, dict]')

        for name in json_dict:
            inv = json_dict[name]['inv']
            used = json_dict[name]['used']

            for entry_json in inv:
                entry = self.Entry.from_json(entry_json)
                self.add_inv(entry=entry)

            if with_used:
                self._traded[name] = used

        return self

    def dump(self, file_path: Union[str, pathlib.Path]):
        file_path = pathlib.Path(file_path)
        dump_dir = file_path.parent

        os.makedirs(dump_dir, exist_ok=True)

        with open(file_path, 'w') as f:
            f.write(json.dumps(self.to_json(fmt='dict'), indent=4, sort_keys=True))

    def to_csv(self, file_path: Union[str, pathlib.Path]):
        inv_dict = {'inv_l': {}, 'inv_s': {}}

        for ticker in self._inv:
            if (long_inv := self.get_inv(ticker=ticker, direction=TransactionSide.LongOpen)) is not None:
                inv_dict['inv_l'][ticker] = long_inv.volume

            if (short_inv := self.get_inv(ticker=ticker, direction=TransactionSide.ShortOpen)) is not None:
                inv_dict['inv_s'][ticker] = short_inv.volume

        inv_df = pd.DataFrame(inv_dict)
        inv_df.to_csv(file_path)

    def load(self, file_path: Union[str, pathlib.Path], with_used=False):
        if not os.path.isfile(file_path):
            LOGGER.error(f'No such file {file_path}')
            return

        with open(file_path, 'r') as f:
            json_str = f.read()

        self.clear()
        self.from_json(json_str, with_used=with_used)

    @property
    def tickers(self):
        return self._tickers

    @property
    def info(self) -> pd.DataFrame:
        info_dict = {'inv_l': {}, 'inv_s': {}}

        for ticker in self.tickers:
            inv_l = self.get_inv(ticker, TransactionSide.LongOpen)
            inv_s = self.get_inv(ticker, TransactionSide.ShortOpen)

            if inv_l is not None:
                info_dict['inv_l'][ticker] = inv_l.volume

            if inv_s is not None:
                info_dict['inv_s'][ticker] = inv_s.volume

        return pd.DataFrame(info_dict)


class RiskProfile(object):
    class Risk(Exception):
        def __init__(self, risk_type: str, code: int, msg: str, *args, **kwargs):
            self.code = code
            self.type = risk_type
            self.msg = msg

            super().__init__(msg, *args)

            for kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    def __init__(self, mds: MarketDataService, balance: Balance, **kwargs):
        self.mds = mds
        self.balance = balance

        self.rules = NameSpace(
            entry=set(),
            # --- individual constrains ---
            max_percentile={},
            max_trade_long={},
            max_trade_short={},
            max_exposure_long={},
            max_exposure_short={},
            max_notional_long={},
            max_notional_short={},
            # --- global constrains ---
            max_ttl_notional_long=None,
            max_ttl_notional_short=None,
            max_net_notional_long=None,
            max_net_notional_short=None,
        )

        self.rules.update(kwargs)

    def __repr__(self):
        return f'<RiskProfile>{{id={id(self)}}}'

    def __call__(self, *order: TradeInstruction):
        if len(order) == 1:
            return self.check(order=order[0])
        else:
            return self.check_basket(*order)

    def set_rule(self, key: str, value: float, ticker: str = None):
        if key in self.rules:
            limit_set = self.rules[key]
            new_limit = value

            # update global constrains
            if ticker is None:
                if not isinstance(limit_set, dict):
                    old_limit = limit_set
                    self.rules[key] = new_limit
                    LOGGER.info(f'{self} limit updated: <{key}>: {old_limit} -> {new_limit}')
                else:
                    LOGGER.error(f'Invalid action: limit <{key}> requires a valid ticker')
            # update individual constrains
            else:
                if isinstance(limit_set, dict):
                    self.rules.entry.add(ticker)
                    old_limit = limit_set.get(ticker, 'null')
                    self.rules[key][ticker] = new_limit
                    LOGGER.info(f'{self} limit updated: <{key}>({ticker}): {old_limit} -> {new_limit}')
                else:
                    LOGGER.error(f'Invalid action: can not set any ticker for limit <{key}>')
        else:
            LOGGER.error(f'Invalid action: limit <{key}> not found!')

    def get(self, ticker: str) -> Dict[str, Union[float, Dict[str, float]]]:
        limit = NameSpace(name=f'RiskLimit.{ticker}', market_price=self.mds.market_price.get(ticker))

        limit['working'] = self._get_volume(ticker=ticker, flag='working')
        limit['traded'] = self._get_volume(ticker=ticker, flag='traded')
        limit['exposure'] = self._get_volume(ticker=ticker, flag='exposure')

        # --- global constrains ---
        if self.rules.max_ttl_notional_long is not None:
            limit['max_ttl_notional_long'] = self.rules.max_ttl_notional_long

        if self.rules.max_ttl_notional_short is not None:
            limit['max_ttl_notional_short'] = self.rules.max_ttl_notional_short

        if self.rules.max_net_notional_long is not None:
            limit['max_net_notional_long'] = self.rules.max_net_notional_long

        if self.rules.max_net_notional_short is not None:
            limit['max_net_notional_short'] = self.rules.max_net_notional_short

        # --- individual constrains ---
        if ticker in self.rules.max_percentile:
            limit['max_percentile'] = self.rules.max_percentile.get(ticker, 1.)

        if ticker in self.rules.max_trade_long:
            limit['max_trade_long'] = self.rules.max_trade_long.get(ticker, np.inf)

        if ticker in self.rules.max_trade_short:
            limit['max_trade_short'] = self.rules.max_trade_short.get(ticker, np.inf)

        if ticker in self.rules.max_exposure_long:
            limit['max_exposure_long'] = self.rules.max_exposure_long.get(ticker, np.inf)

        if ticker in self.rules.max_exposure_short:
            limit['max_exposure_short'] = self.rules.max_exposure_short.get(ticker, np.inf)

        if ticker in self.rules.max_notional_long:
            limit['max_notional_long'] = self.rules.max_notional_long.get(ticker, np.inf)

        if ticker in self.rules.max_notional_short:
            limit['max_notional_short'] = self.rules.max_notional_short.get(ticker, np.inf)

        return limit

    def check(self, order: TradeInstruction):
        ticker = order.ticker

        # step 0: get limits
        limit = self.get(ticker=ticker)
        LOGGER.info(f'{self} defines {limit}')

        try:
            # step 0: check validity
            self._check_validity(order=order, limit=limit)

            # step 1: check inventory limit
            self._check_max_trade(order=order, limit=limit)

            # step 2: check position limit
            self._check_max_exposure(order=order, limit=limit)

            # step 3: check percentile limit
            self._check_max_percentile(order=order, limit=limit)

            # step 4: check notional limit
            self._check_max_notional(order=order, limit=limit)

            # step 5: check portfolio net limit
            self._check_net_portfolio(order=order, limit=limit)

            # step 6: check portfolio total limit
            self._check_ttl_portfolio(order=order, limit=limit)
        except self.Risk as e:
            LOGGER.error(f'<{e.type}.{e.code}>: {e.msg}')
            return False

        return True

    def check_order(self, ticker: str, volume: float, side: TransactionSide):
        fake_order = TradeInstruction(
            ticker=ticker,
            side=side,
            volume=volume
        )

        return self.check(order=fake_order)

    def check_basket(self, *order: TradeInstruction):
        LOGGER.warning('risk control for basket order not implemented, check order individually')

        for _ in order:
            self.check(_)

    def clear(self):
        self.rules.entry.clear()

        self.rules.max_percentile.clear()
        self.rules.max_trade_long.clear()
        self.rules.max_trade_short.clear()
        self.rules.max_exposure_long.clear()
        self.rules.max_exposure_short.clear()
        self.rules.max_notional_long.clear()
        self.rules.max_notional_short.clear()

        self.rules.max_ttl_notional_long = np.inf
        self.rules.max_ttl_notional_short = np.inf
        self.rules.max_net_notional_long = np.inf
        self.rules.max_net_notional_short = np.inf

    def to_json(self, fmt='str') -> Union[str, dict]:
        json_dict = dict(self.rules)
        json_dict['entry'] = list(json_dict['entry'])

        if fmt == 'dict':
            return json_dict
        else:
            return json.dumps(json_dict)

    def from_json(self, json_str: Union[str, dict]):
        if isinstance(json_str, (str, bytes)):
            json_dict = json.loads(json_str)
        elif isinstance(json_str, dict):
            json_dict = json_str
        else:
            raise TypeError(f'Invalid type {type(json_str)}, expect [str, bytes, dict]')

        self.rules.update(json_dict)
        self.rules['entry'] = set(self.rules['entry'])

        return self

    def dump(self, file_path: Union[str, pathlib.Path]):
        file_path = pathlib.Path(file_path)
        dump_dir = file_path.parent

        os.makedirs(dump_dir, exist_ok=True)

        with open(file_path, 'w') as f:
            f.write(json.dumps(self.to_json(fmt='dict'), indent=4, sort_keys=True))

    def load(self, file_path: Union[str, pathlib.Path]):
        if not os.path.isfile(file_path):
            LOGGER.error(f'No such file {file_path}')
            return

        with open(file_path, 'r') as f:
            json_str = f.read()

        self.from_json(json_str)

    def _check_validity(self, order: TradeInstruction, limit: Dict[str, Union[float, Dict[str, float]]]):
        ticker = order.ticker
        market_price = limit['market_price']

        if market_price is None:
            raise self.Risk(
                risk_type='RiskProfile.Internal.Price',
                code=100,
                msg=f'no valid market price for ticker {ticker}'
            )

        return True

    def _check_max_trade(self, order: TradeInstruction, limit: Dict[str, Union[float, Dict[str, float]]]):
        ticker = order.ticker
        action = abs(order.volume)
        side = order.side

        if side.sign > 0:
            flag = 'long'
        elif side.sign < 0:
            flag = 'short'
        else:
            return

        if f'max_trade_{flag}' not in limit:
            raise self.Risk(
                risk_type='RiskProfile.TradeLimit.Invalid',
                code=1003,
                msg=f'{ticker} {side.sign * action} rejected! {ticker} not trade-able!'
            )

        trade_limit = limit[f'max_trade_{flag}']
        working = limit['working']
        traded = limit['traded']
        trade_count = working[flag] + traded[flag]

        # for long order
        if side.sign > 0:
            if trade_count + action > trade_limit:
                raise self.Risk(
                    risk_type='RiskProfile.TradeLimit.Long',
                    code=1001,
                    msg=f'{ticker} {side.sign * action} rejected! lmt={trade_limit}, ttl={trade_count}, inv={trade_limit - trade_count}, action={action}'
                )
        elif side.sign < 0:
            if trade_count + action > trade_limit:
                raise self.Risk(
                    risk_type='RiskProfile.TradeLimit.Short',
                    code=1002,
                    msg=f'{ticker} {side.sign * action} rejected! lmt={trade_limit}, ttl={trade_count}, inv={trade_limit - trade_count}, action={-action}'
                )

        return True

    def _check_max_exposure(self, order: TradeInstruction, limit: Dict[str, Union[float, Dict[str, float]]]):
        ticker = order.ticker
        action = abs(order.volume)
        side = order.side

        if side.sign > 0:
            flag = 'long'
        elif side.sign < 0:
            flag = 'short'
        else:
            return

        if f'max_exposure_{flag}' not in limit:
            return

        working = limit['working']
        exposure = limit['exposure']
        max_exposure = limit[f'max_exposure_{flag}']
        working = working[flag]

        ttl_exposure = exposure['long'] - exposure['short']

        expectation_volume_0 = ttl_exposure + working
        expectation_volume_1 = ttl_exposure + action * side.sign
        expectation_volume_2 = ttl_exposure + action * side.sign + working

        if side.sign > 0:
            if expectation_volume_0 <= max_exposure \
                    and expectation_volume_1 <= max_exposure \
                    and expectation_volume_2 <= max_exposure:
                return True
            else:
                raise self.Risk(
                    risk_type='RiskProfile.ExposureLimit.Long',
                    code=2001,
                    msg=f'{ticker} {side.sign * action} rejected! lmt_exp={max_exposure}, exp={ttl_exposure}, working={working}, action={action}'
                )
        elif side.sign < 0:
            if expectation_volume_0 >= -max_exposure \
                    and expectation_volume_1 >= -max_exposure \
                    and expectation_volume_2 >= -max_exposure:
                return True
            else:
                raise self.Risk(
                    risk_type='RiskProfile.ExposureLimit.Short',
                    code=2002,
                    msg=f'{ticker} {side.sign * action} rejected! lmt_exp={max_exposure}, exp={ttl_exposure}, working={working}, action={-action}'
                )

    def _check_max_percentile(self, order: TradeInstruction, limit: Dict[str, Union[float, Dict[str, float]]]):
        ticker = order.ticker
        action = abs(order.volume)
        side = order.side

        if 'max_percentile' not in limit:
            return

        max_percentile = limit['max_percentile']
        market_price = limit['market_price']
        total_notional = sum([abs(_) for _ in self.balance.exposure_notional(mds=self.mds).values()])

        if np.isfinite(max_percentile) and max_percentile < 1 and np.isfinite(total_notional):
            max_notional = np.divide(total_notional, 1 - max_percentile) * max_percentile
        else:
            return True

        max_position = np.divide(max_notional, market_price)

        working = limit['working']
        exposure = limit['exposure']

        ttl_exposure = exposure['long'] - exposure['short']

        if side.sign > 0:
            working = working['long']
        elif side.sign < 0:
            working = working['short']
        else:
            return True

        expectation_volume_0 = ttl_exposure + working
        expectation_volume_1 = ttl_exposure + action * side.sign
        expectation_volume_2 = ttl_exposure + action * side.sign + working

        if abs(expectation_volume_0) <= max_position \
                and abs(expectation_volume_1) <= max_position \
                and abs(expectation_volume_2) <= max_position:
            return True

        if side.sign > 0:
            raise self.Risk(
                risk_type='RiskProfile.PercentileLimit.Long',
                code=3001,
                msg=f'{ticker} {side.sign * action} rejected! lmt_pct={max_percentile}, lmt_exp={max_position}, exp={ttl_exposure}, working={working}, action={action}'
            )
        elif side.sign < 0:
            raise self.Risk(
                risk_type='RiskProfile.PercentileLimit.Short',
                code=3002,
                msg=f'{ticker} {side.sign * action} rejected! lmt_pct={max_percentile}, lmt_exp={max_position}, exp={ttl_exposure}, working={working}, action={-action}'
            )

    def _check_max_notional(self, order: TradeInstruction, limit: Dict[str, Union[float, Dict[str, float]]]):
        ticker = order.ticker
        action = abs(order.volume)
        side = order.side

        if side.sign > 0:
            flag = 'long'
        elif side.sign < 0:
            flag = 'short'
        else:
            return

        if f'max_notional_{flag}' not in limit:
            return

        market_price = limit['market_price']
        working = limit['working']
        exposure = limit['exposure']
        max_notional = limit[f'max_notional_{flag}']
        working = working[flag]
        ttl_exposure = exposure['long'] - exposure['short']
        max_position = np.divide(max_notional, market_price)

        expectation_volume_0 = ttl_exposure + working
        expectation_volume_1 = ttl_exposure + action * side.sign
        expectation_volume_2 = ttl_exposure + action * side.sign + working

        if side.sign > 0:
            if expectation_volume_0 <= max_position \
                    and expectation_volume_1 <= max_position \
                    and expectation_volume_2 <= max_position:
                return True
            else:
                raise self.Risk(
                    risk_type='RiskProfile.NotionalLimit.Long',
                    code=4001,
                    msg=f'{ticker} {side.sign * action} rejected! lmt_ntl={max_notional}, lmt_exp={max_position}, exp={ttl_exposure}, working={working}, action={action}'
                )
        elif side.sign < 0:
            if expectation_volume_0 >= -max_position \
                    and expectation_volume_1 >= -max_position \
                    and expectation_volume_2 >= -max_position:
                return True
            else:
                raise self.Risk(
                    risk_type='RiskProfile.NotionalLimit.Short',
                    code=4002,
                    msg=f'{ticker} {side.sign * action} rejected! lmt_ntl={max_notional}, lmt_exp={max_position}, exp={ttl_exposure}, working={working}, action={-action}'
                )

    def _check_net_portfolio(self, order: TradeInstruction, limit: Dict[str, Union[float, Dict[str, float]]]):
        ticker = order.ticker
        action = abs(order.volume)
        side = order.side

        if side.sign > 0:
            flag = 'long'
        elif side.sign < 0:
            flag = 'short'
        else:
            return

        if f'max_net_notional_{flag}' not in limit:
            return

        max_net_notional = limit[f'max_net_notional_{flag}']
        market_price = limit['market_price']
        portfolio_working_notional = self.balance.working_notional(mds=self.mds)
        portfolio_exposure_notional = self.balance.exposure_notional(mds=self.mds)

        net_exposure = sum(portfolio_exposure_notional.values())
        net_working = sum(portfolio_working_notional.values())

        expectation_var_0 = net_exposure + net_working
        expectation_var_1 = net_exposure + action * side.sign * market_price
        expectation_var_2 = net_exposure + action * side.sign * market_price + net_working

        if side.sign > 0:
            if expectation_var_0 <= max_net_notional \
                    and expectation_var_1 <= max_net_notional \
                    and expectation_var_2 <= max_net_notional:
                return True

            raise self.Risk(
                risk_type='RiskProfile.NotionalLimit.PortfolioNet.Long',
                code=5001,
                msg=f'{ticker} {side.sign * action} rejected! lmt_ntl={max_net_notional}, net_exp={net_exposure}, net_working={net_working}, action={action}'
            )
        elif side.sign < 0:
            if -max_net_notional <= expectation_var_0 \
                    and -max_net_notional <= expectation_var_1 \
                    and -max_net_notional <= expectation_var_2:
                return True

            raise self.Risk(
                risk_type='RiskProfile.NotionalLimit.PortfolioNet.Short',
                code=5002,
                msg=f'{ticker} {side.sign * action} rejected! lmt_ntl={max_net_notional}, net_exp={net_exposure}, net_working={net_working}, action={action}'
            )

    def _check_ttl_portfolio(self, order: TradeInstruction, limit: Dict[str, Union[float, Dict[str, float]]]):
        ticker = order.ticker
        action = abs(order.volume)
        side = order.side

        if side.sign > 0:
            flag = 'long'
        elif side.sign < 0:
            flag = 'short'
        else:
            return

        if f'max_ttl_notional_{flag}' not in limit:
            return

        market_price = limit['market_price']
        max_notional = limit[f'max_ttl_notional_{flag}']
        working_notional = {'long': 0., 'short': 0.}
        exposure_notional = {'long': 0., 'short': 0.}

        for order_id in list(self.balance.working_order):
            order = self.balance.working_order.get(order_id, None)

            if order is None:
                continue

            if order.side.sign > 0:
                working_notional['long'] += abs(order.working_volume) * market_price
            elif order.side.sign < 0:
                working_notional['short'] += abs(order.working_volume) * market_price

        for ticker, notional in self.balance.exposure_notional(mds=self.mds).items():
            if notional > 0:
                exposure_notional['long'] += abs(notional)
            else:
                exposure_notional['short'] += abs(notional)

        ttl_exposure = exposure_notional[flag]
        ttl_working = working_notional[flag]

        expectation_var_0 = ttl_exposure + ttl_working
        expectation_var_1 = ttl_exposure + action * market_price
        expectation_var_2 = ttl_exposure + action * market_price + ttl_working

        if expectation_var_0 <= max_notional \
                and expectation_var_1 <= max_notional \
                and expectation_var_2 <= max_notional:
            return True

        if side.sign > 0:
            raise self.Risk(
                risk_type='RiskProfile.NotionalLimit.PortfolioTotal.Long',
                code=5003,
                msg=f'{ticker} {side.sign * action} rejected! lmt_ntl={max_notional}, ttl_exp={ttl_exposure}, ttl_working={ttl_working}, action={action}'
            )
        elif side.sign < 0:
            raise self.Risk(
                risk_type='RiskProfile.NotionalLimit.PortfolioTotal.Short',
                code=5004,
                msg=f'{ticker} {side.sign * action} rejected! lmt_ntl={max_notional}, ttl_exp={ttl_exposure}, ttl_working={ttl_working}, action={action}'
            )

    def _get_volume(self, ticker: str, flag: str = 'working') -> Dict[str, float]:
        volume = {'long': 0., 'short': 0.}
        if flag == 'working':
            for order_id in list(self.balance.working_order):
                order = self.balance.working_order.get(order_id, None)

                if order is None or order.ticker != ticker or not order.is_working:
                    continue

                if order.side.sign > 0:
                    volume['long'] += abs(order.working_volume)
                elif order.side.sign < 0:
                    volume['short'] += abs(order.working_volume)
        elif flag == 'exposure':
            for trade_id in list(self.balance.trades):
                trade = self.balance.trades.get(trade_id, None)

                if trade is None or trade.ticker != ticker:
                    continue

                if trade.side.sign > 0:
                    volume['long'] += abs(trade.volume)
                elif trade.side.sign < 0:
                    volume['short'] += abs(trade.volume)
        elif flag == 'traded':
            for trade_id in list(self.balance.trades):
                trade = self.balance.trades.get(trade_id, None)

                if trade is None \
                        or trade.ticker != ticker \
                        or trade.trade_time.date() != self.market_time.date():  # apply to A-Stock when daily inventory is limited
                    continue

                if trade.side.sign > 0:
                    volume['long'] += abs(trade.volume)
                elif trade.side.sign < 0:
                    volume['short'] += abs(trade.volume)
        else:
            raise ValueError(f'Invalid flag {flag}')

        return volume

    @property
    def market_time(self):
        return self.mds.market_time

    @property
    def info(self) -> pd.DataFrame:
        info_dict = defaultdict(dict)

        rules = self.rules.copy()

        for ticker in rules['entry']:
            for key in ['max_percentile', 'max_trade_long', 'max_trade_short', 'max_exposure_long', 'max_exposure_short', 'max_notional_long', 'max_notional_short']:
                if ticker in rules[key]:
                    info_dict[ticker][key] = rules[key][ticker]

        for key in ['max_ttl_notional_long', 'max_ttl_notional_short', 'max_net_notional_long', 'max_net_notional_short']:
            if rules[key] is not None:
                info_dict['global'][key] = rules[key]

        return pd.DataFrame(info_dict).T


class SimMatch(object):
    def __init__(self, ticker, **kwargs):
        self.ticker = ticker
        self.topic_set = kwargs.pop('topic_set', TOPIC)
        self.event_engine = kwargs.pop('event_engine', EVENT_ENGINE)

        self.fee = kwargs.pop('fee', 0.)
        self.working: Dict[str, TradeInstruction] = {}
        self.history: Dict[str, TradeInstruction] = {}

        self.market_time = datetime.datetime.min

    def __call__(self, **kwargs):
        order = kwargs.pop('order', None)
        market_data = kwargs.pop('market_data', None)

        if order is not None:
            if order.order_type == OrderType.LimitOrder:
                self.launch_order(order=order)
            elif order.order_type == OrderType.CancelOrder:
                self.cancel_order(order=order)
            else:
                raise ValueError(f'Invalid order {order}')

        if market_data is not None:
            self.market_time = max(self.market_time, market_data.market_time)

            if isinstance(market_data, BarData):
                self._check_bar_data(market_data=market_data)
            elif isinstance(market_data, TickData):
                self._check_tick_data(market_data=market_data)
            elif isinstance(market_data, TradeData):
                self._check_trade_data(market_data=market_data)
            elif isinstance(market_data, OrderBook):
                self._check_order_book(market_data=market_data)

    def register(self, topic_set=None, event_engine=None):
        if topic_set is not None:
            self.topic_set = topic_set

        if event_engine is not None:
            self.event_engine = event_engine

        self.event_engine.register_handler(topic=self.topic_set.launch_order(ticker=self.ticker), handler=self.launch_order)
        self.event_engine.register_handler(topic=self.topic_set.cancel_order(ticker=self.ticker), handler=self.cancel_order)
        self.event_engine.register_handler(topic=self.topic_set.realtime(ticker=self.ticker), handler=self)

    def unregister(self):
        self.event_engine.unregister_handler(topic=self.topic_set.launch_order(ticker=self.ticker), handler=self.launch_order)
        self.event_engine.unregister_handler(topic=self.topic_set.cancel_order(ticker=self.ticker), handler=self.cancel_order)
        self.event_engine.unregister_handler(topic=self.topic_set.realtime(ticker=self.ticker), handler=self)

    def launch_order(self, order: TradeInstruction, **kwargs):
        if (order.order_id in self.working) or (order.order_id in self.history):
            raise ValueError(f'Invalid instruction {order}, OrderId already in working or history')
        elif order.limit_price is None:
            LOGGER.warning(f'order {order} does not have a valid limit price!')
            # raise ValueError(f'Invalid instruction {order}, instruction must have a LimitPrice')

        order.set_order_state(order_state=OrderState.Placed, market_datetime=self.market_time)
        self.working[order.order_id] = order
        if 'market_time' not in kwargs:
            kwargs['market_time'] = self.market_time
        self.on_order(order=order, **kwargs)

    def cancel_order(self, order: TradeInstruction = None, order_id: str = None, **kwargs):
        if order is None and order_id is None:
            raise ValueError('Must assign a order or order_id to cancel order')
        elif order_id is None:
            order_id = order.order_id

        # if order_id not in self.working:
        #     raise ValueError(f'Invalid cancel order {order}, OrderId not found')

        order: TradeInstruction = self.working.pop(order_id, None)
        if order is None:
            LOGGER.info(f'[{self.market_time:%Y-%m-%d %H:%M:%S}] failed to cancel {order_id} order!')
            return

        if order.order_state == OrderState.Filled:
            pass
        else:
            order.set_order_state(order_state=OrderState.Canceled, market_datetime=self.market_time)
            LOGGER.info(f'[{self.market_time:%Y-%m-%d %H:%M:%S}] Sim-canceled {order.side.name} {order.ticker} order!')

        self.history[order_id] = order
        self.on_order(order=order, **kwargs)

    def _check_bar_data(self, market_data: BarData):
        for order_id in list(self.working):
            order = self.working.get(order_id)
            if order is None:
                pass
            elif order.order_state in [OrderState.Placed, OrderState.PartFilled]:
                if order.side.sign > 0:
                    # match order based on worst offer
                    if order.limit_price is None:
                        self._match(order=order, match_price=market_data.VWAP)
                    elif market_data.high_price < order.limit_price:
                        self._match(order=order, match_price=market_data.high_price)
                    # match order based on limit price
                    elif market_data.low_price < order.limit_price:
                        self._match(order=order, match_price=order.limit_price)
                    # no match
                    else:
                        pass
                elif order.side.sign < 0:
                    # match order based on worst offer
                    if order.limit_price is None:
                        self._match(order=order, match_price=market_data.VWAP)
                    elif market_data.low_price > order.limit_price:
                        self._match(order=order, match_price=market_data.low_price)
                    # match order based on limit price
                    elif market_data.high_price > order.limit_price:
                        self._match(order=order, match_price=order.limit_price)
                    # no match
                    else:
                        pass
            else:
                continue
                # raise ValueError(f'Invalid working order state {order}')

    def _check_trade_data(self, market_data: TradeData):
        for order_id in list(self.working):
            order = self.working.get(order_id)
            if order is None:
                pass
            elif order.is_working:
                if order.start_time > market_data.market_time:
                    pass
                elif order.limit_price is None:
                    if order.side.sign * market_data.side.sign > 0:
                        self._match(order=order, match_volume=market_data.volume, match_price=market_data.market_price)
                elif order.side.sign > 0 and market_data.market_price < order.limit_price:
                    self._match(order=order, match_volume=market_data.volume, match_price=market_data.market_price)
                elif order.side.sign < 0 and market_data.market_price > order.limit_price:
                    self._match(order=order, match_volume=market_data.volume, match_price=market_data.market_price)
            else:
                continue
                # raise ValueError(f'Invalid working order state {order}')

    def _check_order_book(self, market_data: OrderBook):
        for order_id in list(self.working):
            order = self.working.get(order_id)

            match_volume = 0.
            match_notional = 0.

            if order is None:
                pass
            elif order.order_state in [OrderState.Placed, OrderState.PartFilled]:
                if order.limit_price is None:
                    if order.side.sign > 0:
                        for entry in market_data.ask:
                            if match_volume < order.working_volume:
                                addition_volume = min(entry.volume, order.working_volume - match_volume)
                                match_volume += addition_volume
                                match_notional += addition_volume * entry.price
                            else:
                                break
                    else:
                        for entry in market_data.bid:
                            if match_volume < order.working_volume:
                                addition_volume = min(entry.volume, order.working_volume - match_volume)
                                match_volume += addition_volume
                                match_notional += addition_volume * entry.price
                            else:
                                break
                elif order.side.sign > 0 and market_data.best_ask_price <= order.limit_price:
                    for entry in market_data.ask:
                        if entry.price <= order.limit_price:
                            if match_volume < order.working_volume:
                                addition_volume = min(entry.volume, order.working_volume - match_volume)
                                match_volume += addition_volume
                                match_notional += addition_volume * entry.price
                            else:
                                break
                        else:
                            break
                elif order.side.sign < 0 and market_data.best_bid_price >= order.limit_price:
                    for entry in market_data.bid:
                        if entry.price >= order.limit_price:
                            if match_volume < order.working_volume:
                                addition_volume = min(entry.volume, order.working_volume - match_volume)
                                match_volume += addition_volume
                                match_notional += addition_volume * entry.price
                            else:
                                break
                        else:
                            break

                if match_volume:
                    self._match(order=order, match_volume=match_volume, match_price=match_notional / match_volume)
            else:
                continue
                # raise ValueError(f'Invalid working order state {order}')

    def _check_tick_data(self, market_data: TickData):
        return self._check_order_book(market_data=market_data.order_book)

    def _match(self, order: TradeInstruction, match_volume: float = None, match_price: float = None):
        if match_volume is None:
            match_volume = order.working_volume
        elif match_volume > order.working_volume:
            match_volume = order.working_volume

        if order.limit_price is None:
            pass
        elif match_price is None:
            match_price = order.limit_price
        elif order.side.sign > 0 and match_price > order.limit_price:
            LOGGER.warning(f'match price greater than limit price for bid order {order}')
            match_price = order.limit_price
        elif order.side.sign < 0 and match_price < order.limit_price:
            match_price = order.limit_price
            LOGGER.warning(f'match price less than limit price for ask order {order}')

        if match_volume:
            report = TradeReport(
                ticker=order.ticker,
                side=order.side,
                volume=match_volume,
                notional=match_volume * match_price * order.multiplier,
                trade_time=self.market_time,
                order_id=order.order_id,
                price=match_price,
                multiplier=order.multiplier,
                fee=self.fee * match_volume * match_price * order.multiplier
            )

            LOGGER.info(f'[{self.market_time:%Y-%m-%d %H:%M:%S}] Sim-filled {order.ticker} {order.side.name} {report.volume:,.2f} @ {report.price:.2f}')
            order.fill(trade_report=report)

            if order.order_state == OrderState.Filled:
                self.working.pop(order.order_id, None)
                self.history[order.order_id] = order

            self.on_report(report=report)
            self.on_order(order=order)
            return report
        else:
            return None

    def on_order(self, order, **kwargs):
        self.event_engine.put(topic=self.topic_set.on_order, order=order)

    def on_report(self, report, **kwargs):
        self.event_engine.put(topic=self.topic_set.on_report, report=report, **kwargs)

    def eod(self):
        for order_id in list(self.working):
            self.cancel_order(order_id=order_id)

    def clear(self):
        self.fee = 0.
        self.working.clear()
        self.history.clear()
        self.market_time = datetime.datetime.min
