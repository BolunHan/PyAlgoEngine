from __future__ import annotations

import abc
import datetime
import json
import os
import pathlib
import time
import traceback
import uuid
from collections import defaultdict, deque
from enum import Enum
from threading import Thread, Semaphore

import numpy as np
import pandas as pd

from . import LOGGER
from .algo_engine import ALGO_ENGINE, AlgoTemplate
from .market_engine import MarketDataService, Singleton
from ..base import TransactionSide, TransactionDirection as Direction, TransactionOffset as Offset, TradeInstruction, MarketData, OrderState, TradeReport

LOGGER = LOGGER.getChild('TradeEngine')
__all__ = ['DirectMarketAccess', 'PositionManagementService', 'Balance', 'Inventory', 'RiskProfile']


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
    Direct Market Access

    send launch/cancel order direct to market(exchange)

    also contains an order buff designed to process order and control risk

    2 ways to implement this api
    - override the abstractmethod _launch_order_handler, _cancel_order_handler, _reject_order_handler to api directly
    - or use event engine
    """

    def __init__(self, mds: MarketDataService, risk_profile: RiskProfile, cool_down: float = None):
        assert cool_down is None or cool_down > 0, 'Order buff cool down must greater than 0.'

        self.mds = mds
        self.risk_profile = risk_profile
        self.cool_down = cool_down

        self.order_queue = deque()
        self.worker = Thread(target=self._order_buffer)
        self.lock = Semaphore(0)
        self.enabled = False

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

    def _launch_order_buffed(self, order: TradeInstruction, **kwargs):
        self.lock.release()
        self.order_queue.append(('launch', order, kwargs))

    def _cancel_order_buffed(self, order: TradeInstruction, **kwargs):
        self.lock.release()
        self.order_queue.append(('cancel', order, kwargs))

    def _launch_order_no_wait(self, order: TradeInstruction, **kwargs):
        LOGGER.info(f'{self} sent a LAUNCH signal of {order}')

        if not self.enabled:
            LOGGER.warning(f'{order} Rejected by {self}! {self} not enabled!')
            order.set_order_state(order_state=OrderState.Rejected, timestamp=self.mds.timestamp)
            self._reject_order_handler(order=order, **kwargs)
        elif not (is_pass := self.risk_profile.check(order=order)):
            LOGGER.warning(f'{order} Rejected by risk control! Invalid action {order.ticker} {order.side.name} {order.volume}!')
            order.set_order_state(order_state=OrderState.Rejected, timestamp=self.mds.timestamp)
            self._reject_order_handler(order=order, **kwargs)
        else:
            order.set_order_state(order_state=OrderState.Sent, timestamp=self.mds.timestamp)
            self._launch_order_handler(order=order, **kwargs)

    def _cancel_order_no_wait(self, order: TradeInstruction, **kwargs):
        LOGGER.info(f'{self} sent a CANCEL signal of {order}')

        order.set_order_state(order_state=OrderState.Canceling, timestamp=self.mds.timestamp)
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

    def _order_buffer(self):
        while True:
            ts = time.time()
            self.lock.acquire(blocking=True)

            try:
                action, order, kwargs = self.order_queue.popleft()
            except IndexError as e:
                if not self.enabled:
                    break
                else:
                    raise e

            if action == 'launch':
                self._launch_order_no_wait(order=order, **kwargs)
            elif action == 'cancel':
                self._cancel_order_no_wait(order=order, **kwargs)
            else:
                LOGGER.info(f'Invalid order action {action}!')

            if self.cool_down and (cool_down := (ts + self.cool_down - time.time())) > 0:
                time.sleep(cool_down)

            if not self.enabled:
                break

    def start(self):
        if self.enabled:
            LOGGER.error(f'{self} already started!')

        self.enabled = True

        if self.cool_down:
            self.worker.start()

    def shut_down(self):
        if not self.enabled:
            LOGGER.error(f'{self} already stopped!')

        self.enabled = False

        if self.cool_down:
            self.lock.release()
            self.worker = Thread(target=self._order_buffer)
            LOGGER.info(f'Order buff shutting down!')

    @property
    def timestamp(self):
        return self.mds.timestamp

    @property
    def market_price(self):
        return self.mds.market_price

    @property
    def market_time(self):
        return self.mds.market_time


class PositionManagementService(object):
    """
    Position Module controls the position of a single strategy,

    The tracker provides basic tracing of PnL, exposure, holding time and interface with risk monitor module
    The Strategy should interface with Position module, not the algo

    a range of easy method is provided to facilitate development
    """

    def __init__(
            self,
            dma: DirectMarketAccess,
            algo_engine=None,
            default_algo: str = None,
            no_cache: bool = False,
            **kwargs
    ):
        self.dma = dma
        self.algo_engine = algo_engine if algo_engine is not None else ALGO_ENGINE
        self.algo_registry = self.algo_engine.registry
        self.default_algo = self.algo_registry.passive if default_algo is None else default_algo
        self.position_id = kwargs.pop('position_id', uuid.uuid4().hex)
        self.no_cache = no_cache

        self.algos: dict[str, AlgoTemplate] = {}
        self.working_algos: dict[str, AlgoTemplate] = {}

        # cache
        self._exposure: dict[str, float] | None = None
        self._working: dict[str, dict[str, float]] | None = None

    def __call__(self, market_data: MarketData):
        self.on_market_data(market_data=market_data)

    def on_market_data(self, market_data: MarketData):
        for algo_id in list(self.working_algos):
            algo = self.algos.get(algo_id)

            if algo is None:
                continue

            algo.on_market_data(market_data=market_data)

    def on_filled(self, report: TradeReport, **kwargs):
        order_id = report.order_id
        algo = self.reversed_order_mapping.get(order_id)

        if algo is None:
            return 0

        result = algo.on_filled(report=report, **kwargs)
        self._update_status()
        self.clear_cache()
        return result

    def on_canceled(self, order_id: str, **kwargs):
        algo = self.reversed_order_mapping.get(order_id)

        if algo is None:
            return 0

        result = algo.on_canceled(order_id=order_id, **kwargs)
        self._update_status()
        self.clear_cache()
        return result

    def on_rejected(self, order: TradeInstruction, **kwargs):
        order_id = order.order_id
        algo = self.reversed_order_mapping.get(order_id)

        if algo is None:
            return 0

        result = algo.on_rejected(order=order, **kwargs)
        self._update_status()
        self.clear_cache()
        return result

    def on_algo_done(self, algo: AlgoTemplate):
        self.working_algos.pop(algo.algo_id, None)

    def on_algo_error(self, algo: AlgoTemplate):
        self.working_algos.pop(algo.algo_id, None)
        LOGGER.warning(f'{algo} encounter error, manual intervention')

    def open(self, ticker: str, target_volume: float, trade_side: TransactionSide, algo: str = None, **kwargs):
        if algo is None:
            algo = self.default_algo

        if target_volume:
            algo = self.algo_registry.to_algo(name=algo)(
                handler=self,
                ticker=ticker,
                side=trade_side,
                target_volume=target_volume,
                dma=self.dma,
                **kwargs
            )

            LOGGER.debug(f'{algo} opening {ticker} {trade_side.side_name} {target_volume} position!')
            self.algos[algo.algo_id] = self.working_algos[algo.algo_id] = algo

            algo.launch(**kwargs)
            self._update_status()
            return algo

    def unwind_ticker(self, ticker: str, **kwargs):
        LOGGER.info(f'fully cancel and unwind {ticker} position!')

        # cancel all
        for algo_id in list(self.algos):
            algo = self.algos.get(algo_id)

            if algo is not None and ticker == algo.ticker and algo.working_order:
                algo.is_active = False
                algo.cancel(**kwargs)

        # calculate exposure
        exposure = self.exposure_volume.get(ticker)
        working = self.working_volume.get(ticker, {})
        working_long = working.get('Long', 0)
        working_short = working.get('Short', 0)

        if not exposure:
            LOGGER.info(f'No exposure for {ticker}, no unwind actions!')
            # no exposure, good!
            return
        elif working_long and working_short:
            # with exposure, and working orders on both side, no action
            LOGGER.info(f'Multiple trade actions for {ticker}, skip unwind actions! Try again later!')
            return
        elif (exposure > 0 and working_short) or (exposure < 0 and working_long):
            # with exposure, and working unwinding orders, no action
            LOGGER.info(f'Unwinding actions exists for {ticker}, skip unwind actions! Try again later!')
            return

        to_unwind = abs(exposure)
        side = (Direction.DIRECTION_SHORT if exposure > 0 else Direction.DIRECTION_LONG) | Offset.OFFSET_CLOSE
        self.open(ticker=ticker, target_volume=to_unwind, trade_side=side)

    def add_exposure(self, ticker: str, volume: float, notional: float, side: TransactionSide, timestamp: float):
        """
        this is a method to add dummy algo and fills it.

        the method provides an easy way to amend exposure
        """

        algo = self.algo_registry.to_algo(name=self.algo_registry.passive)(
            handler=self,
            ticker=ticker,
            side=side,
            target_volume=volume,
            dma=None,
        )
        self.algos[algo.algo_id] = algo

        order = TradeInstruction(ticker=ticker, order_id=f'Dummy.{uuid.uuid4().int}', volume=volume, side=side, timestamp=timestamp)
        report = TradeReport(ticker=ticker, volume=volume, price=notional / volume if volume else np.nan, notional=notional, side=side, timestamp=timestamp, order_id=order.order_id)
        order.fill(report)
        algo.status = algo.Status.done
        algo.order[order.order_id] = order
        self._update_status()

        return report

    def unwind_all(self, **kwargs):
        exposure = self.exposure_volume
        additional_kwargs = kwargs.copy()

        for ticker in exposure:
            self.unwind_ticker(ticker, **additional_kwargs)

        return 0.

    def cancel_all(self, **kwargs):
        # EMERGENCY ONLY
        for algo_id in list(self.working_algos):
            algo = self.algos.get(algo_id)

            if algo is not None:
                algo.cancel(**kwargs)

        return 0

    def to_json(self, fmt='str') -> str | dict:
        json_dict = {}
        map_id = self.position_id

        json_dict[map_id] = {}

        # dump algos
        for algo_id in list(self.algos):
            algo = self.algos.get(algo_id)

            if algo is not None:
                json_dict[map_id][algo_id] = algo.to_json(fmt='dict')

        if fmt == 'dict':
            return json_dict
        else:
            return json.dumps(json_dict)

    def _update_status(self):
        for algo_id in list(self.working_algos):
            algo = self.algos.get(algo_id)

            if algo is None:
                continue

            if algo.status == algo.Status.closed or algo.status == algo.Status.done:
                self.on_algo_done(algo=algo)
            elif algo.status == algo.Status.rejected or algo.status == algo.Status.error:
                self.on_algo_error(algo=algo)

    def _algo_pnl(self, algo: AlgoTemplate):
        if algo.exposure_volume:
            if (market_price := self.market_price.get(algo.ticker)) is not None:
                pnl = market_price * algo.exposure_volume * algo.multiplier + algo.cash_flow
            else:
                pnl = np.nan
        else:
            pnl = algo.cash_flow
        return pnl

    def clear_cache(self):
        self._exposure = None
        self._working = None

    def clear(self):
        self.algos.clear()
        self.working_algos.clear()
        self.clear_cache()

    def pnl(self) -> dict[str, float]:
        pnl = {}
        for algo_id in list(self.algos):
            algo = self.algos.get(algo_id)

            if algo is None:
                continue

            ticker = algo.ticker
            pnl[ticker] = self._algo_pnl(algo=algo) + pnl.get(ticker, 0)

        return pnl

    @property
    def notional(self) -> dict[str, float]:
        notional = {}
        for algo_id in list(self.algos):
            algo = self.algos.get(algo_id)

            if algo is None:
                continue

            ticker = algo.ticker
            notional[ticker] = algo.filled_notional + notional.get(ticker, 0)

        return notional

    @property
    def working_volume(self) -> dict[str, dict[str, float]]:
        """
        a dictionary indicating current working volume of all orders

        {'Long': +float, 'Short': +float}

        :return: a dict with non-negative numbers
        """

        if not self.no_cache and self._working is not None:
            return self._working

        working_long = {}
        working_short = {}
        working = {'Long': working_long, 'Short': working_short}

        for algo_id in list(self.working_algos):
            algo = self.algos.get(algo_id)
            ticker = algo.ticker

            if algo is not None:
                if algo.side.sign > 0:
                    working_long[ticker] = working_long.get(ticker, 0.) + algo.working_volume
                elif algo.side.sign < 0:
                    working_short[ticker] = working_short.get(ticker, 0.) + algo.working_volume

        for side in working:
            _ = working[side]

            for ticker in list(_):
                if not _[ticker]:
                    _.pop(ticker)

        return working

    @property
    def exposure_volume(self) -> dict[str, float]:
        """
        a dictionary indicating current net exposed volume of all orders

        :return: a dict with float numbers (positive and negatives)
        """

        if not self.no_cache and self._exposure is not None:
            return self._exposure

        exposure = {}

        for algo_id in list(self.algos):
            algo = self.algos.get(algo_id)

            if algo is not None:
                ticker = algo.ticker
                exposure[ticker] = exposure.get(ticker, 0.) + algo.exposure_volume

        for ticker in list(exposure):
            if not exposure[ticker]:
                exposure.pop(ticker)

        return exposure

    @property
    def working_volume_net(self) -> dict[str, float]:
        """
        a dictionary indicating current working volume of all orders

        :return: a dict with summed working volume for each ticker numbers, with positive value as net-long and negative value as net-short
        """
        working = {}

        for algo_id in list(self.algos):
            algo = self.algos.get(algo_id)

            if algo is not None:
                ticker = algo.ticker
                working[ticker] = working.get(ticker, 0.) + algo.working_volume * algo.side.sign

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
    def orders(self) -> dict[str, TradeInstruction]:
        orders = {}

        for algo_id in list(self.algos):
            algo = self.algos.get(algo_id)

            if algo is None:
                continue

            orders.update(algo.order)

        return orders

    @property
    def working_order(self) -> dict[str, TradeInstruction]:
        working_order = {}

        for algo_id in list(self.algos):
            algo = self.algos.get(algo_id)

            if algo is None:
                continue

            working_order.update(algo.working_order)

        return working_order

    @property
    def trades(self) -> dict[str, TradeReport]:
        trades = {}

        for algo_id in list(self.algos):
            algo = self.algos.get(algo_id)

            if algo is None:
                continue

            trades.update(algo.trades)

        return trades

    @property
    def order_mapping(self) -> dict[str, dict[str, TradeInstruction]]:
        order_mapping = {}

        for algo_id in list(self.algos):
            algo = self.algos.get(algo_id)

            if algo is None:
                continue

            order_mapping[algo.algo_id] = algo.order

        return order_mapping

    @property
    def reversed_order_mapping(self) -> dict[str, AlgoTemplate]:
        reversed_order_mapping = {}

        for algo_id in list(self.algos):
            algo = self.algos.get(algo_id)

            if algo is None:
                continue

            for order_id in list(algo.order):
                reversed_order_mapping[order_id] = algo

        return reversed_order_mapping


class Balance(object, metaclass=Singleton):
    """
    Balance handles mapping of PositionTracker <-> Strategy
    """

    def __init__(self, inventory: Inventory = None):
        self.inventory = inventory if inventory is not None else Inventory()

        self.strategy = {}
        self.trade_logs: list[TradeReport] = []
        self.position_tracker: dict[str, PositionManagementService] = {}

        self.last_update_timestamp = None

    def __repr__(self):
        return f'<Balance>{{id={id(self)}}}'

    def add(self, map_id: str = None, strategy=None, position_tracker: PositionManagementService = None):
        if strategy is None and position_tracker is None:
            raise ValueError('Must assign ether strategy or position_tracker')

        if map_id is None:
            map_id = uuid.uuid4().hex

        if strategy is not None:
            self.strategy[map_id] = strategy

        if position_tracker is not None:
            self.position_tracker[map_id] = position_tracker
        else:
            try:
                position_tracker = strategy.position_tracker
                self.position_tracker[map_id] = position_tracker
            except Exception as _:
                LOGGER.error(traceback.format_exc())

    def pop(self, map_id: str):
        self.strategy.pop(map_id, None)
        self.position_tracker.pop(map_id, None)

    def get(self, **kwargs) -> PositionManagementService | None:
        map_id: str | None = kwargs.pop('map_id', None)
        strategy = kwargs.pop('strategy', None)

        if map_id is not None:
            map_id: str
            return self.position_tracker.get(map_id)
        elif strategy is not None:
            map_id = self.reversed_strategy_mapping.get(id(strategy))

            if map_id is None:
                raise KeyError(f'Can not found strategy {strategy}')
            return self.position_tracker.get(map_id)
        else:
            raise TypeError('Must assign one value of map_id, strategy or position_tracker')

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

    def get_tracker(self, strategy_name: str = None, strategy_id=None) -> PositionManagementService | None:
        strategy = self.get_strategy(strategy_name=strategy_name, strategy_id=strategy_id)

        if strategy is None:
            return None

        map_id = self.reversed_strategy_mapping.get(id(strategy))
        tracker = self.position_tracker.get(map_id)
        return tracker

    def on_update(self, market_time=None):
        pass
        # step 0: update market time
        # self.last_update_timestamp = time.time() if market_time is None else market_time

        # step 1: write balance file
        # self.dump(file_path=pathlib.Path(WORKING_DIRECTORY).joinpath('Dumps', 'balance.updated.json'))

        # step 2: write trade file
        # self.dump_trades(file_path=pathlib.Path(WORKING_DIRECTORY).joinpath('Dumps', 'trades.updated.csv'))

    def on_order(self, order: TradeInstruction, **kwargs):
        order_id = order.order_id
        order_state = order.order_state
        status_code = 0

        for position_id in list(self.position_tracker):
            position_tracker = self.position_tracker.get(position_id)

            if position_tracker is None:
                continue

            if order_id in position_tracker.working_order:
                if position_tracker.working_order[order_id] is not order:
                    LOGGER.error(f'Order object not static! stored id {id(position_tracker.working_order[order_id])}, updated id {id(order)}')

                if order_state == OrderState.Canceled:
                    position_tracker.on_canceled(order_id=order_id, **kwargs)
                elif order_state == OrderState.Rejected:
                    position_tracker.on_rejected(order=order, **kwargs)

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

        for position_id in list(self.position_tracker):
            position_tracker = self.position_tracker.get(position_id)

            if position_tracker is None:
                continue

            if order_id in position_tracker.working_order:
                position_tracker.on_filled(report=report, **kwargs)

                status_code = 1
                break

        if not status_code:
            LOGGER.warning(f'No match for report {report}')

        self.on_update()
        self.trade_logs.append(report)
        return status_code

    def reset(self):
        self.position_tracker.clear()
        self.strategy.clear()
        self.trade_logs.clear()

    def to_json(self, fmt='str') -> str | dict:
        json_dict = {}

        for map_id in self.position_tracker:
            tracker = self.position_tracker.get(map_id)

            if tracker is not None:
                json_dict.update(tracker.to_json(fmt='dict'))

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

        for map_id in json_dict:
            if map_id not in self.strategy:
                LOGGER.error(f'No strategy with key {map_id} found! Must register strategy before loading balance!')
                continue

            pos_tracker = self.position_tracker[map_id]
            algo_json = json_dict[map_id]

            for algo_id in algo_json:
                algo_dict = algo_json[algo_id]
                algo = pos_tracker.algo_engine.from_json(algo_dict)
                pos_tracker.algos[algo.algo_id] = pos_tracker.working_algos[algo.algo_id] = algo

                if algo.status == algo.Status.closed or algo.status == algo.Status.done:
                    pos_tracker.on_algo_done(algo=algo)
                elif algo.status == algo.Status.rejected or algo.status == algo.Status.error:
                    pos_tracker.on_algo_error(algo=algo)

        return self

    def dump(self, file_path: str | pathlib.Path):
        file_path = pathlib.Path(file_path)
        dump_dir = file_path.parent

        os.makedirs(dump_dir, exist_ok=True)

        with open(file_path, 'w') as f:
            f.write(json.dumps(self.to_json(fmt='dict'), indent=4, sort_keys=True))

    def dump_trades(self, file_path: pathlib.Path | str = None, ts_from: float = None, ts_to: float = None) -> dict:
        """
        export all trade monitored by position manager

        :param file_path: Optional, the exported path, without it, the dict will not be dumped
        :param ts_from: timestamp from
        :param ts_to: timestamp to
        :return: a dict containing all the trades
        """
        trades_dict = {}

        for mapping_id in self.position_tracker:
            tracker = self.position_tracker[mapping_id]
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
                    strategy=mapping_id,
                    ticker=report.ticker,
                    side=report.side.side_name,
                    volume=report.volume,
                    price=report.price,
                    notional=report.notional,
                    time=report.trade_time,
                    ts=report.timestamp,
                )

        if file_path and trades_dict:
            trades_df = pd.DataFrame(trades_dict).T
            trades_df.sort_values('ts')
            trades_df.to_csv(file_path)

        return trades_dict

    def dump_trades_all(self, file_path: pathlib.Path | str = None, ts_from: float = None, ts_to: float = None) -> list:
        """
        export all the trades received by Balance module, even if there is no strategy corresponding to it.

        :param file_path: Optional, the exported path, without it, the dict will not be dumped
        :param ts_from: timestamp from
        :param ts_to: timestamp to
        :return: a list containing all the trades info
        """
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

        if file_path and trade_logs:
            trades_df = pd.DataFrame(trade_logs)
            trades_df.sort_values('ts')
            trades_df.to_csv(file_path)

        return trade_logs

    def load(self, file_path: str | pathlib.Path):
        if not os.path.isfile(file_path):
            LOGGER.error(f'No such file {file_path}')
            return

        with open(file_path, 'r') as f:
            json_str = f.read()

        self.from_json(json_str)

    @property
    def tracker_mapping(self) -> dict[str, str]:
        mapping = {}

        for map_id in self.position_tracker:
            tracker = self.position_tracker.get(map_id)

            if tracker is None:
                continue

            mapping[map_id] = tracker.position_id

        return mapping

    @property
    def reversed_tracker_mapping(self) -> dict[str, str]:
        mapping = {}

        for id_0, id_1 in self.tracker_mapping.items():
            mapping[id_1] = id_0

        return mapping

    @property
    def strategy_mapping(self) -> dict[str, int]:
        mapping = {}

        for map_id in self.strategy:
            strategy = self.strategy.get(map_id)

            if strategy is None:
                continue

            mapping[map_id] = id(strategy)

        return mapping

    @property
    def reversed_strategy_mapping(self) -> dict[int, str]:
        mapping = {}

        for id_0, id_1 in self.strategy_mapping.items():
            mapping[id_1] = id_0

        return mapping

    @property
    def working_volume_summed(self) -> dict[str, float]:
        working_summed = {}

        for tracker_id in list(self.position_tracker):
            tracker = self.position_tracker.get(tracker_id)

            if tracker is not None:
                for side in tracker.working_volume:
                    working = tracker.working_volume[side]

                    for ticker in working:
                        working_summed[ticker] = working_summed.get(ticker, 0.) + abs(working.get(ticker, 0.))

        for ticker in list(working_summed):
            if not working_summed[ticker]:
                working_summed.pop(ticker)

        return working_summed

    @property
    def exposure_volume(self) -> dict[str, float]:
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
    def working_volume(self) -> dict[str, dict[str, float]]:

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

    def exposure_notional(self, mds) -> dict[str, float]:
        notional = {}

        for ticker in self.exposure_volume:
            notional[ticker] = self.exposure_volume.get(ticker, 0.) * mds.market_price.get(ticker, 0)

        return notional

    def working_notional(self, mds) -> dict[str, float]:
        notional = {}

        for ticker in (tracker_working := self.working_volume_summed):
            notional[ticker] = tracker_working[ticker] * mds.market_price.get(ticker, 0)

        return notional

    @property
    def orders(self) -> dict[str, TradeInstruction]:
        orders = {}

        for tracker_id in list(self.position_tracker):
            tracker = self.position_tracker.get(tracker_id)

            if tracker is None:
                continue

            orders.update(tracker.orders)

        return orders

    @property
    def working_order(self) -> dict[str, TradeInstruction]:
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
        from .market_engine import MDS

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
    def trades_session(self) -> dict[str, TradeReport]:
        trades = {_.trade_id: _ for _ in self.trade_logs}

        return trades

    @property
    def trades(self) -> dict[str, TradeReport]:
        return self.trades_today

    @property
    def info(self) -> pd.DataFrame:
        info_dict = {
            'exposure': self.exposure_volume,
            'working_lone': self.working_volume['Long'],
            'working_short': self.working_volume['Short'],
        }

        return pd.DataFrame(info_dict).fillna(0)


class Inventory(object, metaclass=Singleton):
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
        def __init__(self, ticker: str, volume: float, price: float, security_type: Inventory.SecurityType, direction: Direction, **kwargs):
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
            return f'<Inventory.Entry>(ticker={self.ticker}, side={self.direction.name}, volume={self.volume:,}, fee={self.fee:.2f})'

        def __add__(self, other):
            if isinstance(other, self.__class__):
                return self.merge(other)

            raise TypeError(f'Can only merge type {self.__class__.__name__}')

        def __bool__(self):
            return self.volume.__bool__()

        def apply_cash_dividend(self, dividend: Inventory.CashDividend):
            raise NotImplementedError()

        def apply_stock_dividend(self, dividend: Inventory.StockDividend):
            raise NotImplementedError()

        def apply_conversion(self, stock_conversion: Inventory.StockConversion):
            raise NotImplementedError()

        def apply_split(self, stock_split: Inventory.StockSplit):
            raise NotImplementedError()

        def merge(self, entry: Inventory.Entry, inplace=False, **kwargs):
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

        def to_json(self, fmt='str') -> str | dict:
            json_dict = dict(
                ticker=self.ticker,
                volume=self.volume,
                price=self.price,
                security_type=self.security_type.value,
                direction=self.direction.value,
                notional=self.notional,
                fee=self.fee,
                recalled=self.recalled
            )

            if fmt == 'dict':
                return json_dict
            else:
                return json.dumps(json_dict)

        @classmethod
        def from_json(cls, json_str: str | dict):
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
                security_type=Inventory.SecurityType(json_dict['security_type']),
                direction=Direction(json_dict['direction']),
                notional=json_dict['notional'],
                fee=json_dict.get('fee', 0.),
                recalled=json_dict.get('recalled', 0.),
            )

            return entry

        @property
        def available(self):
            return max(self.volume - self.recalled, 0.)

    def __init__(self):
        self._inv: dict[str, list[Inventory.Entry]] = {}
        self._traded: dict[str, float] = {}
        self._tickers = set()

    def __repr__(self):
        return f'<Inventory>{{id={id(self)}}}'

    def __call__(self, ticker: str):
        return dict(
            Long=self.available_volume(ticker=ticker, direction=Direction.DIRECTION_LONG),
            Short=self.available_volume(ticker=ticker, direction=Direction.DIRECTION_SHORT)
        )

    def recall(self, ticker: str, volume: float, direction: Direction = Direction.DIRECTION_LONG):
        key = f'{ticker}.{direction.name}'
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
        key = f'{entry.ticker}.{entry.direction.name}'
        _ = self._inv.get(key, [])

        _.append(entry)

        self._inv[key] = _

    def get_inv(self, ticker: str, direction: Direction = Direction.DIRECTION_LONG) -> Entry | None:
        key = f'{ticker}.{direction.name}'
        _ = self._inv.get(key, [])

        merged_entry = None
        for entry in _:
            if merged_entry is None:
                merged_entry = entry
            else:
                merged_entry = merged_entry + entry

        return merged_entry

    def use_inv(self, ticker: str, volume: float, direction: Direction = Direction.DIRECTION_LONG):
        key = f'{ticker}.{direction.name}'

        self._traded[key] = self._traded.get(key, 0.) + volume

    def available_volume(self, ticker: str, direction: Direction = Direction.DIRECTION_LONG) -> float:
        inv = self.get_inv(ticker=ticker, direction=direction)

        if inv is None:
            return 0.

        used = self._traded.get(ticker, 0.)
        return inv.available - used

    def clear(self):
        self._inv.clear()
        self._traded.clear()
        self._tickers.clear()

    def to_json(self, fmt='str') -> str | dict:
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

    def from_json(self, json_str: str | dict, with_used=False):
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

    def dump(self, file_path: str | pathlib.Path):
        file_path = pathlib.Path(file_path)
        dump_dir = file_path.parent

        os.makedirs(dump_dir, exist_ok=True)

        with open(file_path, 'w') as f:
            f.write(json.dumps(self.to_json(fmt='dict'), indent=4, sort_keys=True))

    def to_csv(self, file_path: str | pathlib.Path):
        inv_dict = {'inv_l': {}, 'inv_s': {}}

        for ticker in self._inv:
            if (long_inv := self.get_inv(ticker=ticker, direction=Direction.DIRECTION_LONG)) is not None:
                inv_dict['inv_l'][ticker] = long_inv.volume

            if (short_inv := self.get_inv(ticker=ticker, direction=Direction.DIRECTION_SHORT)) is not None:
                inv_dict['inv_s'][ticker] = short_inv.volume

        inv_df = pd.DataFrame(inv_dict)
        inv_df.to_csv(file_path)

    def load(self, file_path: str | pathlib.Path, with_used=False):
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
            inv_l = self.get_inv(ticker=ticker, direction=Direction.DIRECTION_LONG)
            inv_s = self.get_inv(ticker=ticker, direction=Direction.DIRECTION_SHORT)

            if inv_l is not None:
                info_dict['inv_l'][ticker] = inv_l.volume

            if inv_s is not None:
                info_dict['inv_s'][ticker] = inv_s.volume

        return pd.DataFrame(info_dict)


class RiskProfile(object, metaclass=Singleton):
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

    def get(self, ticker: str) -> dict[str, float | dict[str, float]]:
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
            volume=volume,
            timestamp=self.mds.timestamp
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

    def to_json(self, fmt='str') -> str | dict:
        json_dict = dict(self.rules)
        json_dict['entry'] = list(json_dict['entry'])

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

        self.rules.update(json_dict)
        self.rules['entry'] = set(self.rules['entry'])

        return self

    def dump(self, file_path: str | pathlib.Path):
        file_path = pathlib.Path(file_path)
        dump_dir = file_path.parent

        os.makedirs(dump_dir, exist_ok=True)

        with open(file_path, 'w') as f:
            f.write(json.dumps(self.to_json(fmt='dict'), indent=4, sort_keys=True))

    def load(self, file_path: str | pathlib.Path):
        if not os.path.isfile(file_path):
            LOGGER.error(f'No such file {file_path}')
            return

        with open(file_path, 'r') as f:
            json_str = f.read()

        self.from_json(json_str)

    def _check_validity(self, order: TradeInstruction, limit: dict[str, float | dict[str, float]]):
        ticker = order.ticker
        market_price = limit['market_price']

        if market_price is None:
            raise self.Risk(
                risk_type='RiskProfile.Internal.Price',
                code=100,
                msg=f'no valid market price for ticker {ticker}'
            )

        return True

    def _check_max_trade(self, order: TradeInstruction, limit: dict[str, float | dict[str, float]]):
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

    def _check_max_exposure(self, order: TradeInstruction, limit: dict[str, float | dict[str, float]]):
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

    def _check_max_percentile(self, order: TradeInstruction, limit: dict[str, float | dict[str, float]]):
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

    def _check_max_notional(self, order: TradeInstruction, limit: dict[str, float | dict[str, float]]):
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

    def _check_net_portfolio(self, order: TradeInstruction, limit: dict[str, float | dict[str, float]]):
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

    def _check_ttl_portfolio(self, order: TradeInstruction, limit: dict[str, float | dict[str, float]]):
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

    def _get_volume(self, ticker: str, flag: str = 'working') -> dict[str, float]:
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
