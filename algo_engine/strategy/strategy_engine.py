import abc
import datetime
import time
from collections.abc import Callable
from functools import cached_property

from . import LOGGER
from ..backtest import SimMatch, ProgressReplay
from ..base import MarketData, TradeReport, TradeInstruction, TransactionSide, TransactionDirection as Direction, TransactionOffset as Offset
from ..engine import PositionManagementService, TOPIC, EVENT_ENGINE

LOGGER = LOGGER.getChild('Strategy')


class StrategyEngineTemplate(object, metaclass=abc.ABCMeta):
    def __init__(self, position_tracker: PositionManagementService):
        self.position_tracker = position_tracker

    def __call__(self, **kwargs):
        if 'market_data' in kwargs:
            self.on_market_data(market_data=kwargs['market_data'])

    @abc.abstractmethod
    def on_market_data(self, market_data: MarketData, **kwargs): ...

    @abc.abstractmethod
    def on_report(self, report: TradeReport, **kwargs): ...

    @abc.abstractmethod
    def on_order(self, order: TradeInstruction, **kwargs): ...

    @property
    def mds(self):
        return self.position_tracker.dma.mds

    @property
    def dma(self):
        return self.position_tracker.dma

    @property
    def risk_profile(self):
        return self.position_tracker.dma.risk_profile

    @property
    def balance(self):
        return self.position_tracker.dma.risk_profile.balance

    @property
    def inventory(self):
        return self.position_tracker.dma.risk_profile.balance.inventory

    @cached_property
    def lock(self):
        from . import REPLAY_LOCK
        return REPLAY_LOCK


class StrategyEngine(StrategyEngineTemplate):
    def __init__(self, position_tracker: PositionManagementService, **kwargs):
        super().__init__(position_tracker=position_tracker)

        self.event_engine = kwargs.pop('event_engine', EVENT_ENGINE)
        self.topic_set = kwargs.pop('topic_set', TOPIC)

        self._on_market_data = []
        self._on_report = []
        self._on_order = []
        self._on_eod = []
        self._on_bod = []
        self.subscription = set()

        self.attach_strategy(strategy=kwargs.pop('strategy', None))
        self.add_handler(**kwargs)

    def __call__(self, **kwargs):
        if 'market_data' in kwargs:
            self.on_market_data(market_data=kwargs['market_data'])

        if self.lock.locked():
            self.lock.release()

    def add_handler(self, **kwargs):
        if 'on_market_data' in kwargs:
            self._on_market_data.append(kwargs['on_market_data'])

        if 'on_report' in kwargs:
            self._on_report.append(kwargs['on_report'])

        if 'on_order' in kwargs:
            self._on_order.append(kwargs['on_order'])

        if 'on_eod' in kwargs:
            self._on_eod.append(kwargs['on_eod'])

        if 'on_bod' in kwargs:
            self._on_bod.append(kwargs['on_bod'])

    def remove_handler(self, **kwargs):
        if 'on_market_data' in kwargs:
            self._on_market_data.remove(kwargs['on_market_data'])

        if 'on_report' in kwargs:
            self._on_report.remove(kwargs['on_report'])

        if 'on_order' in kwargs:
            self._on_order.remove(kwargs['on_order'])

        if 'on_eod' in kwargs:
            self._on_eod.remove(kwargs['on_eod'])

        if 'on_bod' in kwargs:
            self._on_bod.remove(kwargs['on_bod'])

    def add_handler_safe(self, **kwargs):
        if 'on_market_data' in kwargs:
            if (handler := kwargs['on_market_data']) in self._on_market_data:
                LOGGER.warning(f'on_market_data handler {handler} already registered, skipped!')
            else:
                self._on_market_data.append(handler)

        if 'on_report' in kwargs:
            if (handler := kwargs['on_report']) in self._on_report:
                LOGGER.warning(f'on_report handler {handler} already registered, skipped!')
            else:
                self._on_report.append(handler)

        if 'on_order' in kwargs:
            if (handler := kwargs['on_order']) in self._on_order:
                LOGGER.warning(f'on_order handler {handler} already registered, skipped!')
            else:
                self._on_order.append(handler)

        if 'on_eod' in kwargs:
            if (handler := kwargs['on_eod']) in self._on_eod:
                LOGGER.warning(f'on_eod handler {handler} already registered, skipped!')
            else:
                self._on_eod.append(handler)

        if 'on_bod' in kwargs:
            if (handler := kwargs['on_bod']) in self._on_bod:
                LOGGER.warning(f'on_bod handler {handler} already registered, skipped!')
            else:
                self._on_bod.append(handler)

    def remove_handler_safe(self, **kwargs):
        if 'on_market_data' in kwargs:
            if (handler := kwargs['on_market_data']) in self._on_market_data:
                self._on_market_data.remove(handler)

        if 'on_report' in kwargs:
            if (handler := kwargs['on_report']) in self._on_report:
                self._on_report.remove(handler)

        if 'on_order' in kwargs:
            if (handler := kwargs['on_order']) in self._on_order:
                self._on_order.remove(handler)

        if 'on_eod' in kwargs:
            if (handler := kwargs['on_eod']) in self._on_eod:
                self._on_eod.remove(handler)

        if 'on_bod' in kwargs:
            if (handler := kwargs['on_bod']) in self._on_bod:
                self._on_bod.remove(handler)

    def attach_strategy(self, strategy: object):
        if callable(handler := getattr(strategy, 'on_market_data', None)):
            self._on_market_data.append(handler)

        if callable(handler := getattr(strategy, 'on_report', None)):
            self._on_report.append(handler)

        if callable(handler := getattr(strategy, 'on_order', None)):
            self._on_order.append(handler)

        if callable(handler := getattr(strategy, 'on_eod', None)):
            self._on_eod.append(handler)

        if callable(handler := getattr(strategy, 'on_bod', None)):
            self._on_bod.append(handler)

    def subscribe(self, ticker: str):
        self.subscription.add(ticker)

    def on_market_data(self, market_data: MarketData, **kwargs):

        if market_data.ticker not in self.subscription:
            return

        for handler in self._on_market_data:
            handler(market_data=market_data, **kwargs)

    def on_report(self, report: TradeReport, **kwargs):

        for handler in self._on_report:
            handler(report=report, **kwargs)

    def on_order(self, order: TradeInstruction, **kwargs):

        for handler in self._on_order:
            handler(order=order, **kwargs)

    def register(self, event_engine=None, topic_set=None, auto_register: bool = True):
        if event_engine is None:
            event_engine = self.event_engine

        if topic_set is None:
            topic_set = self.topic_set

        if auto_register:
            event_engine.register_handler(topic=topic_set.realtime, handler=self.mds)
            event_engine.register_handler(topic=topic_set.realtime, handler=self.position_tracker.on_market_data)
            event_engine.register_handler(topic=topic_set.on_order, handler=self.balance.on_order)
            event_engine.register_handler(topic=topic_set.on_report, handler=self.balance.on_report)

        event_engine.register_handler(topic=topic_set.realtime, handler=self.__call__)
        event_engine.register_handler(topic=topic_set.on_order, handler=self.on_order)
        event_engine.register_handler(topic=topic_set.on_report, handler=self.on_report)

    def unregister(self, event_engine=None, topic_set=None, auto_unregister: bool = True):
        if event_engine is None:
            event_engine = self.event_engine

        if topic_set is None:
            topic_set = self.topic_set

        if auto_unregister:
            event_engine.unregister_handler(topic=topic_set.realtime, handler=self.mds)
            event_engine.unregister_handler(topic=topic_set.realtime, handler=self.position_tracker.on_market_data)
            event_engine.unregister_handler(topic=topic_set.on_order, handler=self.balance.on_order)
            event_engine.unregister_handler(topic=topic_set.on_report, handler=self.balance.on_report)

        event_engine.unregister_handler(topic=topic_set.realtime, handler=self.__call__)
        event_engine.unregister_handler(topic=topic_set.on_order, handler=self.on_order)
        event_engine.unregister_handler(topic=topic_set.on_report, handler=self.on_report)

    def cancel(self, ticker: str, side: TransactionSide = None, algo_id: str = None, order_id: str = None, **kwargs):
        position_tracker = self.position_tracker

        if algo_id is not None:
            algo_id = position_tracker.reversed_order_mapping.get(order_id).algo_id
            if algo_id:
                LOGGER.info(f'No algo_id specified, found algo {algo_id} associated with order_id {order_id}! Canceling all trade action associated with algo')
                LOGGER.warning('Strategy should not cancel single trade order, this will break the algo_engine Consistency!')

        if not algo_id:
            LOGGER.warning(f'No algo_id given! Canceling all {ticker} {side.side_name} algos!')

            for _algo_id in list(self.algos):
                algo = self.algos.get(_algo_id)

                if algo is None:
                    continue

                if algo.ticker == ticker and algo.side.sign == side.sign:
                    algo.cancel(**kwargs)
        else:
            algo = self.algos.get(algo_id)

            if algo is None:
                LOGGER.error(f'{self} have no algo with algo_id {algo_id}! Cancel signal ignored! Manual intervention required!')
                return

            if algo.ticker == ticker and algo.side.sign == side.sign:
                algo.cancel(**kwargs)

    def stop(self):
        LOGGER.debug(f'All algo should be self-deactivated on cancel, to be sure {self} will deactivate all the algos!')
        for algo_id in list(self.algos):
            algo = self.algos.get(algo_id)

            if algo is None:
                continue

            algo.is_active = False

        LOGGER.info(f'{self} canceling all the algos')
        for ticker in self.subscription:
            self.cancel(ticker=ticker)

    def unwind_pos(self, ticker: str, volume: float, side: TransactionSide = None, limit_price: float = None, algo: str = None, safe=True, **kwargs) -> tuple[float, float]:
        """
        unwind method provide a safe way to unwind position of given ticker.

        :param ticker: the given exposure
        :param volume: the target unwinding volume, should be a positive number
        :param side: the trade action side, e.g. if strategy wishes to sell (in order to unwind long position), then side = TransactionSide.Sell_to_Unwind
        :param limit_price: Optional, a limit price
        :param algo: Optional the algo to be used to execute unwinding action
        :param safe: True -> unwind volume should not exceed the exposed volume; False -> can flip position. Default is safe=True
        :param kwargs: other kwargs passing to `algo.launch`
        :return: executed volume, remaining volume
        """
        position_tracker = self.position_tracker
        exposure = position_tracker.exposure_volume.get(ticker, 0.)
        working_long = position_tracker.working_volume['Long'].get(ticker, 0.)
        working_short = position_tracker.working_volume['Short'].get(ticker, 0.)
        executed, remains = 0., volume

        if not exposure:
            LOGGER.warning(f'{self} found no {ticker} exposure! Unwind signal ignored! Check PositionManagementService!')
            return executed, remains

        if side is not None and exposure * side.sign > 0:
            LOGGER.warning(f'{self} found {ticker} exposure {exposure}, however strategy is trying to execute {side.side_name} unwind action! Unwind signal ignored! Check PositionManagementService!')
            return executed, remains

        # then it must be
        side = (Direction.DIRECTION_SHORT if exposure > 0 else Direction.DIRECTION_LONG) | Offset.OFFSET_CLOSE

        if side.sign > 0:  # short position, buy action
            working_open = working_short
            working_unwind = working_long
        else:  # long position, sell action
            working_open = working_long
            working_unwind = working_short

        if working_open:
            LOGGER.warning(f'{self} found {ticker} exposure {exposure}, still having {(-side).side_name} order {working_open}! Consider canceling these instruction before unwinding position!')

        if safe:
            unwind_volume_limit = max(abs(exposure) - abs(working_unwind), 0)

            if abs(volume) > unwind_volume_limit:
                LOGGER.warning(f'{self} found {ticker} exposure {exposure}, long order {working_long}, short order {working_short}. The unwinding signal {side.sign} {volume} exceed safe unwinding limit {unwind_volume_limit}!')

                LOGGER.info(f'{self} adjust {ticker} {side.side_name} unwind volume to {volume}, accommodating safe unwind rules!')
                volume = unwind_volume_limit

        if volume:
            self.open_pos(
                ticker=ticker,
                side=side,
                volume=abs(volume),
                limit_price=limit_price,
                algo=algo,
                **kwargs
            )
            executed += abs(volume)
            remains -= abs(volume)

        return executed, remains

    def open_pos(self, ticker: str, volume: float, side: TransactionSide = None, limit_price: float = None, algo: str = None, **kwargs):
        """
        a method to open position
        :param ticker: the given ticker
        :param volume: the target open volume
        :param side: trade side
        :param limit_price: Optional limit
        :param algo: Optional the specified algo
        :param kwargs: other keyword used in algo
        :return:
        """
        target_volume = abs(volume)

        if not target_volume:
            LOGGER.warning(f'Target open amount is {volume}, check the signal!')
            return

        if side is None:
            side = (Direction.DIRECTION_SHORT if volume > 0 else Direction.DIRECTION_LONG) | Offset.OFFSET_OPEN
            LOGGER.warning(f'Trade side of open instruction not specified! Presumed to be {side} by the sign of volume!')

        algo = self.position_tracker.open(
            ticker=ticker,
            target_volume=target_volume,
            trade_side=side,
            algo=algo,
            limit_price=limit_price,
            **kwargs
        )

        return algo

    def eod(self, market_date: datetime.date, **kwargs):

        for handler in self._on_eod:
            handler(market_date=market_date, **kwargs)

    def bod(self, market_date: datetime.date, **kwargs):

        for handler in self._on_bod:
            handler(market_date=market_date, **kwargs)

    def back_test(self, start_date: datetime.date, end_date: datetime.date, data_loader: Callable, **kwargs):
        pass

    def back_test_lite(self, start_date: datetime.date, end_date: datetime.date, data_loader: Callable, **kwargs):
        replay = ProgressReplay(
            loader=data_loader,
            start_date=start_date,
            end_date=end_date,
            bod=self.bod,
            eod=self.eod,
        )

        for ticker in self.subscription:
            replay.add_subscription(ticker, dtype='TickData')
            replay.add_subscription(ticker, dtype='TradeData')

        sim_match = {}
        multi_threading = kwargs.get('multi_threading', False)
        _start_ts = 0.
        self.event_engine.start()

        for _market_data in replay:
            _ticker = _market_data.ticker

            if not _start_ts:
                _start_ts = time.time()

            if _ticker not in sim_match:
                _ = sim_match[_ticker] = SimMatch(ticker=_ticker)
                _.register(event_engine=self.event_engine, topic_set=self.topic_set)

            if multi_threading:
                self.lock.acquire()
                self.event_engine.put(topic=self.topic_set.push(market_data=_market_data), market_data=_market_data)
            else:
                self.mds.on_market_data(market_data=_market_data)
                self.position_tracker.on_market_data(market_data=_market_data)
                self.__call__(market_data=_market_data)
                sim_match[_ticker](market_data=_market_data)

        LOGGER.info(f'All done! time_cost: {time.time() - _start_ts:,.3}s')

    def reset(self):
        self.subscription.clear()
        self._on_market_data.clear()
        self._on_report.clear()
        self._on_order.clear()
        self._on_eod.clear()
        self._on_bod.clear()

    @property
    def algos(self):
        return self.position_tracker.algos


__all__ = ['StrategyEngine', 'StrategyEngineTemplate']
