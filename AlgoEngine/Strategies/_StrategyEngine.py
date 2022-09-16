from __future__ import annotations

import abc
import datetime
import threading
import time
from collections import defaultdict

from PyQuantKit import MarketData, TradeReport, TradeInstruction, TransactionSide

from ..Engine import PositionTracker, TOPIC, EVENT_ENGINE, SimMatch, LOGGER, MDS

LOGGER = LOGGER.getChild('StrategyEngine')


class StrategyEngineTemplate(object, metaclass=abc.ABCMeta):
    def __init__(self, position_tracker: PositionTracker):
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


class StrategyEngine(StrategyEngineTemplate):
    def __init__(self, position_tracker: PositionTracker, **kwargs):
        super().__init__(position_tracker=position_tracker)

        self.mds = kwargs.pop('mds', MDS)
        self.multi_threading = kwargs.pop('multi_threading', False)
        self.lock = threading.Lock()
        self.event_engine = kwargs.pop('event_engine', EVENT_ENGINE)
        self.topic_set = kwargs.pop('topic_set', TOPIC)

        self._on_market_data = []
        self._on_report = []
        self._on_order = []
        self._on_eod = []
        self._on_bod = []
        self.subscription = set()
        self.trade_pos = {}

        self.attach_strategy(strategy=kwargs.pop('strategy', None))
        self.add_handler(**kwargs)

    def __call__(self, **kwargs):
        if 'market_data' in kwargs:
            self.on_market_data(market_data=kwargs['market_data'])

        if self.multi_threading and self.lock.locked():
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

    def register(self, event_engine=None, topic_set=None):
        if event_engine is None:
            event_engine = self.event_engine

        if topic_set is None:
            topic_set = self.topic_set

        event_engine.register_handler(topic=topic_set.realtime, handler=self.__call__)
        event_engine.register_handler(topic=topic_set.on_order, handler=self.on_order)
        event_engine.register_handler(topic=topic_set.on_report, handler=self.on_report)

    def cancel(self, ticker: str, side: TransactionSide = None, pos_id: str = None, trade_id: str = None, **kwargs):
        position_tracker = self.position_tracker

        if trade_id is not None:
            pos_id = position_tracker.reversed_handler_mapping.get(trade_id).pos_id

        if not pos_id:
            LOGGER.warning(f'No pos_id given! Cancel all trade_handler with ticker {ticker}')

            for _pos_id in list(self.trade_pos):
                trade_pos = self.trade_pos.get(_pos_id)

                if trade_pos is None:
                    continue

                for _trade_id in list(trade_pos.trade_handler):
                    trade_handler = trade_pos.trade_handler.get(_trade_id)

                    if trade_handler is None:
                        continue

                    if side is not None and side.sign != trade_handler.side.sign:
                        continue

                    if trade_handler.ticker == ticker:
                        trade_pos.cancel(trade_id=_trade_id, **kwargs)
        else:
            trade_pos = self.trade_pos.get(pos_id)

            if trade_pos is None:
                LOGGER.error(f'PositionTracker contains no TradePos with pos_id {pos_id}! Cancel signal ignored! Manual intervention required!')
                return

            if trade_id is None:
                LOGGER.info(f'No trade_id given! Cancel all trade_handler with ticker {ticker}')

                for _trade_id in list(trade_pos.trade_handler):
                    trade_handler = trade_pos.trade_handler.get(_trade_id)

                    if trade_handler is None:
                        continue

                    if side is not None and side.sign != trade_handler.side.sign:
                        continue

                    if trade_handler.ticker == ticker:
                        trade_pos.cancel(trade_id=_trade_id, **kwargs)
            else:
                trade_pos.cancel(trade_id=trade_id, **kwargs)

    def stop(self):
        LOGGER.info(f'{self} stopping and canceling working orders')

        for pos_id in list(self.trade_pos):
            trade_pos = self.trade_pos.get(pos_id)

            if trade_pos is None:
                continue

            for handler_id in list(trade_pos.trade_handler):
                trade_handler = trade_pos.get(handler_id)

                if trade_handler is None:
                    continue

                for algo_id in list(trade_handler.algo):
                    algo = trade_handler.algo.get(algo_id)

                    if algo is None:
                        continue

                    algo.is_active = False

        for ticker in self.subscription:
            self.cancel(ticker=ticker)

    def unwind_pos(self, ticker: str, side: TransactionSide, volume: float, limit: float = None, algo: str = None, pos_id: str = None, trade_id: str = None, allowed_remains=True, **kwargs) -> float:
        position_tracker = self.position_tracker

        if trade_id is not None:
            pos_id = position_tracker.reversed_handler_mapping.get(trade_id).pos_id

        if pos_id is None:
            remains = to_unwind = abs(volume)

            for _pos_id in list(self.trade_pos):
                trade_pos = self.trade_pos.get(_pos_id)

                if trade_pos is None:
                    continue

                remains = trade_pos.unwind_auto(ticker=ticker, side=side, volume=to_unwind, algo=algo, limit_price=limit, **kwargs)

                if not remains:
                    break

                to_unwind = remains
        else:
            trade_pos = self.trade_pos.get(pos_id)

            if trade_pos is None:
                LOGGER.error(f'PositionTracker contains no TradePos with pos_id {pos_id}! Unwind signal ignored! Manual intervention required!')

            if trade_id is None:
                to_unwind = abs(volume)

                remains = trade_pos.unwind_auto(ticker=ticker, side=side, volume=to_unwind, algo=algo, limit_price=limit, **kwargs)

                if remains:
                    LOGGER.warning(f'{remains} of {ticker} not distributed!')
            else:
                trade_handler = trade_pos.trade_handler.get(trade_id)
                additional_kwargs = {}
                remains = 0.

                if trade_handler is None:
                    LOGGER.error(f'TradePos {trade_pos} contains no TradeHandler with trade_id {trade_id}! Unwind signal ignored! Manual intervention required!')
                elif trade_handler.ticker != ticker:
                    LOGGER.error(f'Invalid Ticker of the Unwind Instruction! Expect {trade_handler.ticker}, got {ticker}. Manual intervention required!')
                elif side.sign == trade_handler.side.sign:
                    LOGGER.error(f'Invalid TransactionSide of the Unwind Instruction! Expect {-trade_handler.side}, got {side.sign}. Error ignored and execute as unwind!')
                else:
                    additional_kwargs['target_volume'] = volume
                    additional_kwargs['algo'] = algo
                    additional_kwargs['limit_price'] = limit

                    trade_pos.unwind(trade_id=trade_id, additional_kwargs=kwargs, **kwargs)

        if not allowed_remains:
            self.open_pos(
                ticker=ticker,
                side=side,
                volume=abs(remains),
                limit=limit,
                algo=algo,
            )
            remains = 0.
        else:
            LOGGER.warning(f'{ticker} remaining unwind action {remains} not sent!')

        return remains

    def open_pos(self, ticker: str, side: TransactionSide, volume: float, limit: float = None, algo: str = None, pos_id: str = None, trade_id: str = None, **kwargs):
        """
        open position for given ticker, side volume with given algo
        """
        position_tracker = self.position_tracker
        additional_kwargs = defaultdict(dict)

        trade_side = side
        target_volume = volume

        if not target_volume:
            return

        if pos_id is None:
            trade_pos = position_tracker.add_pos()
        else:
            trade_pos = position_tracker.add_pos(pos_id=pos_id)

        _trade_id = trade_pos.add_target(ticker=ticker, side=trade_side, target_volume=target_volume, trade_id=trade_id)
        additional_kwargs[_trade_id]['target_volume'] = target_volume
        additional_kwargs[_trade_id]['algo'] = algo
        additional_kwargs[_trade_id]['limit_price'] = limit

        LOGGER.info(f"Signal opening [{', '.join([f'<{handler.ticker}, {handler.side.name}, {handler.target_volume}>' for handler in trade_pos.trade_handler.values()])}] position!")
        self.trade_pos[trade_pos.pos_id] = trade_pos
        trade_pos.open(additional_kwargs=additional_kwargs, **kwargs)

        return trade_pos

    def eod(self, **kwargs):

        for handler in self._on_eod:
            handler(**kwargs)

    def bod(self, **kwargs):

        for handler in self._on_bod:
            handler(**kwargs)

    def back_test(self, start_date: datetime.date, end_date: datetime.date, data_loader: callable, **kwargs):
        from ..Engine import ProgressiveReplay

        replay = ProgressiveReplay(
            loader=data_loader,
            tickers=list(self.subscription),
            dtype=['TickData', 'TradeData'],
            start_date=start_date,
            end_date=end_date,
            bod=self.bod,
            eod=self.eod,
            tick_size=kwargs.get('progress_tick_size', 0.001),
        )

        sim_match = {}
        _start_ts = 0.
        self.event_engine.start()

        for _market_data in replay:
            _ticker = _market_data.ticker

            if not _start_ts:
                _start_ts = time.time()

            if _ticker not in sim_match:
                _ = sim_match[_ticker] = SimMatch(ticker=_ticker)
                _.register(event_engine=self.event_engine, topic_set=self.topic_set)

            if self.multi_threading:
                self.lock.acquire()
                self.event_engine.put(topic=self.topic_set.push(market_data=_market_data), market_data=_market_data)
            else:
                self.mds.on_market_data(market_data=_market_data)
                self.position_tracker.on_market_data(market_data=_market_data)
                self.__call__(market_data=_market_data)
                sim_match[_ticker](market_data=_market_data)

        LOGGER.info(f'All done! time_cost: {time.time() - _start_ts:,.3}s')

    def get_working_orders(self, side: str):  # "long" or "short"
        current_working_trades = self.trade_pos  # returns a Dict[id_str: TradePos] TradePos object is implemented at .Engine.TradeEngine
        exposure_volume = 0.0
        working_volume = 0.0
        pending_volume = 0.0

        for pos_id in current_working_trades:
            trade_pos = current_working_trades.get(pos_id)

            if trade_pos is None:
                continue

            for key in trade_pos.exposure_volume:
                if side == "long":
                    if trade_pos.exposure_volume[key] > 0:
                        exposure_volume += trade_pos.exposure_volume[key]
                if side == "short":
                    if trade_pos.exposure_volume[key] < 0:
                        exposure_volume += trade_pos.exposure_volume[key]

            for trade_handler in trade_pos.trade_handler.values():
                for algo in trade_handler.algo.values():
                    if not algo.is_active:
                        continue

                    algo_target = abs(algo.target_volume)
                    algo_working = abs(algo.working_volume)
                    algo_exposure = abs(algo.exposure_volume)
                    algo_pending = max(0., algo_target - algo_working - algo_exposure)

                    if side == 'long' and algo.side.sign > 0:
                        working_volume += algo_working
                        pending_volume += algo_pending
                    elif side == 'short' and algo.side.sign < 0:
                        working_volume += algo_working
                        pending_volume += algo_pending

        result = {"exposure_volume": exposure_volume, "working_volume": working_volume, 'pending_volume': pending_volume}
        return result

    def reset(self):
        self.subscription.clear()
        self._on_market_data.clear()
        self._on_report.clear()
        self._on_order.clear()
        self._on_eod.clear()
        self._on_bod.clear()


__all__ = ['StrategyEngine', 'StrategyEngineTemplate']
