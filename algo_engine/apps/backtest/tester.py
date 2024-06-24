import abc
import datetime
import threading
import time
from typing import Literal

import numpy as np
from event_engine import EventEngine

from .metrics import TradeMetrics
from .web_app import WebApp
from ...back_test import SimMatch, ProgressiveReplay
from ...base import MarketData, TradeReport, TradeInstruction
from ...engine import EVENT_ENGINE, TOPIC
from ...profile import Profile, PROFILE
from . import LOGGER


class Tester(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            start_date: datetime.date,
            end_date: datetime.date,
            dtype: list[str] = None,
            profile: Profile = None,
            event_engine: EventEngine = None,
            topic_set: EventEngine = None,
            multi_threading: bool = False,
            **kwargs
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.dtype = ['TickData', 'TradeData'] if dtype is None else dtype
        self.profile = PROFILE if profile is None else profile
        self.event_engine = EVENT_ENGINE if event_engine is None else event_engine
        self.topic_set = TOPIC if topic_set is None else topic_set
        self.multi_threading = multi_threading
        self.lock = threading.Lock()

        self.timestamp = 0.
        self.last_price = {}
        self.subscription = set()
        self.web_app = WebApp(start_date=start_date, end_date=end_date, **kwargs)
        self.metrics: dict[str, TradeMetrics] = {}
        self.sim_match: dict[str, SimMatch] = {}

    def register_ticker(self, ticker: str, **kwargs):
        self.subscription.add(ticker)

        self.metrics[ticker] = TradeMetrics()

        self.web_app.register(ticker=ticker, **kwargs)

        sim_match = self.sim_match[ticker] = SimMatch(
            ticker=ticker,
            instant_fill=kwargs.get('instant_fill', True),
            event_engine=self.event_engine
        )
        sim_match.register()

    def unregister_ticker(self, ticker: str, **kwargs):
        self.subscription.remove(ticker)

        self.metrics.pop(ticker)

        # the web app does not provide an unregister method, however, this is not a requirement

        sim_match = self.sim_match.pop(ticker)
        sim_match.unregister()

    def _launch_order(self, ticker: str, volume: float, limit_price: float):
        order = TradeInstruction(ticker=ticker, side=np.sign(volume), volume=abs(float), timestamp=self.timestamp)

        if self.multi_threading:
            self.event_engine.put(topic=self.topic_set.launch_order, order=order)
        else:
            self.sim_match[ticker].launch_order(order=order)

    def buy(self, ticker: str, volume: float = None, limit_price: float = None):
        if volume is None:
            trade_metrics = self.metrics[ticker]
            exposure = trade_metrics.exposure
            volume = -exposure if exposure < 0 else 1

        if limit_price is None:
            limit_price = self.last_price[ticker]

        self._launch_order(ticker=ticker, volume=volume, limit_price=limit_price)

    def sell(self, ticker: str, volume: float = None, limit_price: float = None):
        if volume is None:
            trade_metrics = self.metrics[ticker]
            exposure = trade_metrics.exposure
            volume = -exposure if exposure > 0 else -1

        if limit_price is None:
            limit_price = self.last_price[ticker]

        self._launch_order(ticker=ticker, volume=volume, limit_price=limit_price)

    def register(self):
        self.event_engine.register_handler(topic=self.topic_set.realtime, handler=self.on_market_data)
        self.event_engine.register_handler(topic=self.topic_set.on_order, handler=self.on_order)
        self.event_engine.register_handler(topic=self.topic_set.on_report, handler=self.on_report)

    def unregister(self):
        self.event_engine.unregister_handler(topic=self.topic_set.realtime, handler=self.on_market_data)
        self.event_engine.unregister_handler(topic=self.topic_set.on_order, handler=self.on_order)
        self.event_engine.unregister_handler(topic=self.topic_set.on_report, handler=self.on_report)

    @abc.abstractmethod
    def load_data(self, ticker: str, dtype: Literal['TickData', 'TradeData', 'TransactionData', 'OrderBook']) -> list[MarketData]:
        ...

    @abc.abstractmethod
    def on_market_data(self, market_data: MarketData, **kwargs):
        ...

    @abc.abstractmethod
    def on_report(self, report: TradeReport, **kwargs):
        ...

    @abc.abstractmethod
    def on_order(self, order: TradeInstruction, **kwargs):
        ...

    def bod(self, market_date: datetime.date, **kwargs):
        pass

    def eod(self, market_date: datetime.date, **kwargs):
        pass

    def run(self, **kwargs):
        replay = ProgressiveReplay(
            loader=self.load_data,
            tickers=list(self.subscription),
            dtype=['TickData', 'TradeData'],
            start_date=self.start_date,
            end_date=self.end_date,
            bod=self.bod,
            eod=self.eod,
            tick_size=kwargs.get('progress_tick_size', 0.001),
        )

        if not self.event_engine.active:
            self.event_engine.start()

        _start_ts = time.time()

        for market_data in replay:

            if self.multi_threading:
                self.lock.acquire()
                self.event_engine.put(topic=self.topic_set.push(market_data=market_data), market_data=market_data)
            else:
                self.on_market_data(market_data=market_data)
                self.sim_match[market_data.ticker](market_data=market_data)

            self.timestamp = market_data.timestamp
            self.last_price[market_data.ticker] = market_data.market_price

        LOGGER.info(f'All done! time_cost: {time.time() - _start_ts:,.3}s')
