import datetime

import numpy as np

from . import LOGGER
from ..base import OrderType, MarketData, BarData, TradeData, TickData, OrderState, OrderBook, TradeReport, TradeInstruction, TransactionSide
from ..engine.event_engine import TOPIC, EVENT_ENGINE
from ..profile import PROFILE

LOGGER = LOGGER.getChild('SimMatch')


class SimMatch(object):
    def __init__(self, ticker, event_engine=None, topic_set=None, **kwargs):
        self.ticker = ticker
        self.event_engine = event_engine if event_engine is not None else EVENT_ENGINE
        self.topic_set = topic_set if topic_set is not None else TOPIC
        self.fee_rate = kwargs.get('fee_rate', 0.)

        self.working: dict[str, TradeInstruction] = {}
        self.history: dict[str, TradeInstruction] = {}

        self.timestamp = 0.
        self.last_price = None
        self.matching_config = {
            'instant_fill': kwargs.get('instant_fill', False),
            'lag': {
                'ts': kwargs.get('lag_ts', 0.),
                'n_transaction': kwargs.get('lag_n_transaction', 0)
            },
            'hit_prob': kwargs.get('hit_prob', 1.),  # affecting FoK
            'slippery_rate': kwargs.get('slippery_rate', 0.0001)
        }

    def __call__(self, **kwargs):
        order: TradeInstruction | None = kwargs.pop('order', None)
        market_data: MarketData | None = kwargs.pop('market_data', None)

        if order is not None:
            if order.order_type == OrderType.LimitOrder:
                self.launch_order(order=order)
            elif order.order_type == OrderType.CancelOrder:
                self.cancel_order(order=order)
            else:
                raise ValueError(f'Invalid order {order}')

        if market_data is not None:
            self.timestamp = market_data.timestamp
            self.last_price = market_data.market_price

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

        order.set_order_state(order_state=OrderState.Placed, timestamp=self.timestamp)
        short_circuit = self._check_short_circuit(order=order)

        if short_circuit:
            self.on_order(order=order, **kwargs)
            # in short circuit mode, the worst price will be applied.
            self._match(order=order, match_price=self.worst_price(order.limit_price, self.last_price, side=order.side))

        self.working[order.order_id] = order
        self.on_order(order=order, **kwargs)

    @classmethod
    def best_price(cls, *price, side: TransactionSide | int):
        if side > 0:
            return min(_ for _ in price if _ is not None and np.isfinite(_))
        elif side < 0:
            return max(_ for _ in price if _ is not None and np.isfinite(_))

        raise ValueError(f'Invalid side {side}!')

    @classmethod
    def worst_price(cls, *price, side: TransactionSide | int):
        if side > 0:
            return max(_ for _ in price if _ is not None and np.isfinite(_))
        elif side < 0:
            return min(_ for _ in price if _ is not None and np.isfinite(_))

        raise ValueError(f'Invalid side {side}!')

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
            order.set_order_state(order_state=OrderState.Canceled, timestamp=self.timestamp)
            LOGGER.info(f'[{self.market_time:%Y-%m-%d %H:%M:%S}] Sim-canceled {order.side.name} {order.ticker} order!')

        self.history[order_id] = order
        self.on_order(order=order, **kwargs)

    def _check_short_circuit(self, order: TradeInstruction, **kwargs):
        if order.limit_price is None and self.last_price is None:
            return False

        if self.matching_config['instant_fill'] and all(not _ for _ in self.matching_config['lag'].values()):
            return True

        return False

    def _check_bar_data(self, market_data: BarData):
        for order_id in list(self.working):
            order = self.working.get(order_id)
            if order is None:
                pass
            elif order.order_state in [OrderState.Placed, OrderState.PartFilled]:
                if order.side.sign > 0:
                    # match order based on worst offer
                    if order.limit_price is None:
                        self._match(order=order, match_price=market_data.vwap)
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
                        self._match(order=order, match_price=market_data.vwap)
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
        for order_id in list(self.working):
            order = self.working.get(order_id)

            if order is None:
                pass
            elif order.order_state in [OrderState.Placed, OrderState.PartFilled]:
                if order.limit_price is None:
                    self._match(order=order, match_volume=order.working_volume, match_price=market_data.market_price)
                elif order.side.sign > 0 and market_data.market_price <= order.limit_price:
                    self._match(order=order, match_volume=order.working_volume, match_price=market_data.market_price)
                elif order.side.sign < 0 and market_data.market_price >= order.limit_price:
                    self._match(order=order, match_volume=order.working_volume, match_price=market_data.market_price)
                else:
                    continue
            else:
                continue

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
                timestamp=self.timestamp,
                order_id=order.order_id,
                price=match_price,
                multiplier=order.multiplier,
                fee=self.fee_rate * match_volume * match_price * order.multiplier
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
        self.working.clear()
        self.history.clear()

        self.timestamp = 0.
        self.last_price = None

    @property
    def market_time(self) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(self.timestamp, tz=PROFILE.time_zone)
