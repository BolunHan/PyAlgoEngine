import datetime
import random

import numpy as np

from . import LOGGER
from ..base import OrderType, MarketData, BarData, TransactionData, TradeData, TickData, TickDataLite, OrderState, OrderData, TradeReport, TradeInstruction, TransactionSide, TransactionDirection
from ..engine.event_engine import TOPIC, EVENT_ENGINE
from ..profile import PROFILE

LOGGER = LOGGER.getChild('SimMatch')


class SimMatch(object):
    def __init__(self, ticker: str, event_engine=None, topic_set=None, seed: int = None, **kwargs):
        self.ticker = ticker
        self.event_engine = event_engine if event_engine is not None else EVENT_ENGINE
        self.topic_set = topic_set if topic_set is not None else TOPIC

        self.working: dict[str, TradeInstruction] = {}
        self.history: dict[str, TradeInstruction] = {}

        self.timestamp: float = 0.
        self.last_price: float | None = None
        self.last_transaction_count: int = 0
        self.seed = seed
        self.random = random.Random(self.seed)

        self.matching_config = {
            'fee_rate': kwargs.get('fee_rate', 0.),
            'instant_fill': kwargs.get('instant_fill', False),
            'lag': {
                'ts': kwargs.get('lag_ts', 0.),  # time lag in seconds
                'n_transaction': kwargs.get('lag_n_transaction', 0)  # number of transactions lag
            },
            'hit': {
                'prob': kwargs.get('hit_prob', 1.),  # probability of order being filled
                'slippery': kwargs.get('slippery_rate', 0.0001)  # slippage rate
            }
        }

    def __call__(self, **kwargs):
        order: TradeInstruction | None = kwargs.pop('order', None)
        market_data: MarketData | None = kwargs.pop('market_data', None)

        if order is not None:
            if order.order_type == OrderType.ORDER_LIMIT:
                self.launch_order(order=order)
            elif order.order_type == OrderType.ORDER_CANCEL:
                self.cancel_order(order=order)
            else:
                raise ValueError(f'Invalid order {order}')

        if market_data is not None:
            self.timestamp = market_data.timestamp
            self.last_price = market_data.market_price

            # Update transaction count if this is a transaction
            if isinstance(market_data, (TransactionData, TradeData)):
                self.last_transaction_count += 1

            if isinstance(market_data, BarData):
                self._check_bar_data(market_data=market_data)
            elif isinstance(market_data, TickData):
                self._check_tick_data(market_data=market_data)
            elif isinstance(market_data, TickDataLite):
                self._check_tick_data_lite(market_data=market_data)
            elif isinstance(market_data, OrderData):
                self._check_order_data(market_data=market_data)
            elif isinstance(market_data, (TransactionData, TradeData)):
                self._check_trade_data(market_data=market_data)

    @staticmethod
    def best_price(*price: float, side: TransactionSide | TransactionDirection) -> float:
        """Get best price for the given side."""
        valid_prices = [p for p in price if p is not None and np.isfinite(p)]
        sign = side.sign

        if not valid_prices:
            raise ValueError("No valid prices provided")

        if sign == 1:
            return min(valid_prices)
        elif sign == -1:
            return max(valid_prices)
        else:
            raise ValueError(f'Invalid side {side}!')

    @staticmethod
    def worst_price(*price: float, side: TransactionSide | TransactionDirection) -> float:
        """Get worst price for the given side."""
        valid_prices = [p for p in price if p is not None and np.isfinite(p)]
        sign = side.sign

        if not valid_prices:
            raise ValueError("No valid prices provided")

        if sign == 1:
            return max(valid_prices)
        elif sign == -1:
            return min(valid_prices)
        else:
            raise ValueError(f'Invalid side {side}!')

    def _apply_lag(self, order: TradeInstruction) -> bool:
        """Check if order should be processed considering lag settings."""
        lag_config = self.matching_config['lag']
        lag_ts = lag_config['ts']
        lag_n_transaction = lag_config['n_transaction']

        # No lag configured
        if not lag_ts and not lag_n_transaction:
            return True

        # Check time lag
        time_elapsed = self.timestamp - order.timestamp
        if lag_ts > 0 and time_elapsed < lag_ts:
            return False

        # Check transaction lag
        transactions_since_order = self.last_transaction_count - order._additional.get('transaction_count_at_placement', 0)
        if lag_n_transaction > 0 and transactions_since_order < lag_n_transaction:
            return False

        return True

    def _apply_hit_probability(self, order: TradeInstruction) -> bool:
        """Determine if order should be filled based on hit probability."""
        hit_config = self.matching_config['hit']
        hit_prob = hit_config['prob']

        if hit_prob >= 1.0:
            return True

        return random.random() < hit_prob

    def _apply_slippage(self, price: float, side: TransactionSide | TransactionDirection) -> float:
        """Apply slippage to the execution price."""
        hit_config = self.matching_config['hit']
        slippery_rate = hit_config['slippery']

        slippage = price * slippery_rate
        sign = side.sign

        if sign == 1:
            # For buy-orders, slippage increases the price
            return price + slippage
        elif sign == -1:
            # For sell-orders, slippage decreases the price
            return price - slippage
        else:
            return price

    def _check_short_circuit(self, order: TradeInstruction) -> bool:
        """Check if order should be filled immediately."""
        if order.limit_price is None and self.last_price is None:
            return False

        # Check if instant fill is enabled and no lag is configured
        if self.matching_config['instant_fill'] and all(not _ for _ in self.matching_config['lag'].values()):
            return True

        return False

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
        if order.order_id in self.working or order.order_id in self.history:
            raise ValueError(f'Invalid instruction {order}, OrderId already in working or history')

        if order.limit_price is None and order.order_type == OrderType.ORDER_LIMIT:
            LOGGER.warning(f'order {order} does not have a valid limit price!')

        order.set_order_state(order_state=OrderState.STATE_PLACED, timestamp=self.timestamp)
        order.transaction_count_at_placement = self.last_transaction_count

        # Check for immediate fill conditions
        if self._check_short_circuit(order=order):
            self.on_order(order=order, **kwargs)
            worst_price = self.worst_price(
                order.limit_price if order.limit_price is not None else self.last_price,
                self.last_price,
                side=order.side
            )
            self._match(order=order, match_price=worst_price)

        self.working[order.order_id] = order
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

        if order.order_state == OrderState.STATE_FILLED:
            pass
        else:
            order.set_order_state(order_state=OrderState.STATE_CANCELED, timestamp=self.timestamp)
            LOGGER.info(f'[{self.market_time:%Y-%m-%d %H:%M:%S}] Sim-canceled {order.side.name} {order.ticker} order!')

        self.history[order_id] = order
        self.on_order(order=order, **kwargs)

    def _check_bar_data(self, market_data: BarData):
        for order_id in list(self.working):
            order = self.working.get(order_id)
            if order is None:
                continue

            if not order.is_working:
                continue

            if order.start_time > market_data.market_time:
                continue

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

    def _check_trade_data(self, market_data: TransactionData | TradeData):
        for order_id in list(self.working):
            order = self.working.get(order_id)
            if order is None:
                continue

            if not order.is_working:
                continue

            if order.start_time > market_data.market_time:
                continue

            if order.limit_price is None:
                if order.side.sign * market_data.side.sign > 0:  # copy the next transaction info
                    self._match(order=order, match_volume=market_data.volume, match_price=market_data.price)
            elif order.side.sign > 0 and market_data.market_price < order.limit_price:
                self._match(order=order, match_volume=market_data.volume, match_price=market_data.price)
            elif order.side.sign < 0 and market_data.market_price > order.limit_price:
                self._match(order=order, match_volume=market_data.volume, match_price=market_data.price)

    def _check_tick_data(self, market_data: TickData):
        for order_id in list(self.working):
            order = self.working.get(order_id)
            if order is None:
                continue

            if not order.is_working:
                continue

            if order.start_time > market_data.market_time:
                continue

            match_volume = 0.
            match_notional = 0.

            if order.limit_price is None:
                if order.side.sign > 0:
                    for entry in market_data.ask:
                        price, volume, _ = entry

                        if match_volume < order.working_volume:
                            addition_volume = min(volume, order.working_volume - match_volume)
                            match_volume += addition_volume
                            match_notional += addition_volume * price
                        else:
                            break
                else:
                    for entry in market_data.bid:
                        price, volume, _ = entry

                        if match_volume < order.working_volume:
                            addition_volume = min(volume, order.working_volume - match_volume)
                            match_volume += addition_volume
                            match_notional += addition_volume * price
                        else:
                            break
            elif order.side.sign > 0 and market_data.best_ask_price <= order.limit_price:
                for entry in market_data.ask:
                    price, volume, _ = entry

                    if price <= order.limit_price:
                        if match_volume < order.working_volume:
                            addition_volume = min(volume, order.working_volume - match_volume)
                            match_volume += addition_volume
                            match_notional += addition_volume * price
                        else:
                            break
                    else:
                        break
            elif order.side.sign < 0 and market_data.best_bid_price >= order.limit_price:
                for entry in market_data.bid:
                    price, volume, _ = entry

                    if price >= order.limit_price:
                        if match_volume < order.working_volume:
                            addition_volume = min(volume, order.working_volume - match_volume)
                            match_volume += addition_volume
                            match_notional += addition_volume * price
                        else:
                            break
                    else:
                        break

            if match_volume:
                self._match(order=order, match_volume=match_volume, match_price=match_notional / match_volume)

    def _check_order_data(self, market_data: OrderData) -> None:
        """Process order data from the market.

        Args:
            market_data: The incoming order data from the market.
        """
        for order_id in list(self.working):
            order = self.working.get(order_id)
            if order is None:
                continue

            if not order.is_working:
                continue

            if order.start_time > market_data.market_time:
                continue

            # Check if this order matches our working order
            match_volume = 0.
            match_notional = 0.

            if order.limit_price is None:
                if order.side.sign > 0 and market_data.side.sign < 0:
                    match_volume = market_data.volume
                    match_notional = market_data.price * market_data.volume
                elif order.side.sign < 0 and market_data.side.sign > 0:
                    match_volume = market_data.volume
                    match_notional = market_data.price * market_data.volume
            elif order.side.sign > 0 and market_data.price <= order.limit_price:
                match_volume = market_data.volume
                match_notional = market_data.price * market_data.volume
            elif order.side.sign < 0 and market_data.price >= order.limit_price:
                match_volume = market_data.volume
                match_notional = market_data.price * market_data.volume

            if match_volume:
                self._match(order=order, match_volume=match_volume, match_price=match_notional / match_volume)

    def _check_tick_data_lite(self, market_data: TickDataLite) -> None:
        """Process simplified tick data from the market.

        Args:
            market_data: The incoming tick data (lite version) from the market.
        """
        for order_id in list(self.working):
            order = self.working.get(order_id)

            if order is None:
                continue

            if not order.is_working:
                continue

            if order.start_time > market_data.market_time:
                continue

            # Check if this order matches our working order
            match_volume = 0.
            match_notional = 0.

            if order.limit_price is None:
                if order.side.sign > 0:
                    match_volume = market_data.ask_volume
                    match_notional = market_data.ask_price * market_data.ask_volume
                elif order.side.sign < 0:
                    match_volume = market_data.bid_volume
                    match_notional = market_data.bid_price * market_data.bid_volume
            elif order.side.sign > 0 and market_data.ask_price <= order.limit_price:
                match_volume = market_data.ask_volume
                match_notional = market_data.ask_price * market_data.ask_volume
            elif order.side.sign < 0 and market_data.bid_price >= order.limit_price:
                match_volume = market_data.bid_volume
                match_notional = market_data.bid_price * market_data.bid_volume

            if match_volume:
                self._match(order=order, match_volume=match_volume, match_price=match_notional / match_volume)

    def _match(self, order: TradeInstruction, match_volume: float = None, match_price: float = None) -> TradeReport | None:
        """Attempt to match an order with the given volume and price."""
        # Apply lag check
        if not self._apply_lag(order):
            return None

        # Apply hit probability
        if not self._apply_hit_probability(order):
            return None

        # Determine match volume
        if match_volume is None:
            match_volume = order.working_volume
        else:
            match_volume = min(match_volume, order.working_volume)

        # Determine match price with slippage
        if match_price is None and order.limit_price is not None:
            match_price = order.limit_price
        elif match_price is not None:
            match_price = self._apply_slippage(match_price, order.side)

        # Validate price against limit
        if order.limit_price is not None:
            if order.side.sign > 0 and match_price > order.limit_price:
                LOGGER.warning(f'match price greater than limit price for bid order {order}')
                match_price = order.limit_price
            elif order.side.sign < 0 and match_price < order.limit_price:
                LOGGER.warning(f'match price less than limit price for ask order {order}')
                match_price = order.limit_price

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
                fee=self.matching_config['fee_rate'] * match_volume * match_price * order.multiplier
            )

            LOGGER.info(f'[{self.market_time:%Y-%m-%d %H:%M:%S}] Sim-filled {order.ticker} {order.side.side_name} {report.volume:,.2f} @ {report.price:.2f}')
            order.fill(trade_report=report)

            if order.order_state == OrderState.STATE_FILLED:
                self.working.pop(order.order_id, None)
                self.history[order.order_id] = order

            self.on_report(report=report)
            self.on_order(order=order)
            return report

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
        self.last_transaction_count = 0
        self.random = random.Random(self.seed)

    @property
    def market_time(self) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(self.timestamp, tz=PROFILE.time_zone)
