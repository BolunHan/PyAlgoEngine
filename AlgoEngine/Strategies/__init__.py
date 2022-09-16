from __future__ import annotations

import datetime
import threading
import traceback
from collections import defaultdict

from PyQuantKit import TradeInstruction, MarketData, TradeData, TickData, TransactionData, TransactionSide
from tradingsystem2_api.model import DIRECTION_LONG, DIRECTION_SHORT

from ..Engine import LOGGER
from ..Engine.EventEngine import EVENT_ENGINE, TOPIC
from ..Engine.MarketEngine import MDS
from ..Engine.TradeEngine import Balance, DirectMarketAccess, RiskProfile, PositionTracker

LOGGER = LOGGER.getChild('Strategies')


class SimDMA(DirectMarketAccess):

    def _launch_order_handler(self, order: TradeInstruction, **kwargs):
        EVENT_ENGINE.put(topic=TOPIC.launch_order(ticker=order.ticker), order=order, **kwargs)

    def _cancel_order_handler(self, order: TradeInstruction, **kwargs):
        EVENT_ENGINE.put(topic=TOPIC.cancel_order(ticker=order.ticker), order_id=order.order_id, **kwargs)

    def _reject_order_handler(self, order: TradeInstruction, **kwargs):
        raise NotImplementedError()


REPLAY_LOCK = threading.Lock()
BALANCE = Balance()
RISK_PROFILE = RiskProfile(mds=MDS, balance=BALANCE)
DMA = SimDMA(mds=MDS, risk_profile=RISK_PROFILE)
POSITION_TRACKER = PositionTracker(dma=DMA)
BALANCE.add(position_tracker=POSITION_TRACKER)

CASSANDRA_SESSION = None


class Login(object):
    def __init__(self, **kwargs):
        self.login_type = kwargs.pop('login_type', 'cassandra')
        self.cluster = kwargs.pop('cluster', ['172.31.13.10', '172.31.13.54', '172.31.13.54'])

    def cassandra_login(self):
        global CASSANDRA_SESSION

        if CASSANDRA_SESSION is not None:
            return

        try:
            from cassandra.cluster import Cluster

            cluster = Cluster(self.cluster)
            CASSANDRA_SESSION = cluster.connect('market_data')
            LOGGER.info(f'cassandra session initialized with cluster {self.cluster}')
        except Exception as _:
            raise ConnectionError(f'cassandra connection failed, {traceback.format_exc()}')

    def __call__(self, function):
        def wrapper(*args, **kwargs):
            if 'cassandra' in self.login_type:
                self.cassandra_login()
            else:
                raise ValueError(f'Invalid login type {self.login_type}')

            return function(*args, **kwargs)

        return wrapper


@Login(login_type='cassandra', cluster=['172.31.13.10', '172.31.13.54', '172.31.13.54'])
def data_loader(market_date: datetime.date, ticker: str, dtype: str) -> list[MarketData] | dict[any, MarketData]:
    from cassandra import ConsistencyLevel
    from cassandra.query import SimpleStatement

    type_table = {
        "TradeData": "transactions",
        "OrderBook": "ticks",
        "TransactionData": "orders",
        "TickData": "ticks"
    }

    return_data = []
    if dtype in type_table:
        # access cassandra
        cql_str = f"select * from {type_table[dtype]} where symbol='{ticker}' and action_day={market_date:%Y%m%d}"
        simple_statement = SimpleStatement(cql_str, consistency_level=ConsistencyLevel.ONE, fetch_size=None)
        rows = CASSANDRA_SESSION.execute(simple_statement, timeout=None)

        # process transactions data
        if dtype == "TradeData":
            for row in rows:

                # only get trade data
                # bsflag in xtp are (52, 70) filter 52;
                # bsflag in zhongjin are (32, 66, 83), filter 32 which equals direction=0
                if row.bsflag == 52 or row.direction == 0:
                    continue

                row_datetime = datetime.datetime.strptime(str(row.action_day) + str(row.time), "%Y%m%d%H%M%S%f")
                if not (datetime.time(9, 30) <= row_datetime.time() < datetime.time(11, 30) or
                        datetime.time(13) <= row_datetime.time() < datetime.time(15)):
                    continue

                side = TransactionSide.UNKNOWN
                if row.direction == DIRECTION_LONG:
                    side = TransactionSide.LongOpen
                elif row.direction == DIRECTION_SHORT:
                    side = TransactionSide.ShortOpen
                return_data.append(TradeData(ticker=ticker,
                                             trade_price=row.price,
                                             trade_volume=row.volume,
                                             trade_time=row_datetime,
                                             side=side))

        # process ticks data
        elif dtype == "TickData" or dtype == "OrderBook":
            for tick in rows:

                # skip useless data
                if tick.close == 0:
                    continue

                tick_datetime = datetime.datetime.strptime(str(tick.datetime), "%Y%m%d%H%M%S%f")
                if not (datetime.time(9, 30) <= tick_datetime.time() < datetime.time(11, 30) or
                        datetime.time(13) <= tick_datetime.time() < datetime.time(15)):
                    continue

                data = TickData(ticker=tick.symbol,
                                last_price=float(tick.close),
                                market_time=tick_datetime,
                                total_traded_volume=tick.volume,
                                total_traded_notional=tick.turnover)

                data.order_book.bid.add(price=float(tick.bid_price_1), volume=float(tick.bid_volume_1))
                data.order_book.bid.add(price=float(tick.bid_price_2), volume=float(tick.bid_volume_2))
                data.order_book.bid.add(price=float(tick.bid_price_3), volume=float(tick.bid_volume_3))
                data.order_book.bid.add(price=float(tick.bid_price_4), volume=float(tick.bid_volume_4))
                data.order_book.bid.add(price=float(tick.bid_price_5), volume=float(tick.bid_volume_5))
                data.order_book.bid.add(price=float(tick.bid_price_6), volume=float(tick.bid_volume_6))
                data.order_book.bid.add(price=float(tick.bid_price_7), volume=float(tick.bid_volume_7))
                data.order_book.bid.add(price=float(tick.bid_price_8), volume=float(tick.bid_volume_8))
                data.order_book.bid.add(price=float(tick.bid_price_9), volume=float(tick.bid_volume_9))
                data.order_book.bid.add(price=float(tick.bid_price_10), volume=float(tick.bid_volume_10))

                data.order_book.ask.add(price=float(tick.ask_price_1), volume=float(tick.ask_volume_1))
                data.order_book.ask.add(price=float(tick.ask_price_2), volume=float(tick.ask_volume_2))
                data.order_book.ask.add(price=float(tick.ask_price_3), volume=float(tick.ask_volume_3))
                data.order_book.ask.add(price=float(tick.ask_price_4), volume=float(tick.ask_volume_4))
                data.order_book.ask.add(price=float(tick.ask_price_5), volume=float(tick.ask_volume_5))
                data.order_book.ask.add(price=float(tick.ask_price_6), volume=float(tick.ask_volume_6))
                data.order_book.ask.add(price=float(tick.ask_price_7), volume=float(tick.ask_volume_7))
                data.order_book.ask.add(price=float(tick.ask_price_8), volume=float(tick.ask_volume_8))
                data.order_book.ask.add(price=float(tick.ask_price_9), volume=float(tick.ask_volume_9))
                data.order_book.ask.add(price=float(tick.ask_price_10), volume=float(tick.ask_volume_10))
                return_data.append(data)

            if dtype == "OrderBook":
                return [each.order_book for each in return_data]

        # process orders data
        elif dtype == "TransactionData":
            for row in rows:

                # skip useless data
                row_datetime = datetime.datetime.strptime(str(row.action_day) + str(row.time), "%Y%m%d%H%M%S%f")
                if not (datetime.time(9, 30) <= row_datetime.time() < datetime.time(11, 30) or
                        datetime.time(13) <= row_datetime.time() < datetime.time(15)):
                    continue

                side = TransactionSide.UNKNOWN
                if row.direction == DIRECTION_LONG:
                    side = TransactionSide.LongOrder
                elif row.direction == DIRECTION_SHORT:
                    side = TransactionSide.ShortOrder
                return_data.append(TransactionData(ticker=ticker,
                                                   price=row.price,
                                                   volume=row.volume,
                                                   transaction_time=row_datetime,
                                                   side=side))

    return return_data

