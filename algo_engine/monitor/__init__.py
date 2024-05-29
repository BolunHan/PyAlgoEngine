from ..engine import MDS
from ..engine.market_engine import MarketDataMonitor as Monitor


def add_synthetic_orderbook():
    # init synthetic orderbook monitor
    monitor = SyntheticOrderBookMonitor(mds=MDS)
    MDS.add_monitor(monitor=monitor)
    # override current orderbook
    MDS._order_book = monitor.order_book


from .advanced_data_interface import *

__all__ = ['Monitor', 'SyntheticOrderBookMonitor', 'MinuteBarMonitor']
